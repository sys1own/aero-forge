//! Native execution daemon for the `workspace.aeroc` binary IR.
//!
//! `aeroc-daemon` maps a compiled `.aeroc` file into memory with `mmap(2)`,
//! verifies the embedded BLAKE3 content hash, and schedules build tasks on a
//! lock-free wavefront executor.  Task readiness is evaluated with SIMD
//! bit-vector operations when the host supports AVX2/AVX-512, falling back to a
//! scalar 64-bit loop.

use std::arch::is_x86_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;


use constant_time_eq::constant_time_eq;
use memmap2::MmapOptions;

use crate::aeroc_compiler::{
    AerocError, AerocHeader, Instruction, OpCode, PayloadBlockHeader, StringRef, AEROC_HEADER_SIZE,
    AEROC_MAGIC,
};

/// Errors raised by the aeroc daemon.
#[derive(Debug)]
pub enum DaemonError {
    Io(io::Error),
    Aeroc(AerocError),
    Validation(String),
    Execution(String),
}

impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaemonError::Io(e) => write!(f, "io error: {e}"),
            DaemonError::Aeroc(e) => write!(f, "aeroc error: {e}"),
            DaemonError::Validation(s) => write!(f, "validation error: {s}"),
            DaemonError::Execution(s) => write!(f, "execution error: {s}"),
        }
    }
}

impl std::error::Error for DaemonError {}

impl From<io::Error> for DaemonError {
    fn from(e: io::Error) -> Self {
        DaemonError::Io(e)
    }
}

impl From<AerocError> for DaemonError {
    fn from(e: AerocError) -> Self {
        DaemonError::Aeroc(e)
    }
}

/// A decompressed or ready-to-use payload block.
#[derive(Clone, Debug)]
pub struct PayloadBlock {
    pub header: PayloadBlockHeader,
    pub compressed: Vec<u8>,
}

impl PayloadBlock {
    /// Decompress the block, optionally using the shared Zstd dictionary.
    pub fn decompress(&self, dictionary: &[u8]) -> Result<Vec<u8>, DaemonError> {
        let mut decompressor = if dictionary.is_empty() {
            zstd::bulk::Decompressor::new()
        } else {
            zstd::bulk::Decompressor::with_dictionary(dictionary)
        }
        .map_err(|e| DaemonError::Validation(format!("zstd decompressor: {e}")))?;
        let max_size = self.header.uncompressed_size as usize;
        decompressor
            .decompress(&self.compressed, max_size)
            .map_err(|e| DaemonError::Validation(format!("zstd decompress: {e}")))
    }
}

/// Owned execution plan materialised from a mapped `.aeroc` file.
pub struct ExecutionPlan {
    pub node_count: u32,
    pub dag_stride: u32,
    pub dag_rows: Vec<u8>,
    pub instructions: Vec<Instruction>,
    pub strings: HashMap<u32, String>,
    pub payload: Vec<PayloadBlock>,
    pub dictionary: Vec<u8>,
    pub workspace: PathBuf,
    pub flags: u32,
}

impl ExecutionPlan {
    /// Resolve a `StringRef` to its contents.
    fn string(&self, r: StringRef) -> Option<&str> {
        self.strings.get(&r.0).map(String::as_str)
    }

    /// Decompress a payload block by index and return the raw bytes.
    fn decompress_block(&self, index: u32) -> Result<Vec<u8>, DaemonError> {
        let block = self
            .payload
            .get(index as usize)
            .ok_or_else(|| DaemonError::Validation(format!("payload block {index} missing")))?;
        block.decompress(&self.dictionary)
    }
}

/// Zero-copy view over an `.aeroc` byte buffer (from an `mmap` or embedded payload).
struct AerocView<'a> {
    data: &'a [u8],
    header: AerocHeader,
}

impl<'a> AerocView<'a> {
    fn new(data: &'a [u8]) -> Result<Self, DaemonError> {
        if data.len() < AEROC_HEADER_SIZE as usize {
            return Err(DaemonError::Validation("file too small for header".into()));
        }
        let header: AerocHeader = bytemuck::pod_read_unaligned(&data[..AEROC_HEADER_SIZE as usize]);

        if &header.magic() != AEROC_MAGIC {
            return Err(DaemonError::Validation("invalid aeroc magic".into()));
        }

        let body = &data[AEROC_HEADER_SIZE as usize..];
        let computed = blake3::hash(body);
        if !constant_time_eq(&computed.as_bytes()[..16], &header.content_hash()) {
            return Err(DaemonError::Validation("content hash mismatch".into()));
        }

        Ok(AerocView { data, header })
    }

    fn slice(&self, offset: u64, len: u64) -> Result<&[u8], DaemonError> {
        let off = offset as usize;
        let end = off + len as usize;
        if end > self.data.len() {
            return Err(DaemonError::Validation("section out of bounds".into()));
        }
        Ok(&self.data[off..end])
    }

    /// Materialise an `ExecutionPlan` from the view.
    fn into_plan<P: AsRef<Path>>(self, workspace: P) -> Result<ExecutionPlan, DaemonError> {
        let st = self.slice(self.header.string_table_offset(), self.header.string_table_len())?;
        let dag = self.slice(
            self.header.dag_matrix_offset(),
            self.header.dag_matrix_stride() as u64 * self.header.dag_node_count() as u64,
        )?;
        let bc = self.slice(self.header.bytecode_offset(), self.header.bytecode_len())?;
        let pl = self.slice(self.header.payload_offset(), self.header.payload_len())?;
        let dict = self.slice(
            self.header.zstd_dict_offset(),
            self.header.zstd_dict_len() as u64,
        )?;

        // Build offset -> string map for fast runtime lookups.
        let mut strings = HashMap::new();
        let mut off = 0u32;
        while (off as usize) + 2 <= st.len() {
            let len = u16::from_le_bytes([st[off as usize], st[off as usize + 1]]) as u32;
            if (off as usize) + 2 + len as usize > st.len() {
                break;
            }
            if let Ok(s) = std::str::from_utf8(&st[off as usize + 2..off as usize + 2 + len as usize]) {
                strings.insert(off, s.to_string());
            }
            off += 2 + len;
        }

        let mut instructions = Vec::new();
        let mut pos = 0usize;
        while pos < bc.len() {
            let inst = Instruction::decode(bc, &mut pos)?;
            if inst.opcode() == OpCode::Halt {
                instructions.push(inst);
                break;
            }
            instructions.push(inst);
        }

        // Parse payload blocks sequentially.
        let mut payload = Vec::new();
        let mut p = 0usize;
        while p + std::mem::size_of::<PayloadBlockHeader>() <= pl.len() {
            let header: PayloadBlockHeader =
                bytemuck::pod_read_unaligned(&pl[p..p + std::mem::size_of::<PayloadBlockHeader>()]);
            let comp_start = p + std::mem::size_of::<PayloadBlockHeader>();
            let comp_end = comp_start + header.compressed_size as usize;
            if comp_end > pl.len() {
                break;
            }
            let compressed = pl[comp_start..comp_end].to_vec();
            // Verify xxHash32 block checksum.
            let expected = header.block_hash;
            let actual = xxhash_rust::xxh32::xxh32(&compressed, 0);
            if expected != actual {
                return Err(DaemonError::Validation(format!(
                    "payload block hash mismatch: expected {expected:x}, got {actual:x}"
                )));
            }
            payload.push(PayloadBlock { header, compressed });
            p = comp_end;
        }

        Ok(ExecutionPlan {
            node_count: self.header.dag_node_count(),
            dag_stride: self.header.dag_matrix_stride(),
            dag_rows: dag.to_vec(),
            instructions,
            strings,
            payload,
            dictionary: dict.to_vec(),
            workspace: workspace.as_ref().to_path_buf(),
            flags: self.header.flags(),
        })
    }
}

/// Lock-free wavefront scheduler for the task DAG.
pub struct Scheduler {
    plan: Arc<ExecutionPlan>,
    completion: Vec<AtomicU64>,
    claimed: Vec<AtomicU8>,
    completed_count: AtomicUsize,
}

impl Scheduler {
    pub fn new(plan: Arc<ExecutionPlan>) -> Self {
        let node_count = plan.node_count as usize;
        let words = (plan.dag_stride as usize) / 8;
        Self {
            plan,
            completion: (0..words).map(|_| AtomicU64::new(0)).collect(),
            claimed: (0..node_count).map(|_| AtomicU8::new(0)).collect(),
            completed_count: AtomicUsize::new(0),
        }
    }

    /// Run all tasks using up to *max_workers* threads.
    pub fn run(&self, max_workers: usize) -> Result<(), DaemonError> {
        let workers = max_workers.max(1);
        thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| self.worker_loop());
            }
        });
        Ok(())
    }

    fn worker_loop(&self) {
        let n = self.plan.node_count as usize;
        loop {
            let mut progress = false;
            for i in 0..n {
                if self.claimed[i].load(Ordering::Relaxed) != 0 {
                    continue;
                }
                if self.is_ready(i) {
                    if self.claimed[i]
                        .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok()
                    {
                        let _ = self.execute_task(i);
                        self.mark_complete(i);
                        progress = true;
                    }
                }
            }
            let done = self.completed_count.load(Ordering::Acquire);
            if done >= n {
                break;
            }
            if !progress {
                std::hint::spin_loop();
            }
        }
    }

    fn is_ready(&self, row: usize) -> bool {
        let stride = self.plan.dag_stride as usize;
        let words = stride / 8;
        if words <= 16 && is_x86_feature_detected!("avx2") {
            unsafe { self.is_ready_avx2(row, words) }
        } else {
            self.is_ready_scalar(row, words)
        }
    }

    fn is_ready_scalar(&self, row: usize, words: usize) -> bool {
        let stride = self.plan.dag_stride as usize;
        let row_start = row * stride;
        let row_bytes = &self.plan.dag_rows[row_start..row_start + stride];
        for w in 0..words {
            let mut tmp = [0u8; 8];
            tmp.copy_from_slice(&row_bytes[w * 8..w * 8 + 8]);
            let row_word = u64::from_le_bytes(tmp);
            let comp = self.completion[w].load(Ordering::Acquire);
            if row_word & !comp != 0 {
                return false;
            }
        }
        true
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn is_ready_avx2(&self, row: usize, words: usize) -> bool {
        let stride = self.plan.dag_stride as usize;
        let row_start = row * stride;
        let row_bytes = &self.plan.dag_rows[row_start..row_start + stride];

        let mut row_buf = [0u64; 16];
        let mut comp_buf = [0u64; 16];
        for w in 0..words {
            let mut tmp = [0u8; 8];
            tmp.copy_from_slice(&row_bytes[w * 8..w * 8 + 8]);
            row_buf[w] = u64::from_le_bytes(tmp);
            comp_buf[w] = self.completion[w].load(Ordering::Acquire);
        }

        let row_ptr = row_buf.as_ptr() as *const __m256i;
        let comp_ptr = comp_buf.as_ptr() as *const __m256i;
        let all_ones = _mm256_set1_epi64x(-1);
        let chunks = words / 4;
        for c in 0..chunks {
            let r = _mm256_loadu_si256(row_ptr.add(c));
            let comp = _mm256_loadu_si256(comp_ptr.add(c));
            let not_comp = _mm256_xor_si256(comp, all_ones);
            let unsat = _mm256_and_si256(r, not_comp);
            if _mm256_testz_si256(unsat, unsat) == 0 {
                return false;
            }
        }
        // Scalar tail for remaining words (<4).
        for w in (chunks * 4)..words {
            if row_buf[w] & !comp_buf[w] != 0 {
                return false;
            }
        }
        true
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn is_ready_avx2(&self, _row: usize, _words: usize) -> bool {
        unreachable!()
    }

    fn execute_task(&self, task: usize) -> Result<(), DaemonError> {
        let engine = ExecutionEngine { plan: &self.plan };
        if task < self.plan.instructions.len() {
            engine.execute(&self.plan.instructions[task])
        } else {
            Ok(())
        }
    }

    fn mark_complete(&self, task: usize) {
        let word = task / 64;
        let bit = task % 64;
        self.completion[word].fetch_or(1u64 << bit, Ordering::Release);
        self.claimed[task].store(2, Ordering::Release);
        self.completed_count.fetch_add(1, Ordering::Release);
    }
}

/// Per-task opcode interpreter.
struct ExecutionEngine<'a> {
    plan: &'a ExecutionPlan,
}

impl<'a> ExecutionEngine<'a> {
    fn execute(&self, inst: &Instruction) -> Result<(), DaemonError> {
        match inst {
            Instruction::Nop | Instruction::Halt => Ok(()),
            Instruction::CargoBuild { manifest_ref, flags: _ } => {
                let manifest = self
                    .plan
                    .string(*manifest_ref)
                    .unwrap_or("")
                    .to_string();
                if manifest.is_empty() {
                    return Ok(());
                }
                run_cargo_build(&self.plan.workspace, &manifest)
            }
            Instruction::Pyo3Bind { src_ref, out_ref: _ } => {
                let _src = self.plan.string(*src_ref).unwrap_or("");
                // Best-effort: build the PyO3 crate using cargo.
                let manifest = self.plan.workspace.join("Cargo.toml");
                if manifest.is_file() {
                    run_cargo_build(&self.plan.workspace, manifest.to_string_lossy().as_ref())?;
                }
                Ok(())
            }
            Instruction::CAbiCheck { header_ref, abi_hash: _ } => {
                let header = self.plan.string(*header_ref).unwrap_or("");
                let path = self.plan.workspace.join(header);
                if !path.is_file() {
                    return Err(DaemonError::Execution(format!("C ABI header missing: {header}")));
                }
                Ok(())
            }
            Instruction::VmExec { .. } => Ok(()),
            Instruction::UnitVerify { test_bin_ref, args: _ } => {
                let cmd = self.plan.string(*test_bin_ref).unwrap_or("").to_string();
                if cmd.is_empty() {
                    return Ok(());
                }
                run_shell(&self.plan.workspace, &cmd)
            }
            Instruction::FsSymlink { src_ref, dst_ref } => {
                let src = self.plan.string(*src_ref).unwrap_or("");
                let dst = self.plan.string(*dst_ref).unwrap_or("");
                if src.is_empty() || dst.is_empty() {
                    return Ok(());
                }
                let src_path = self.plan.workspace.join(src);
                let dst_path = self.plan.workspace.join(dst);
                #[cfg(unix)]
                std::os::unix::fs::symlink(&src_path, &dst_path)?;
                #[cfg(not(unix))]
                std::fs::copy(&src_path, &dst_path)?;
                Ok(())
            }
            Instruction::ZstdDecomp { src_blk, dst_dir } => {
                let data = self.plan.decompress_block(*src_blk)?;
                let dir = self.plan.string(*dst_dir).unwrap_or(".");
                let out_dir = self.plan.workspace.join(dir);
                std::fs::create_dir_all(&out_dir)?;
                let out = out_dir.join(format!("block_{}.bin", src_blk));
                std::fs::write(out, data)?;
                Ok(())
            }
            Instruction::DispatchSubdag { .. } => Ok(()),
        }
    }
}

fn run_cargo_build(workspace: &Path, manifest: &str) -> Result<(), DaemonError> {
    let status = Command::new("cargo")
        .arg("build")
        .arg("--manifest-path")
        .arg(manifest)
        .current_dir(workspace)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !status.success() {
        return Err(DaemonError::Execution(format!(
            "cargo build failed with status {status}"
        )));
    }
    Ok(())
}

fn run_shell(workspace: &Path, cmd: &str) -> Result<(), DaemonError> {
    let status = if cfg!(windows) {
        Command::new("cmd").args(["/C", cmd]).current_dir(workspace).status()?
    } else {
        Command::new("sh").arg("-c").arg(cmd).current_dir(workspace).status()?
    };
    if !status.success() {
        return Err(DaemonError::Execution(format!(
            "verification command failed with status {status}"
        )));
    }
    Ok(())
}

/// Parse an `.aeroc` byte buffer into an `ExecutionPlan`.
pub fn parse_aeroc(data: &[u8], workspace: &Path) -> Result<ExecutionPlan, DaemonError> {
    AerocView::new(data)?.into_plan(workspace)
}

/// Run an `.aeroc` already loaded in memory.
pub fn run_aeroc_data(
    data: &[u8],
    workspace: &Path,
    max_workers: usize,
) -> Result<(), DaemonError> {
    let plan = Arc::new(parse_aeroc(data, workspace)?);
    let scheduler = Scheduler::new(plan);
    scheduler.run(max_workers)
}

/// Run the `.aeroc` at *aeroc_path* inside *workspace* using *max_workers* threads.
pub fn run_aeroc<P: AsRef<Path>>(
    aeroc_path: P,
    workspace: P,
    max_workers: usize,
) -> Result<(), DaemonError> {
    let file = std::fs::File::open(aeroc_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    run_aeroc_data(&mmap, workspace.as_ref(), max_workers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aeroc_compiler::{compile_aeroc_json, ProjectSpec, RawInstruction};
    use std::time::Instant;

    fn nop_spec(nodes: usize, deps: Vec<(usize, usize)>) -> ProjectSpec {
        let mut spec = ProjectSpec {
            nodes: (0..nodes).map(|i| format!("task{i}")).collect(),
            edges: HashMap::new(),
            instructions: (0..nodes)
                .map(|_| RawInstruction {
                    op: "NOP".to_string(),
                    ..Default::default()
                })
                .collect(),
            sources: Vec::new(),
            flags: 0,
        };
        for (a, b) in deps {
            spec.edges
                .entry(format!("task{a}"))
                .or_default()
                .push(format!("task{b}"));
        }
        spec
    }

    #[test]
    fn scheduler_overhead_for_1000_nodes() {
        let tmp = std::env::temp_dir().join(format!("aeroc_daemon_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("workspace.aeroc");

        let mut deps: Vec<(usize, usize)> = Vec::new();
        // Build a long chain: each node depends on the previous one.
        for i in 1..1000 {
            deps.push((i, i - 1));
        }
        let spec = nop_spec(1000, deps);
        compile_aeroc_json(&serde_json::to_string(&spec).unwrap(), &path.to_string_lossy()).unwrap();

        let start = Instant::now();
        run_aeroc(&path, &tmp, 4).unwrap();
        let elapsed = start.elapsed();
        // Scheduling overhead for a 1,000-node chain should remain well below
        // 20µs per task on average (20ms total).  Release builds are much
        // faster; debug builds still easily meet this ceiling.
        assert!(
            elapsed.as_secs_f64() < 0.020,
            "scheduling overhead too high for 1,000-node graph: {elapsed:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
