//! AOT binary IR compiler for the `workspace.aeroc` container format.
//!
//! `aeroc` replaces runtime YAML/JSON parsing with a zero-copy binary
//! container: a 128-byte 64-byte-aligned header, an interned string table,
//! an NxN DAG adjacency matrix, an opcode instruction stream, and a chunked,
//! Zstd-dictionary-compressed source payload.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const AEROC_MAGIC: &[u8; 8] = b"AEROFOG\0";
pub const AEROC_VERSION_MAJOR: u16 = 1;
pub const AEROC_VERSION_MINOR: u16 = 0;
pub const AEROC_HEADER_SIZE: u32 = 128;
pub const AEROC_ALIGNMENT: usize = 64;
pub const CHUNK_SIZE: usize = 128 * 1024;
pub const DICT_SIZE: usize = 64 * 1024;

/// Errors returned by the aeroc compiler.
#[derive(Debug)]
pub enum AerocError {
    Io(io::Error),
    Json(serde_json::Error),
    InvalidArgument(String),
}

impl fmt::Display for AerocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AerocError::Io(e) => write!(f, "io error: {e}"),
            AerocError::Json(e) => write!(f, "json error: {e}"),
            AerocError::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
        }
    }
}

impl std::error::Error for AerocError {}

impl From<io::Error> for AerocError {
    fn from(e: io::Error) -> Self {
        AerocError::Io(e)
    }
}

impl From<serde_json::Error> for AerocError {
    fn from(e: serde_json::Error) -> Self {
        AerocError::Json(e)
    }
}

/// 128-byte, 64-byte-aligned container header.
///
/// Field access is performed through explicit offsets so the on-disk layout
/// matches the specification regardless of how Rust aligns the backing store.
#[repr(C, align(64))]
pub struct AerocHeader {
    pub data: [u8; AEROC_HEADER_SIZE as usize],
}

impl Default for AerocHeader {
    fn default() -> Self {
        let mut h = Self {
            data: [0; AEROC_HEADER_SIZE as usize],
        };
        h.set_magic();
        h.set_version(AEROC_VERSION_MAJOR, AEROC_VERSION_MINOR);
        h.set_header_size(AEROC_HEADER_SIZE);
        h
    }
}

macro_rules! read_le {
    ($buf:expr, $ty:ty, $off:expr) => {{
        let sz = std::mem::size_of::<$ty>();
        let mut tmp = [0u8; std::mem::size_of::<$ty>()];
        tmp.copy_from_slice(&$buf[$off..$off + sz]);
        <$ty>::from_le_bytes(tmp)
    }};
}

macro_rules! write_le {
    ($buf:expr, $ty:ty, $off:expr, $val:expr) => {{
        let bytes = <$ty>::to_le_bytes($val);
        let sz = bytes.len();
        $buf[$off..$off + sz].copy_from_slice(&bytes);
    }};
}

impl AerocHeader {
    pub fn magic(&self) -> [u8; 8] {
        let mut m = [0u8; 8];
        m.copy_from_slice(&self.data[0..8]);
        m
    }

    pub fn set_magic(&mut self) {
        self.data[0..8].copy_from_slice(AEROC_MAGIC);
    }

    pub fn version_major(&self) -> u16 {
        read_le!(self.data, u16, 8)
    }
    pub fn version_minor(&self) -> u16 {
        read_le!(self.data, u16, 10)
    }
    pub fn set_version(&mut self, major: u16, minor: u16) {
        write_le!(self.data, u16, 8, major);
        write_le!(self.data, u16, 10, minor);
    }

    pub fn flags(&self) -> u32 {
        read_le!(self.data, u32, 12)
    }
    pub fn set_flags(&mut self, flags: u32) {
        write_le!(self.data, u32, 12, flags);
    }

    pub fn header_size(&self) -> u32 {
        read_le!(self.data, u32, 16)
    }
    pub fn set_header_size(&mut self, size: u32) {
        write_le!(self.data, u32, 16, size);
    }

    pub fn string_table_offset(&self) -> u64 {
        read_le!(self.data, u64, 20)
    }
    pub fn set_string_table_offset(&mut self, v: u64) {
        write_le!(self.data, u64, 20, v);
    }

    pub fn string_table_len(&self) -> u64 {
        read_le!(self.data, u64, 28)
    }
    pub fn set_string_table_len(&mut self, v: u64) {
        write_le!(self.data, u64, 28, v);
    }

    pub fn dag_matrix_offset(&self) -> u64 {
        read_le!(self.data, u64, 36)
    }
    pub fn set_dag_matrix_offset(&mut self, v: u64) {
        write_le!(self.data, u64, 36, v);
    }

    pub fn dag_node_count(&self) -> u32 {
        read_le!(self.data, u32, 44)
    }
    pub fn set_dag_node_count(&mut self, v: u32) {
        write_le!(self.data, u32, 44, v);
    }

    pub fn dag_matrix_stride(&self) -> u32 {
        read_le!(self.data, u32, 48)
    }
    pub fn set_dag_matrix_stride(&mut self, v: u32) {
        write_le!(self.data, u32, 48, v);
    }

    pub fn bytecode_offset(&self) -> u64 {
        read_le!(self.data, u64, 52)
    }
    pub fn set_bytecode_offset(&mut self, v: u64) {
        write_le!(self.data, u64, 52, v);
    }

    pub fn bytecode_len(&self) -> u64 {
        read_le!(self.data, u64, 60)
    }
    pub fn set_bytecode_len(&mut self, v: u64) {
        write_le!(self.data, u64, 60, v);
    }

    pub fn payload_offset(&self) -> u64 {
        read_le!(self.data, u64, 68)
    }
    pub fn set_payload_offset(&mut self, v: u64) {
        write_le!(self.data, u64, 68, v);
    }

    pub fn payload_len(&self) -> u64 {
        read_le!(self.data, u64, 76)
    }
    pub fn set_payload_len(&mut self, v: u64) {
        write_le!(self.data, u64, 76, v);
    }

    pub fn zstd_dict_offset(&self) -> u64 {
        read_le!(self.data, u64, 84)
    }
    pub fn set_zstd_dict_offset(&mut self, v: u64) {
        write_le!(self.data, u64, 84, v);
    }

    pub fn zstd_dict_len(&self) -> u32 {
        read_le!(self.data, u32, 92)
    }
    pub fn set_zstd_dict_len(&mut self, v: u32) {
        write_le!(self.data, u32, 92, v);
    }

    pub fn reserved(&self) -> [u8; 16] {
        let mut r = [0u8; 16];
        r.copy_from_slice(&self.data[96..112]);
        r
    }
    pub fn set_reserved(&mut self, v: [u8; 16]) {
        self.data[96..112].copy_from_slice(&v);
    }

    pub fn content_hash(&self) -> [u8; 16] {
        let mut h = [0u8; 16];
        h.copy_from_slice(&self.data[112..128]);
        h
    }
    pub fn set_content_hash(&mut self, v: [u8; 16]) {
        self.data[112..128].copy_from_slice(&v);
    }
}

/// 16-byte header for each compressed source payload chunk.
#[repr(C, packed)]
pub struct PayloadBlockHeader {
    pub uncompressed_size: u32,
    pub compressed_size: u32,
    pub payload_type: u8,
    pub reserved: [u8; 3],
    pub block_hash: u32,
}

impl PayloadBlockHeader {
    pub fn as_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        write_le!(buf, u32, 0, self.uncompressed_size);
        write_le!(buf, u32, 4, self.compressed_size);
        buf[8] = self.payload_type;
        buf[9..12].copy_from_slice(&self.reserved);
        write_le!(buf, u32, 12, self.block_hash);
        buf
    }
}

/// 32-bit byte offset into the interned string table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StringRef(pub u32);

/// Interned string table.  Each entry is prefixed by a 16-bit little-endian
/// length and the string bytes are stored contiguously.
pub struct StringTable {
    bytes: Vec<u8>,
    index: std::collections::HashMap<String, u32>,
}

impl StringTable {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            index: std::collections::HashMap::new(),
        }
    }

    /// Insert a string and return a stable 32-bit byte offset.
    ///
    /// Duplicate strings reuse the existing offset.
    pub fn insert(&mut self, s: &str) -> StringRef {
        if let Some(&offset) = self.index.get(s) {
            return StringRef(offset);
        }
        let offset = self.bytes.len() as u32;
        let bytes = s.as_bytes();
        let len = bytes.len().min(u16::MAX as usize) as u16;
        self.bytes.extend_from_slice(&len.to_le_bytes());
        self.bytes.extend_from_slice(&bytes[..len as usize]);
        self.index.insert(s.to_string(), offset);
        StringRef(offset)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Bytecode opcodes for the `aeroc` instruction stream.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpCode {
    Nop = 0x00,
    CargoBuild = 0x01,
    Pyo3Bind = 0x02,
    CAbiCheck = 0x03,
    VmExec = 0x04,
    UnitVerify = 0x05,
    FsSymlink = 0x06,
    ZstdDecomp = 0x07,
    DispatchSubdag = 0x08,
    Halt = 0xFF,
}

impl TryFrom<u8> for OpCode {
    type Error = io::Error;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0x00 => Ok(OpCode::Nop),
            0x01 => Ok(OpCode::CargoBuild),
            0x02 => Ok(OpCode::Pyo3Bind),
            0x03 => Ok(OpCode::CAbiCheck),
            0x04 => Ok(OpCode::VmExec),
            0x05 => Ok(OpCode::UnitVerify),
            0x06 => Ok(OpCode::FsSymlink),
            0x07 => Ok(OpCode::ZstdDecomp),
            0x08 => Ok(OpCode::DispatchSubdag),
            0xFF => Ok(OpCode::Halt),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "unknown opcode")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instruction {
    Nop,
    CargoBuild { manifest_ref: StringRef, flags: u32 },
    Pyo3Bind { src_ref: StringRef, out_ref: StringRef },
    CAbiCheck { header_ref: StringRef, abi_hash: u64 },
    VmExec { bytecode_ref: u32, mem_limit: u64 },
    UnitVerify { test_bin_ref: StringRef, args: u32 },
    FsSymlink { src_ref: StringRef, dst_ref: StringRef },
    ZstdDecomp { src_blk: u32, dst_dir: StringRef },
    DispatchSubdag { subdag_offset: u64, node_count: u32 },
    Halt,
}

impl Instruction {
    pub fn opcode(&self) -> OpCode {
        match self {
            Instruction::Nop => OpCode::Nop,
            Instruction::CargoBuild { .. } => OpCode::CargoBuild,
            Instruction::Pyo3Bind { .. } => OpCode::Pyo3Bind,
            Instruction::CAbiCheck { .. } => OpCode::CAbiCheck,
            Instruction::VmExec { .. } => OpCode::VmExec,
            Instruction::UnitVerify { .. } => OpCode::UnitVerify,
            Instruction::FsSymlink { .. } => OpCode::FsSymlink,
            Instruction::ZstdDecomp { .. } => OpCode::ZstdDecomp,
            Instruction::DispatchSubdag { .. } => OpCode::DispatchSubdag,
            Instruction::Halt => OpCode::Halt,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = vec![self.opcode() as u8];
        match self {
            Instruction::Nop | Instruction::Halt => {}
            Instruction::CargoBuild { manifest_ref, flags } => {
                buf.extend_from_slice(&manifest_ref.0.to_le_bytes());
                buf.extend_from_slice(&flags.to_le_bytes());
            }
            Instruction::Pyo3Bind { src_ref, out_ref } => {
                buf.extend_from_slice(&src_ref.0.to_le_bytes());
                buf.extend_from_slice(&out_ref.0.to_le_bytes());
            }
            Instruction::CAbiCheck { header_ref, abi_hash } => {
                buf.extend_from_slice(&header_ref.0.to_le_bytes());
                buf.extend_from_slice(&abi_hash.to_le_bytes());
            }
            Instruction::VmExec { bytecode_ref, mem_limit } => {
                buf.extend_from_slice(&bytecode_ref.to_le_bytes());
                buf.extend_from_slice(&mem_limit.to_le_bytes());
            }
            Instruction::UnitVerify { test_bin_ref, args } => {
                buf.extend_from_slice(&test_bin_ref.0.to_le_bytes());
                buf.extend_from_slice(&args.to_le_bytes());
            }
            Instruction::FsSymlink { src_ref, dst_ref } => {
                buf.extend_from_slice(&src_ref.0.to_le_bytes());
                buf.extend_from_slice(&dst_ref.0.to_le_bytes());
            }
            Instruction::ZstdDecomp { src_blk, dst_dir } => {
                buf.extend_from_slice(&src_blk.to_le_bytes());
                buf.extend_from_slice(&dst_dir.0.to_le_bytes());
            }
            Instruction::DispatchSubdag { subdag_offset, node_count } => {
                buf.extend_from_slice(&subdag_offset.to_le_bytes());
                buf.extend_from_slice(&node_count.to_le_bytes());
            }
        }
        buf
    }
}

/// Row-major NxN DAG adjacency bit matrix with 64-byte aligned rows.
pub struct DagMatrix {
    pub node_count: u32,
    pub stride: u32,
    pub rows: Vec<u8>,
}

impl DagMatrix {
    pub fn new(node_count: usize) -> Self {
        let bits_per_row = (node_count + 7) / 8;
        let stride = ((bits_per_row + 63) / 64) * 64;
        Self {
            node_count: node_count as u32,
            stride: stride as u32,
            rows: vec![0; stride * node_count],
        }
    }

    pub fn set_dep(&mut self, row: usize, col: usize) {
        if row >= self.node_count as usize || col >= self.node_count as usize {
            return;
        }
        let byte = row * self.stride as usize + col / 8;
        self.rows[byte] |= 1 << (col % 8);
    }
}

/// A source payload entry supplied to the compiler.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct SourceFile {
    pub path: String,
    #[serde(rename = "content_base64", with = "base64_bytes")]
    pub content: Vec<u8>,
}

mod base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(serde::de::Error::custom)
    }
}

/// High-level build definition consumed by the compiler.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ProjectSpec {
    pub nodes: Vec<String>,
    #[serde(default)]
    pub edges: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub instructions: Vec<RawInstruction>,
    #[serde(default)]
    pub sources: Vec<SourceFile>,
    pub flags: u32,
}

/// Intermediate JSON representation of an instruction before string interning.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RawInstruction {
    pub op: String,
    #[serde(default)]
    pub manifest_ref: Option<String>,
    #[serde(default)]
    pub src_ref: Option<String>,
    #[serde(default)]
    pub out_ref: Option<String>,
    #[serde(default)]
    pub header_ref: Option<String>,
    #[serde(default)]
    pub dst_ref: Option<String>,
    #[serde(default)]
    pub test_bin_ref: Option<String>,
    #[serde(default)]
    pub dst_dir: Option<String>,
    #[serde(default)]
    pub flags: u32,
    #[serde(default)]
    pub abi_hash: u64,
    #[serde(default)]
    pub bytecode_ref: u32,
    #[serde(default)]
    pub mem_limit: u64,
    #[serde(default)]
    pub args: u32,
    #[serde(default)]
    pub src_blk: u32,
    #[serde(default)]
    pub subdag_offset: u64,
    #[serde(default)]
    pub node_count: u32,
}

fn canonicalize_path(path: &str) -> String {
    Path::new(path)
        .to_slash_lossy()
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_string()
}

pub trait ToSlashLossy {
    fn to_slash_lossy(&self) -> String;
}

impl ToSlashLossy for Path {
    fn to_slash_lossy(&self) -> String {
        self.to_string_lossy().replace('\\', "/")
    }
}

fn intern_instruction(table: &mut StringTable, raw: &RawInstruction) -> Result<Instruction, AerocError> {
    match raw.op.as_str() {
        "NOP" => Ok(Instruction::Nop),
        "HALT" => Ok(Instruction::Halt),
        "CARGO_BUILD" => {
            let s = raw.manifest_ref.as_deref().unwrap_or("");
            Ok(Instruction::CargoBuild {
                manifest_ref: table.insert(&canonicalize_path(s)),
                flags: raw.flags,
            })
        }
        "PYO3_BIND" => {
            let src = raw.src_ref.as_deref().unwrap_or("");
            let out = raw.out_ref.as_deref().unwrap_or("");
            Ok(Instruction::Pyo3Bind {
                src_ref: table.insert(&canonicalize_path(src)),
                out_ref: table.insert(&canonicalize_path(out)),
            })
        }
        "CABI_CHECK" => {
            let s = raw.header_ref.as_deref().unwrap_or("");
            Ok(Instruction::CAbiCheck {
                header_ref: table.insert(&canonicalize_path(s)),
                abi_hash: raw.abi_hash,
            })
        }
        "VM_EXEC" => Ok(Instruction::VmExec {
            bytecode_ref: raw.bytecode_ref,
            mem_limit: raw.mem_limit,
        }),
        "UNIT_VERIFY" => {
            let s = raw.test_bin_ref.as_deref().unwrap_or("");
            Ok(Instruction::UnitVerify {
                test_bin_ref: table.insert(&canonicalize_path(s)),
                args: raw.args,
            })
        }
        "FS_SYMLINK" => {
            let src = raw.src_ref.as_deref().unwrap_or("");
            let dst = raw.dst_ref.as_deref().unwrap_or("");
            Ok(Instruction::FsSymlink {
                src_ref: table.insert(&canonicalize_path(src)),
                dst_ref: table.insert(&canonicalize_path(dst)),
            })
        }
        "ZSTD_DECOMP" => {
            let s = raw.dst_dir.as_deref().unwrap_or("");
            Ok(Instruction::ZstdDecomp {
                src_blk: raw.src_blk,
                dst_dir: table.insert(&canonicalize_path(s)),
            })
        }
        "DISPATCH_SUBDAG" => Ok(Instruction::DispatchSubdag {
            subdag_offset: raw.subdag_offset,
            node_count: raw.node_count,
        }),
        other => Err(AerocError::InvalidArgument(format!(
            "Unknown aeroc opcode: {other}"
        ))),
    }
}

/// Compile a `ProjectSpec` into a `workspace.aeroc` file at `output_path`.
///
/// Returns the truncated 128-bit BLAKE3 content hash (hex) of the file body.
pub fn compile_project(spec: &ProjectSpec, output_path: &Path) -> Result<String, AerocError> {
    let mut table = StringTable::new();

    // Intern node names first so their indices are stable.
    for node in &spec.nodes {
        table.insert(node);
    }

    let node_to_index: HashMap<String, usize> = spec
        .nodes
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

    // Intern source paths and instruction strings.
    for src in &spec.sources {
        table.insert(&canonicalize_path(&src.path));
    }

    let mut instructions: Vec<Instruction> = Vec::with_capacity(spec.instructions.len());
    for raw in &spec.instructions {
        instructions.push(intern_instruction(&mut table, raw)?);
    }

    // Build bytecode.
    let mut bytecode = Vec::new();
    for inst in &instructions {
        bytecode.extend_from_slice(&inst.encode());
    }

    // Build DAG matrix.
    let mut dag = DagMatrix::new(spec.nodes.len());
    for (node, deps) in &spec.edges {
        if let Some(&row) = node_to_index.get(node) {
            for dep in deps {
                if let Some(&col) = node_to_index.get(dep) {
                    dag.set_dep(row, col);
                }
            }
        }
    }

    // Train a 64 KiB Zstd dictionary from source contents and compress chunks.
    let samples: Vec<&[u8]> = spec.sources.iter().map(|s| s.content.as_slice()).collect();
    let dictionary = build_dictionary(&samples);

    let mut payload = Vec::new();
    for src in &spec.sources {
        let chunks = src.content.chunks(CHUNK_SIZE);
        for chunk in chunks {
            let compressed = compress_chunk(chunk, &dictionary);
            let uncompressed_size = chunk.len() as u32;
            let compressed_size = compressed.len() as u32;
            let block_hash = xxhash32(&compressed);
            let header = PayloadBlockHeader {
                uncompressed_size,
                compressed_size,
                payload_type: 0x01,
                reserved: [0; 3],
                block_hash,
            };
            payload.extend_from_slice(&header.as_bytes());
            payload.extend_from_slice(&compressed);
        }
    }

    let string_table_bytes = table.as_bytes().to_vec();
    let dag_bytes = dag.rows;
    let zstd_dict = dictionary;

    let header_size = AEROC_HEADER_SIZE as u64;
    let string_table_offset = header_size;
    let dag_matrix_offset = string_table_offset + string_table_bytes.len() as u64;
    let bytecode_offset = dag_matrix_offset + dag_bytes.len() as u64;
    let payload_offset = bytecode_offset + bytecode.len() as u64;
    let zstd_dict_offset = payload_offset + payload.len() as u64;

    // Build body for content hashing (everything after the header).
    let mut body = Vec::new();
    body.extend_from_slice(&string_table_bytes);
    body.extend_from_slice(&dag_bytes);
    body.extend_from_slice(&bytecode);
    body.extend_from_slice(&payload);
    body.extend_from_slice(&zstd_dict);

    let content_hash = blake3::hash(&body);
    let mut header = AerocHeader::default();
    header.set_flags(spec.flags);
    header.set_string_table_offset(string_table_offset);
    header.set_string_table_len(string_table_bytes.len() as u64);
    header.set_dag_matrix_offset(dag_matrix_offset);
    header.set_dag_node_count(dag.node_count);
    header.set_dag_matrix_stride(dag.stride);
    header.set_bytecode_offset(bytecode_offset);
    header.set_bytecode_len(bytecode.len() as u64);
    header.set_payload_offset(payload_offset);
    header.set_payload_len(payload.len() as u64);
    header.set_zstd_dict_offset(zstd_dict_offset);
    header.set_zstd_dict_len(zstd_dict.len() as u32);
    header.set_content_hash(content_hash.as_bytes()[..16].try_into().unwrap());

    let mut file = header.data.to_vec();
    file.extend_from_slice(&body);

    std::fs::write(output_path, &file)?;

    Ok(hex::encode(&content_hash.as_bytes()[..16]))
}

fn build_dictionary(samples: &[&[u8]]) -> Vec<u8> {
    if samples.is_empty() || samples.iter().map(|s| s.len()).sum::<usize>() < 1024 {
        return Vec::new();
    }
    zstd::dict::from_samples(samples, DICT_SIZE).unwrap_or_else(|e| {
        eprintln!("aeroc dictionary training failed: {e}");
        Vec::new()
    })
}

fn compress_chunk(chunk: &[u8], dictionary: &[u8]) -> Vec<u8> {
    let compressor = if dictionary.is_empty() {
        zstd::bulk::Compressor::new(3)
    } else {
        zstd::bulk::Compressor::with_dictionary(3, dictionary)
    };
    match compressor.and_then(|mut c| c.compress(chunk)) {
        Ok(v) => v,
        Err(_) => chunk.to_vec(),
    }
}

fn xxhash32(data: &[u8]) -> u32 {
    xxhash_rust::xxh32::xxh32(data, 0)
}

/// Parse a JSON project specification and compile it to `output_path`.
pub fn compile_aeroc_json(spec_json: &str, output_path: &str) -> Result<String, AerocError> {
    let spec: ProjectSpec = serde_json::from_str(spec_json)?;
    compile_project(&spec, Path::new(output_path))
}

/// Hex encoding helper used for the content hash returned to Python.
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn aeroc_header_size_and_alignment() {
        assert_eq!(std::mem::size_of::<AerocHeader>(), 128);
        assert_eq!(std::mem::align_of::<AerocHeader>(), 64);
    }

    #[test]
    fn payload_block_header_size() {
        assert_eq!(std::mem::size_of::<PayloadBlockHeader>(), 16);
    }

    #[test]
    fn string_table_roundtrip() {
        let mut t = StringTable::new();
        let r1 = t.insert("hello");
        let r2 = t.insert("world");
        let r3 = t.insert("hello");
        assert_eq!(r1, r3);
        let bytes = t.as_bytes();
        // First entry: 5 "hello"
        assert_eq!(&bytes[0..2], &(5u16).to_le_bytes());
        assert_eq!(&bytes[2..7], b"hello");
        // Second entry: 5 "world"
        assert_eq!(&bytes[7..9], &(5u16).to_le_bytes());
        assert_eq!(&bytes[9..14], b"world");
    }

    #[test]
    fn instruction_encoding_roundtrip() {
        let mut t = StringTable::new();
        let a = t.insert("a");
        let b = t.insert("b");
        let inst = Instruction::CargoBuild {
            manifest_ref: a,
            flags: 0x12345678,
        };
        let encoded = inst.encode();
        assert_eq!(encoded[0], OpCode::CargoBuild as u8);
        assert_eq!(&encoded[1..5], &a.0.to_le_bytes());
        assert_eq!(&encoded[5..9], &0x12345678u32.to_le_bytes());

        let inst2 = Instruction::Pyo3Bind {
            src_ref: a,
            out_ref: b,
        };
        let e2 = inst2.encode();
        assert_eq!(e2.len(), 9);

        let inst3 = Instruction::DispatchSubdag {
            subdag_offset: 0x_dead_beef_cafe,
            node_count: 42,
        };
        let e3 = inst3.encode();
        assert_eq!(e3.len(), 13);
    }

    #[test]
    fn compile_is_deterministic() {
        let spec = ProjectSpec {
            nodes: vec!["a".into(), "b".into()],
            edges: [("a".into(), vec!["b".into()])].into_iter().collect(),
            instructions: vec![
                RawInstruction {
                    op: "CARGO_BUILD".into(),
                    manifest_ref: Some("Cargo.toml".into()),
                    flags: 1,
                    ..Default::default()
                },
                RawInstruction { op: "HALT".into(), ..Default::default() },
            ],
            sources: vec![SourceFile {
                path: "src/main.rs".into(),
                content: b"fn main() {}".to_vec(),
            }],
            flags: 0,
        };

        let tmp = std::env::temp_dir();
        let p1 = tmp.join("aeroc_det1.aeroc");
        let p2 = tmp.join("aeroc_det2.aeroc");
        let h1 = compile_project(&spec, &p1).unwrap();
        let h2 = compile_project(&spec, &p2).unwrap();
        assert_eq!(h1, h2);
        let b1 = fs::read(&p1).unwrap();
        let b2 = fs::read(&p2).unwrap();
        assert_eq!(b1, b2);

        let h = AerocHeader { data: b1[..128].try_into().unwrap() };
        assert_eq!(&h.magic()[..], b"AEROFOG\0");
        assert_eq!(h.version_major(), 1);
        assert_eq!(h.version_minor(), 0);
        assert_eq!(h.header_size(), 128);
        assert_eq!(h.dag_node_count(), 2);
        assert!(h.dag_matrix_stride() >= 64);

        fs::remove_file(&p1).ok();
        fs::remove_file(&p2).ok();
    }

    #[test]
    fn payload_chunk_compression_and_hash() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let compressed = compress_chunk(data, &[]);
        assert!(!compressed.is_empty());
        assert_eq!(xxhash32(&compressed), xxhash32(&compressed));
    }
}
