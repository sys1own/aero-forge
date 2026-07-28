//! Minimal self-extracting runner for `workspace.aeroc.bin` executables.
//!
//! The runner reads its own executable, finds the trailing `AEROCBIN` footer,
//! maps the embedded `workspace.aeroc` payload, and executes it with the native
//! wavefront scheduler.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

#[path = "../../src/aeroc_compiler.rs"]
mod aeroc_compiler;
#[path = "../../src/aeroc_daemon.rs"]
mod aeroc_daemon;

use aeroc_compiler::{AEROC_BIN_MAGIC, AEROC_TRAILER_SIZE};
use aeroc_daemon::run_aeroc_data;
use bytemuck::pod_read_unaligned;
use memmap2::MmapOptions;

fn executable_path() -> PathBuf {
    // Fall back to /proc/self/exe on Linux for a reliable, symlink-free path.
    if cfg!(target_os = "linux") {
        fs::read_link("/proc/self/exe").unwrap_or_else(|_| env::current_exe().expect("current exe"))
    } else {
        env::current_exe().expect("current exe")
    }
}

fn main() {
    let exe = executable_path();
    let workspace: PathBuf = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().expect("current dir"));
    let max_workers = env::var("AEROC_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let file = fs::File::open(&exe).unwrap_or_else(|e| {
        eprintln!("aeroc-runner: cannot open {}: {e}", exe.display());
        process::exit(1);
    });

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .unwrap_or_else(|e| {
                eprintln!("aeroc-runner: cannot mmap {}: {e}", exe.display());
                process::exit(1);
            })
    };

    if mmap.len() < AEROC_TRAILER_SIZE {
        eprintln!("aeroc-runner: executable too small to contain trailer");
        process::exit(1);
    }

    let trailer_offset = mmap.len() - AEROC_TRAILER_SIZE;
    let trailer = pod_read_unaligned::<aeroc_compiler::AerocTrailerFooter>(&mmap[trailer_offset..]);
    if &trailer.magic_trailer != AEROC_BIN_MAGIC {
        eprintln!("aeroc-runner: missing AEROCBIN trailer");
        process::exit(1);
    }

    let offset = trailer.aeroc_offset as usize;
    let size = trailer.payload_size as usize;
    if offset + size > mmap.len() {
        eprintln!("aeroc-runner: embedded payload out of bounds");
        process::exit(1);
    }

    if let Err(e) = run_aeroc_data(&mmap[offset..offset + size], &workspace, max_workers) {
        eprintln!("aeroc-runner: execution failed: {e}");
        process::exit(1);
    }
}
