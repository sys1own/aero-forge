//! Parallel ``workspace.aeroc`` unpacker.
//!
//! Decompresses every source payload block, verifies its xxHash32, and writes
//! the original file tree back to disk.

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;


use constant_time_eq::constant_time_eq;
use memmap2::MmapOptions;
use rayon::prelude::*;

use crate::aeroc_compiler::{
    AerocHeader, PayloadBlockHeader, SourceMapEntry, AEROC_HEADER_SIZE, AEROC_MAGIC,
};

/// Errors raised while unpacking a ``workspace.aeroc``.
#[derive(Debug)]
pub enum UnpackError {
    Io(io::Error),
    Validation(String),
    Decompress(String),
}

impl fmt::Display for UnpackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnpackError::Io(e) => write!(f, "io error: {e}"),
            UnpackError::Validation(s) => write!(f, "validation error: {s}"),
            UnpackError::Decompress(s) => write!(f, "decompress error: {s}"),
        }
    }
}

impl std::error::Error for UnpackError {}

impl From<io::Error> for UnpackError {
    fn from(e: io::Error) -> Self {
        UnpackError::Io(e)
    }
}

/// Lightweight view of one compressed payload block inside the mmap.
struct PayloadBlockInfo {
    header: PayloadBlockHeader,
    compressed: std::ops::Range<usize>,
}

/// A mapped ``workspace.aeroc`` with section accessors.
pub struct AerocFile {
    mmap: memmap2::Mmap,
    header: AerocHeader,
}

impl AerocFile {
    /// Map *path* into memory and validate magic/hash.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, UnpackError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() < AEROC_HEADER_SIZE as usize {
            return Err(UnpackError::Validation("file too small for header".into()));
        }
        let header: AerocHeader = bytemuck::pod_read_unaligned(
            &mmap[..std::mem::size_of::<AerocHeader>()],
        );

        if &header.magic() != AEROC_MAGIC {
            return Err(UnpackError::Validation("invalid aeroc magic".into()));
        }

        let body = &mmap[AEROC_HEADER_SIZE as usize..];
        let computed = blake3::hash(body);
        if !constant_time_eq(&computed.as_bytes()[..16], &header.content_hash()) {
            return Err(UnpackError::Validation("content hash mismatch".into()));
        }

        Ok(AerocFile { mmap, header })
    }

    fn slice(&self, offset: u64, len: u64) -> Result<&[u8], UnpackError> {
        let off = offset as usize;
        let end = off + len as usize;
        if end > self.mmap.len() {
            return Err(UnpackError::Validation("section out of bounds".into()));
        }
        Ok(&self.mmap[off..end])
    }

    /// Read a UTF-8 string from the string table at the given byte offset.
    fn read_string(&self, offset: u32) -> Result<&str, UnpackError> {
        let table = self.slice(self.header.string_table_offset(), self.header.string_table_len())?;
        let off = offset as usize;
        if off + 2 > table.len() {
            return Err(UnpackError::Validation("string offset out of bounds".into()));
        }
        let mut len_bytes = [0u8; 2];
        len_bytes.copy_from_slice(&table[off..off + 2]);
        let len = u16::from_le_bytes(len_bytes) as usize;
        if off + 2 + len > table.len() {
            return Err(UnpackError::Validation("string length out of bounds".into()));
        }
        std::str::from_utf8(&table[off + 2..off + 2 + len])
            .map_err(|_| UnpackError::Validation("invalid utf-8 in string table".into()))
    }

    /// Build an index of every payload block by walking the payload section.
    fn payload_blocks(&self) -> Result<Vec<PayloadBlockInfo>, UnpackError> {
        let data = self.slice(self.header.payload_offset(), self.header.payload_len())?;
        let mut blocks = Vec::new();
        let mut cursor = 0usize;
        while cursor + 16 <= data.len() {
            let header: PayloadBlockHeader = bytemuck::pod_read_unaligned(&data[cursor..cursor + 16]);
            let end = cursor + 16 + header.compressed_size as usize;
            if end > data.len() {
                return Err(UnpackError::Validation("payload block truncated".into()));
            }
            blocks.push(PayloadBlockInfo {
                header,
                compressed: (self.header.payload_offset() as usize + cursor + 16)
                    ..(self.header.payload_offset() as usize + end),
            });
            cursor = end;
        }
        Ok(blocks)
    }

    /// Return the source map as a slice of entries.
    fn source_map(&self) -> Result<Vec<SourceMapEntry>, UnpackError> {
        let data = self.slice(self.header.source_map_offset(), self.header.source_map_len())?;
        let entry_size = std::mem::size_of::<SourceMapEntry>();
        if data.len() % entry_size != 0 {
            return Err(UnpackError::Validation("source map size misaligned".into()));
        }
        let mut entries = Vec::with_capacity(data.len() / entry_size);
        for chunk in data.chunks_exact(entry_size) {
            entries.push(bytemuck::pod_read_unaligned(chunk));
        }
        Ok(entries)
    }

    /// Zstd dictionary bytes (may be empty).
    fn dictionary(&self) -> Result<&[u8], UnpackError> {
        self.slice(
            self.header.zstd_dict_offset(),
            self.header.zstd_dict_len() as u64,
        )
    }
}

/// Unpack *aeroc_path* into *output_dir*, preserving the stored source tree.
pub fn unpack_aeroc<P: AsRef<Path>>(aeroc_path: P, output_dir: P) -> Result<usize, UnpackError> {
    let file = AerocFile::open(aeroc_path)?;
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let dictionary = file.dictionary()?.to_vec();
    let blocks = file.payload_blocks()?;
    let entries = file.source_map()?;

    // Parallel extraction: one source file per rayon task.
    entries.par_iter().try_for_each(|entry| {
        let path = file.read_string(entry.path_ref)?;
        let out_path = output_dir.join(path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let start = entry.start_block as usize;
        let end = start + entry.block_count as usize;
        if end > blocks.len() {
            return Err(UnpackError::Validation(format!(
                "source {path} block range out of bounds"
            )));
        }

        // Pre-size the output buffer to the sum of the declared uncompressed sizes.
        let mut total_uncompressed = 0usize;
        let mut out = Vec::new();
        for idx in start..end {
            let info = &blocks[idx];
            total_uncompressed += info.header.uncompressed_size as usize;
        }
        out.reserve(total_uncompressed);

        for idx in start..end {
            let info = &blocks[idx];
            let compressed = &file.mmap[info.compressed.clone()];

            // Verify xxHash32 of the compressed payload before decompression.
            let actual_hash = crate::aeroc_compiler::xxhash32(compressed);
            if actual_hash != info.header.block_hash {
                return Err(UnpackError::Validation(format!(
                    "block {idx} hash mismatch for {path}"
                )));
            }

            let mut decompressor = if dictionary.is_empty() {
                zstd::bulk::Decompressor::new()
            } else {
                zstd::bulk::Decompressor::with_dictionary(&dictionary)
            }
            .map_err(|e| UnpackError::Decompress(format!("zstd decompressor: {e}")))?;

            let chunk = decompressor
                .decompress(compressed, info.header.uncompressed_size as usize)
                .map_err(|e| UnpackError::Decompress(format!("zstd decompress: {e}")))?;
            out.extend_from_slice(&chunk);
        }

        fs::write(&out_path, &out)?;
        Ok(())
    })?;

    Ok(entries.len())
}
