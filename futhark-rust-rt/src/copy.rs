//! COPIES: Futhark's LMAD copies between device and host memory (`rts/c/gpu.h`, `rts/c/copy.h`), scalar reads and
//! writes of device memory, and the GPU pass timer.
//!
//! A device-to-device copy is classified as the C runtime classifies it: a batch of transpositions (the built-in
//! `map_transpose_*` kernels, the variant chosen by the same rules), a flat copy (`copy_buffer_to_buffer`), or a
//! general copy (the built-in `lmad_copy_*` kernel). Host copies must be flat, as in the C runtime.

use crate::program::KArg;
use crate::{launch, DevMem, Error, FutharkCtx, HostMem, Program};

/// One side of a copy: a memory variable, an offset and strides in elements.
pub struct Lmad<'a, M> {
    pub mem: &'a M,
    pub offset: i64,
    pub strides: &'a [i64],
}

impl<'a, M> Lmad<'a, M> {
    pub fn new(mem: &'a M, offset: i64, strides: &'a [i64]) -> Lmad<'a, M> {
        Lmad { mem, offset, strides }
    }
}

/// A batch of transpositions: `k` arrays of `n` by `m`.
#[derive(Debug, Clone, Copy)]
struct TrDims {
    k: i64,
    n: i64,
    m: i64,
}

const TR_BLOCK_DIM: i64 = 16;
const TR_TILE_DIM: i64 = TR_BLOCK_DIM * 2;
const TR_ELEMS_PER_THREAD: i64 = 8;
const MAX_TR_THREAD_BLOCKS: i32 = 65535;

fn round4(n: u64) -> u64 {
    n.div_ceil(4) * 4
}

/// One compute pass timed with GPU timestamps (the shim's `TimedPass`).
pub(crate) struct PassTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
}

impl PassTimer {
    pub(crate) fn new(device: &wgpu::Device) -> PassTimer {
        PassTimer {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("futhark pass timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("futhark timestamp resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("futhark timestamp read"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    pub(crate) fn writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites { query_set: &self.queries, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }
    }

    /// The pass's GPU nanoseconds. As the shim does: wait for the pass's submission, then resolve the counters in a
    /// submission of their own (resolved in the pass's own command buffer, Metal returned zero timestamps in 3 of 4
    /// runs, measured 2026-10-27).
    pub(crate) fn read(self, ctx: &FutharkCtx) -> Result<f64, Error> {
        ctx.sync()?;
        let mut enc = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("futhark timestamps") });
        enc.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, Some(16));
        ctx.queue().submit(Some(enc.finish()));
        ctx.map_read(&self.read, 16)?;
        let ticks: [u64; 2] = {
            let view = crate::compat::mapped(&self.read, 0..16)?;
            let word = |i: usize| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&view[i * 8..i * 8 + 8]);
                u64::from_le_bytes(b)
            };
            [word(0), word(1)]
        };
        self.read.unmap();
        self.read.destroy();
        self.resolve.destroy();
        if ticks[1] < ticks[0] {
            return Err(Error::new(format!("a pass ended before it began: timestamps {} then {}", ticks[0], ticks[1])));
        }
        Ok((ticks[1] - ticks[0]) as f64 * f64::from(ctx.queue().get_timestamp_period()))
    }
}

/// `gpu_scalar_from_device`: element `i` of type `T` of a device block. Read through the 4-byte aligned window
/// around it (every block's size is a multiple of 4, so the window is inside the buffer).
pub fn read_scalar<T: crate::Elem>(ctx: &FutharkCtx, mem: &DevMem, i: i64) -> Result<T, Error> {
    let block = mem.block()?;
    let off = i as u64 * T::SIZE as u64;
    let start = off & !3;
    let len = round4(off + T::SIZE as u64) - start;
    if len > ctx.readback().size() {
        return Err(Error::new(format!("our software has a bug: a scalar read of {len} bytes")));
    }
    let mut enc = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("futhark scalar read") });
    enc.copy_buffer_to_buffer(block.buffer(), start, ctx.readback(), 0, Some(len));
    ctx.queue().submit(Some(enc.finish()));
    ctx.count(|s| s.submits += 1);
    ctx.map_read(ctx.readback(), len)?;
    let v = {
        let view = crate::compat::mapped(ctx.readback(), 0..len)?;
        let at = (off - start) as usize;
        T::read_le(&view[at..at + T::SIZE])
    };
    ctx.readback().unmap();
    Ok(v)
}

/// `gpu_scalar_to_device`: element `i` of a device block set to `v`. A write that is not 4-byte aligned and sized is
/// made on the aligned window around it, read first.
pub fn write_scalar<T: crate::Elem>(ctx: &FutharkCtx, mem: &DevMem, i: i64, v: T) -> Result<(), Error> {
    let block = mem.block()?;
    let off = i as u64 * T::SIZE as u64;
    let mut bytes = vec![0u8; T::SIZE];
    v.write_le(&mut bytes);
    if off.is_multiple_of(4) && T::SIZE.is_multiple_of(4) {
        ctx.queue().write_buffer(block.buffer(), off, &bytes);
        return Ok(());
    }
    let start = off & !3;
    let len = round4(off + T::SIZE as u64) - start;
    let mut window = vec![0u8; len as usize];
    copy_dev2host_bytes(ctx, &mut window, block.buffer(), start)?;
    let at = (off - start) as usize;
    window[at..at + T::SIZE].copy_from_slice(&bytes);
    ctx.queue().write_buffer(block.buffer(), start, &window);
    Ok(())
}

/// `memcpy_gpu2host`: `dst.len()` bytes of `src` from `src_offset`, through a mapped buffer padded to 4 bytes.
fn copy_dev2host_bytes(ctx: &FutharkCtx, dst: &mut [u8], src: &wgpu::Buffer, src_offset: u64) -> Result<(), Error> {
    if dst.is_empty() {
        return Ok(());
    }
    let size = round4(dst.len() as u64);
    let readback = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("tmp_readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("futhark read-back") });
    enc.copy_buffer_to_buffer(src, src_offset, &readback, 0, Some(size));
    ctx.queue().submit(Some(enc.finish()));
    ctx.count(|s| s.submits += 1);
    ctx.map_read(&readback, size)?;
    {
        let view = crate::compat::mapped(&readback, 0..size)?;
        let n = dst.len();
        dst.copy_from_slice(&view[..n]);
    }
    readback.unmap();
    readback.destroy();
    Ok(())
}

/// `memcpy_host2gpu`: `src` written at `dst_offset`, padded to 4 bytes with zeros (the C runtime pads with the bytes
/// after `src`, and warns that the padding may overwrite data that follows; so does this, on the same condition).
fn copy_host2dev_bytes(ctx: &FutharkCtx, dst: &wgpu::Buffer, dst_offset: u64, src: &[u8]) -> Result<(), Error> {
    if src.is_empty() {
        return Ok(());
    }
    let size = round4(src.len() as u64);
    if size != src.len() as u64 && dst_offset + size != dst.size() {
        eprintln!("memcpy_host2gpu: Potentially could corrupt data due to padding!");
    }
    if size == src.len() as u64 {
        ctx.queue().write_buffer(dst, dst_offset, src);
    } else {
        let mut padded = src.to_vec();
        padded.resize(size as usize, 0);
        ctx.queue().write_buffer(dst, dst_offset, &padded);
    }
    Ok(())
}

/// `gpu_memcpy`: a flat device copy, rounded up to 4 bytes; a copy within one buffer goes through a temporary.
fn memcpy_dev2dev(ctx: &FutharkCtx, dst: &DevMem, dst_offset: u64, src: &DevMem, src_offset: u64, nbytes: u64) -> Result<(), Error> {
    let size = round4(nbytes);
    let (db, sb) = (dst.block()?, src.block()?);
    if size != nbytes && dst_offset + size != db.size() {
        eprintln!("gpu_memcpy: Potentially could corrupt data due to padding!");
    }
    let mut enc = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("futhark copy") });
    if dst.same_block(src) {
        let tmp = ctx.new_storage(size.max(4))?;
        enc.copy_buffer_to_buffer(sb.buffer(), src_offset, &tmp, 0, Some(size));
        enc.copy_buffer_to_buffer(&tmp, 0, db.buffer(), dst_offset, Some(size));
        ctx.queue().submit(Some(enc.finish()));
        ctx.sync()?;
        tmp.destroy();
    } else {
        enc.copy_buffer_to_buffer(sb.buffer(), src_offset, db.buffer(), dst_offset, Some(size));
        ctx.queue().submit(Some(enc.finish()));
    }
    ctx.count(|s| s.submits += 1);
    Ok(())
}

// ---- the classifiers of copy.h ------------------------------------------------------------------------------

fn lmad_is_tr(strides: &[i64], shape: &[i64]) -> Option<(i64, i64)> {
    let r = strides.len();
    for i in 1..r {
        let (mut n, mut m) = (1i64, 1i64);
        let mut ok = true;
        let mut expected = 1i64;
        for j in (0..i).rev() {
            ok = ok && strides[j] == expected;
            expected = expected.wrapping_mul(shape[j]);
            n = n.wrapping_mul(shape[j]);
        }
        for j in (i..r).rev() {
            ok = ok && strides[j] == expected;
            expected = expected.wrapping_mul(shape[j]);
            m = m.wrapping_mul(shape[j]);
        }
        if ok {
            return Some((n, m));
        }
    }
    None
}

/// `lmad_map_tr`: `(num_arrays, n, m)` when the copy is a batch of transpositions.
fn lmad_map_tr(dst_strides: &[i64], src_strides: &[i64], shape: &[i64]) -> Option<(i64, i64, i64)> {
    let r = shape.len();
    let mut rowmajor = vec![0i64; r];
    rowmajor[r - 1] = 1;
    for i in (0..r.saturating_sub(1)).rev() {
        rowmajor[i] = rowmajor[i + 1].wrapping_mul(shape[i + 1]);
    }
    let mut map_r = 0;
    let mut num_arrays = 1i64;
    for i in 0..r {
        if dst_strides[i] != rowmajor[i] || src_strides[i] != rowmajor[i] {
            break;
        }
        num_arrays = num_arrays.wrapping_mul(shape[i]);
        map_r += 1;
    }
    if r == map_r {
        return None;
    }
    if rowmajor[map_r..] == dst_strides[map_r..] {
        lmad_is_tr(&src_strides[map_r..], &shape[map_r..]).map(|(n, m)| (num_arrays, n, m))
    } else if rowmajor[map_r..] == src_strides[map_r..] {
        lmad_is_tr(&dst_strides[map_r..], &shape[map_r..]).map(|(n, m)| (num_arrays, m, n))
    } else {
        None
    }
}

fn lmad_contiguous_search(checked: usize, expected: i64, strides: &[i64], shape: &[i64], used: &mut [bool]) -> bool {
    let r = strides.len();
    for _ in 0..r {
        for j in 0..r {
            if !used[j] && strides[j] == expected && strides[j] >= 0 {
                used[j] = true;
                if checked + 1 == r || lmad_contiguous_search(checked + 1, expected.wrapping_mul(shape[j]), strides, shape, used) {
                    return true;
                }
                used[j] = false;
            }
        }
    }
    false
}

fn lmad_contiguous(strides: &[i64], shape: &[i64]) -> bool {
    let mut used = vec![false; strides.len()];
    lmad_contiguous_search(0, 1, strides, shape, &mut used)
}

fn lmad_memcpyable(dst_strides: &[i64], src_strides: &[i64], shape: &[i64]) -> bool {
    lmad_contiguous(dst_strides, shape) && (0..shape.len()).all(|i| dst_strides[i] == src_strides[i] || shape[i] == 1)
}

// ---- device to device ---------------------------------------------------------------------------------------

/// `gpu_map_transpose`: `k` transpositions of `n` by `m` from `src` to `dst` (each a memory and an element offset),
/// the kernel variant chosen as the C runtime chooses it.
fn map_transpose(ctx: &FutharkCtx, prog: &Program, suffix: &str, dst: (&DevMem, i64), src: (&DevMem, i64), d: TrDims) -> Result<(), Error> {
    let (dst, dst_offset) = dst;
    let (src, src_offset) = src;
    let TrDims { k, n, m } = d;
    let mulx = TR_BLOCK_DIM / n;
    let muly = TR_BLOCK_DIM / m;
    let small = dst_offset + k * n * m <= 2147483647 && src_offset + k * n * m <= 2147483647;
    let tiled = |name: String| (name, [(m + TR_TILE_DIM - 1) / TR_TILE_DIM, (n + TR_TILE_DIM - 1) / TR_TILE_DIM, k], [TR_TILE_DIM, TR_TILE_DIM / TR_ELEMS_PER_THREAD, 1]);
    let (name, grid, block): (String, [i64; 3], [i64; 3]) = if !small {
        tiled(format!("map_transpose_{suffix}_large"))
    } else if m <= TR_BLOCK_DIM / 2 && n <= TR_BLOCK_DIM / 2 {
        (
            format!("map_transpose_{suffix}_small"),
            [(k * n * m + TR_BLOCK_DIM * TR_BLOCK_DIM - 1) / (TR_BLOCK_DIM * TR_BLOCK_DIM), 1, 1],
            [TR_BLOCK_DIM * TR_BLOCK_DIM, 1, 1],
        )
    } else if m <= TR_BLOCK_DIM / 2 && TR_BLOCK_DIM < n {
        let x_elems = m;
        let y_elems = (n + muly - 1) / muly;
        (
            format!("map_transpose_{suffix}_low_width"),
            [(x_elems + TR_BLOCK_DIM - 1) / TR_BLOCK_DIM, (y_elems + TR_BLOCK_DIM - 1) / TR_BLOCK_DIM, k],
            [TR_BLOCK_DIM, TR_BLOCK_DIM, 1],
        )
    } else if n <= TR_BLOCK_DIM / 2 && TR_BLOCK_DIM < m {
        let x_elems = (m + mulx - 1) / mulx;
        let y_elems = n;
        (
            format!("map_transpose_{suffix}_low_height"),
            [(x_elems + TR_BLOCK_DIM - 1) / TR_BLOCK_DIM, (y_elems + TR_BLOCK_DIM - 1) / TR_BLOCK_DIM, k],
            [TR_BLOCK_DIM, TR_BLOCK_DIM, 1],
        )
    } else {
        tiled(format!("map_transpose_{suffix}"))
    };
    // The C runtime holds the grid in int32_t.
    let mut g = [grid[0] as i32, grid[1] as i32, grid[2] as i32];
    let repeat_1 = g[1] / MAX_TR_THREAD_BLOCKS;
    let repeat_2 = g[2] / MAX_TR_THREAD_BLOCKS;
    if repeat_1 > 0 {
        g[1] = MAX_TR_THREAD_BLOCKS;
    }
    if repeat_2 > 0 {
        g[2] = MAX_TR_THREAD_BLOCKS;
    }
    let mut args = vec![KArg::I64(dst_offset), KArg::I64(src_offset)];
    if small {
        args.extend([KArg::I32(k as i32), KArg::I32(m as i32), KArg::I32(n as i32), KArg::I32(mulx as i32), KArg::I32(muly as i32)]);
    } else {
        args.extend([KArg::I64(k), KArg::I64(m), KArg::I64(n), KArg::I64(mulx), KArg::I64(muly)]);
    }
    args.extend([KArg::I32(repeat_1), KArg::I32(repeat_2)]);
    let idx = prog.kernel_index(&name)?;
    let u = |x: i32| x as u32;
    launch(ctx, prog, idx, [u(g[0]), u(g[1]), u(g[2])], [block[0] as u32, block[1] as u32, block[2] as u32], &args, &[dst, src])
}

/// `lmad_copy_gpu2gpu_<n>b`: a device copy of `elem_bytes`-byte elements, LMAD to LMAD.
pub fn lmad_copy_dev2dev(ctx: &FutharkCtx, prog: &Program, elem_bytes: usize, dst: Lmad<'_, DevMem>, src: Lmad<'_, DevMem>, shape: &[i64]) -> Result<(), Error> {
    let Lmad { mem: dst, offset: dst_offset, strides: dst_strides } = dst;
    let Lmad { mem: src, offset: src_offset, strides: src_strides } = src;
    let suffix = match elem_bytes {
        1 => "1b",
        2 => "2b",
        4 => "4b",
        8 => "8b",
        n => return Err(Error::new(format!("our software has a bug: a copy of {n}-byte elements"))),
    };
    let size: i64 = shape.iter().product();
    if size == 0 {
        return Ok(());
    }
    if let Some((k, n, m)) = lmad_map_tr(dst_strides, src_strides, shape) {
        // lmad_copy_gpu2gpu calls map_transpose_gpu2gpu(k, n, m), whose parameters are (k, m, n), and which calls
        // gpu_map_transpose(k, n, m) with its own n and m: the transposition is of m by n.
        map_transpose(ctx, prog, suffix, (dst, dst_offset), (src, src_offset), TrDims { k, n: m, m: n })
    } else if lmad_memcpyable(dst_strides, src_strides, shape) {
        let e = elem_bytes as i64;
        memcpy_dev2dev(ctx, dst, (dst_offset * e) as u64, src, (src_offset * e) as u64, (size * e) as u64)
    } else {
        let r = shape.len();
        if r > 8 {
            return Err(Error::new("Futhark runtime limitation:\nCannot copy array of greater than rank 8.\n"));
        }
        let mut args = vec![KArg::I64(dst_offset), KArg::I64(src_offset), KArg::I64(size), KArg::I32(r as i32)];
        for i in 0..8 {
            if i < r {
                args.extend([KArg::I64(shape[i]), KArg::I64(dst_strides[i]), KArg::I64(src_strides[i])]);
            } else {
                args.extend([KArg::I64(0), KArg::I64(0), KArg::I64(0)]);
            }
        }
        let idx = prog.kernel_index(&format!("lmad_copy_{suffix}"))?;
        let w = 256i64;
        launch(ctx, prog, idx, [((size + w - 1) / w) as i32 as u32, 1, 1], [w as u32, 1, 1], &args, &[dst, src])
    }
}

/// `lmad_copy_host2gpu`: a flat host-to-device copy (the C runtime has no other).
pub fn lmad_copy_host2dev(ctx: &FutharkCtx, elem_bytes: usize, dst: Lmad<'_, DevMem>, src: Lmad<'_, HostMem>, shape: &[i64]) -> Result<(), Error> {
    let Lmad { mem: dst, offset: dst_offset, strides: dst_strides } = dst;
    let Lmad { mem: src, offset: src_offset, strides: src_strides } = src;
    let size = elem_bytes as i64 * shape.iter().product::<i64>();
    if size == 0 {
        return Ok(());
    }
    if !lmad_memcpyable(dst_strides, src_strides, shape) {
        return Err(Error::new("Futhark runtime limitation:\nCannot copy unstructured array from host to GPU.\n"));
    }
    let e = elem_bytes as i64;
    let bytes = src.bytes()?;
    let bytes = bytes.borrow();
    let from = (src_offset * e) as usize;
    let part = bytes.get(from..from + size as usize).ok_or_else(|| Error::new("a host-to-device copy past its host block"))?;
    copy_host2dev_bytes(ctx, dst.block()?.buffer(), (dst_offset * e) as u64, part)
}

/// `lmad_copy_gpu2host`: a flat device-to-host copy (the C runtime has no other).
pub fn lmad_copy_dev2host(ctx: &FutharkCtx, elem_bytes: usize, dst: Lmad<'_, HostMem>, src: Lmad<'_, DevMem>, shape: &[i64]) -> Result<(), Error> {
    let Lmad { mem: dst, offset: dst_offset, strides: dst_strides } = dst;
    let Lmad { mem: src, offset: src_offset, strides: src_strides } = src;
    let size = elem_bytes as i64 * shape.iter().product::<i64>();
    if size == 0 {
        return Ok(());
    }
    if !lmad_memcpyable(dst_strides, src_strides, shape) {
        return Err(Error::new("Futhark runtime limitation:\nCannot copy unstructured array from GPU to host.\n"));
    }
    let e = elem_bytes as i64;
    let bytes = dst.bytes()?;
    let mut bytes = bytes.borrow_mut();
    let to = (dst_offset * e) as usize;
    let part = bytes.get_mut(to..to + size as usize).ok_or_else(|| Error::new("a device-to-host copy past its host block"))?;
    copy_dev2host_bytes(ctx, part, src.block()?.buffer(), (src_offset * e) as u64)
}

/// A host-to-host copy, element by element (GenericC's `compileCopyWith`).
pub fn lmad_copy_host2host(elem_bytes: usize, dst: Lmad<'_, HostMem>, src: Lmad<'_, HostMem>, shape: &[i64]) -> Result<(), Error> {
    let Lmad { mem: dst, offset: dst_offset, strides: dst_strides } = dst;
    let Lmad { mem: src, offset: src_offset, strides: src_strides } = src;
    let s = src.bytes()?;
    let d = dst.bytes()?;
    let s = s.borrow().clone();
    let mut d = d.borrow_mut();
    let r = shape.len();
    let total: i64 = shape.iter().product();
    for flat in 0..total.max(0) {
        let (mut rem, mut di, mut si) = (flat, dst_offset, src_offset);
        for j in (0..r).rev() {
            let x = rem % shape[j];
            rem /= shape[j];
            di += x * dst_strides[j];
            si += x * src_strides[j];
        }
        let (di, si) = (di as usize * elem_bytes, si as usize * elem_bytes);
        let from = s.get(si..si + elem_bytes).ok_or_else(|| Error::new("a host copy past its source"))?;
        let n = d.len();
        d.get_mut(di..di + elem_bytes).ok_or_else(|| Error::new(format!("a host copy past its {n}-byte destination")))?.copy_from_slice(from);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_classifiers_are_copy_h_s() {
        // Over the shape [5, 3]: a row-major destination and a column-major source (n = 5, m = 3), and the reverse.
        assert_eq!(lmad_map_tr(&[3, 1], &[1, 5], &[5, 3]), Some((1, 5, 3)));
        assert_eq!(lmad_map_tr(&[1, 5], &[3, 1], &[5, 3]), Some((1, 3, 5)));
        // A flat copy is not a transposition, and is memcpy-able.
        assert_eq!(lmad_map_tr(&[5, 1], &[5, 1], &[3, 5]), None);
        assert!(lmad_memcpyable(&[5, 1], &[5, 1], &[3, 5]));
        // A strided source is neither.
        assert!(!lmad_memcpyable(&[1], &[2], &[7]));
        // Size-1 dimensions may have any stride.
        assert!(lmad_memcpyable(&[1, 1], &[9, 1], &[1, 4]));
    }

    #[test]
    fn host_copies_follow_the_lmads() {
        let src = HostMem::from_values::<i32>(&[0, 1, 2, 3, 4, 5]);
        let dst = HostMem::zeros(24);
        // A transposition of the 2x3 source into 3x2.
        lmad_copy_host2host(4, Lmad::new(&dst, 0, &[1, 2]), Lmad::new(&src, 0, &[3, 1]), &[2, 3]).expect("the copy");
        let got: Vec<i32> = (0..6).map(|i| dst.read::<i32>(i).expect("a read")).collect();
        assert_eq!(got, vec![0, 3, 1, 4, 2, 5]);
    }
}
