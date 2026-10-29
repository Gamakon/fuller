//! A GENERATED PROGRAM on a context: its tuning, its override constants, its kernels, and the launch.
//!
//! [`ProgramDesc`] is what a generated module states (the WGSL, the kernel table, the tuning parameters, the macro
//! definitions); [`Program`] is that program in one context, made by the module's `state`. [`launch`] is
//! `gpu_launch_kernel` of `rts/c/backends/webgpu.h`: the scalars written into the kernel's uniform buffer at the
//! kernel's offsets, the memory bound at its binding indices, the override constants the kernel uses, the dynamic
//! block sizes and shared memory sizes as further overrides, one compute pass, one submission.

use std::cell::RefCell;
use std::rc::Rc;

use crate::ctx::{Layouts, Module};
use crate::{DevMem, Error, FutharkCtx};

/// A kernel's interface, as Futhark's `KernelInterface` (`Futhark.CodeGen.ImpCode.WebGPU`) states it.
#[derive(Debug)]
pub struct KernelInfo {
    /// The WGSL entry point.
    pub name: &'static str,
    pub scalars_binding: u32,
    pub scalars_size: u64,
    pub scalar_offsets: &'static [u64],
    pub binding_indices: &'static [u32],
    /// The override constants the kernel uses (only these are given: WebGPU rejects an unknown override).
    pub used_overrides: &'static [&'static str],
    /// The block dimensions given at each launch: (dimension, override).
    pub dynamic_block_dims: &'static [(u32, &'static str)],
    /// The overrides holding shared memory sizes, given at each launch by the first arguments.
    pub shared_mem_overrides: &'static [&'static str],
    /// A built-in kernel (a copy or a transposition): its own WGSL module, instantiated from a template.
    pub builtin: Option<BuiltinSource>,
}

/// A built-in kernel's WGSL: the prelude, then the template with `NAME` and `ELEM_TYPE` replaced (and `<ELEM_TYPE>`
/// made atomic for 1- and 2-byte elements), as `CWebGPU.builtinKernels` writes it.
#[derive(Debug)]
pub struct BuiltinSource {
    pub template: &'static str,
    pub name: &'static str,
    pub elem: &'static str,
    pub atomic: bool,
}

impl BuiltinSource {
    pub fn wgsl(&self, prelude: &str) -> String {
        let base = if self.atomic { self.template.replace("<ELEM_TYPE>", "<atomic<ELEM_TYPE>>") } else { self.template.to_string() };
        format!("{prelude}{}", base.replace("ELEM_TYPE", self.elem).replace("NAME", self.name))
    }
}

/// A tuning parameter: its name, its class as Futhark prints it, its default (0: the backend decides).
#[derive(Debug)]
pub struct TuningParam {
    pub name: &'static str,
    pub class: &'static str,
    pub default: i64,
}

/// What a generated module states about its program.
pub struct ProgramDesc {
    pub wgsl: &'static str,
    pub prelude: &'static str,
    pub kernels: &'static [KernelInfo],
    pub tuning: &'static [TuningParam],
    pub macro_names: &'static [&'static str],
    /// The macro definitions' values, given the tuning values and whether each was set by the user.
    pub macro_values: fn(&FutharkCtx, &[i64], &[bool]) -> Vec<f64>,
}

const MARK: &str = "/*futhark-specialise:";

/// The WGSL with every specialisation constant filled in: `/*futhark-specialise:NAME*/ 0` becomes the value of the
/// override constant `NAME` (a register array's size, a tuning parameter fixed when the module is made; the fork's
/// `scalarSpaceSize`). A marker naming no macro of the program is an error.
pub fn specialise(wgsl: &str, macros: &[(&str, f64)]) -> Result<String, Error> {
    let mut out = String::with_capacity(wgsl.len());
    let mut rest = wgsl;
    while let Some(i) = rest.find(MARK) {
        out.push_str(&rest[..i]);
        let after = &rest[i + MARK.len()..];
        let end = after.find("*/ 0").ok_or_else(|| Error::new("a specialisation marker with no `*/ 0`"))?;
        let name = &after[..end];
        let v = macros.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).ok_or_else(|| Error::new(format!("a specialisation constant of no macro: {name}")))?;
        if v.fract() != 0.0 || !(1.0..=f64::from(i32::MAX)).contains(&v) {
            return Err(Error::new(format!("the specialisation constant {name} = {v} is not a positive i32")));
        }
        out.push_str(&format!("{}", v as i32));
        rest = &after[end + "*/ 0".len()..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A kernel ready to launch.
struct KernelRt {
    module: Rc<Module>,
    layouts: Rc<Layouts>,
    /// The macro overrides this kernel uses, in the program's macro order.
    base: Vec<(&'static str, f64)>,
    scalars: wgpu::Buffer,
    /// Kernels with no dynamic block size and no shared memory override have one pipeline.
    static_pipeline: RefCell<Option<wgpu::ComputePipeline>>,
}

/// A program in one context.
pub struct Program {
    desc: &'static ProgramDesc,
    /// The program's WGSL, its specialisation constants filled in ([`specialise`]).
    wgsl: String,
    tuning: Vec<i64>,
    tuning_set: Vec<bool>,
    macros: Vec<(&'static str, f64)>,
    kernels: RefCell<Vec<Option<Rc<KernelRt>>>>,
}

/// The default of a tuning class when the program gives none (`wgpu_size_setup`): Futhark's GPU defaults, a grid
/// twice as large for histograms.
fn class_default(ctx: &FutharkCtx, name: &str, class: &str) -> i64 {
    if class.starts_with("thread_block_size") {
        256
    } else if class.starts_with("grid_size") {
        if name.contains(".seghist_") {
            32
        } else {
            16
        }
    } else if class.starts_with("tile_size") {
        32
    } else if class.starts_with("reg_tile_size") {
        2
    } else if class.starts_with("shared_memory") {
        ctx.size_max("shared_memory")
    } else if class.starts_with("cache") {
        ctx.size_max("cache")
    } else if class.starts_with("threshold") {
        32 * 1024
    } else {
        0
    }
}

impl Program {
    pub fn new(ctx: &FutharkCtx, desc: &'static ProgramDesc) -> Result<Program, Error> {
        let tuning: Vec<i64> = desc
            .tuning
            .iter()
            .map(|t| if t.default == 0 { class_default(ctx, t.name, t.class) } else { t.default })
            .collect();
        let tuning_set = vec![false; tuning.len()];
        let values = (desc.macro_values)(ctx, &tuning, &tuning_set);
        let macros: Vec<(&'static str, f64)> = desc.macro_names.iter().copied().zip(values).collect();
        let wgsl = specialise(desc.wgsl, &macros)?;
        Ok(Program { desc, wgsl, tuning, tuning_set, macros, kernels: RefCell::new(vec![None; desc.kernels.len()]) })
    }

    pub fn desc(&self) -> &'static ProgramDesc {
        self.desc
    }
    /// The program's WGSL as its module is made: specialised ([`specialise`]).
    pub fn wgsl(&self) -> &str {
        &self.wgsl
    }
    pub fn tuning(&self, i: usize) -> i64 {
        self.tuning[i]
    }
    pub fn tuning_set(&self, i: usize) -> bool {
        self.tuning_set[i]
    }
    pub fn tunings(&self) -> &[i64] {
        &self.tuning
    }
    pub fn tunings_set(&self) -> &[bool] {
        &self.tuning_set
    }

    /// The index of a kernel by its WGSL entry point name (the built-in copies and transpositions).
    pub fn kernel_index(&self, name: &str) -> Result<usize, Error> {
        self.desc
            .kernels
            .iter()
            .position(|k| k.name == name)
            .ok_or_else(|| Error::new(format!("our software has a bug: the program has no kernel {name}")))
    }

    fn kernel(&self, ctx: &FutharkCtx, idx: usize) -> Result<Rc<KernelRt>, Error> {
        if let Some(k) = &self.kernels.borrow()[idx] {
            return Ok(k.clone());
        }
        let info = &self.desc.kernels[idx];
        let module = match &info.builtin {
            Some(b) => ctx.module(&b.wgsl(self.desc.prelude), info.name)?,
            None => ctx.module(&self.wgsl, "Futhark program")?,
        };
        let layouts = ctx.layouts(info.scalars_binding, info.scalars_size, info.binding_indices);
        let base = self.macros.iter().filter(|(n, _)| info.used_overrides.contains(n)).copied().collect();
        let scalars = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel scalars"),
            size: info.scalars_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k = Rc::new(KernelRt { module, layouts, base, scalars, static_pipeline: RefCell::new(None) });
        self.kernels.borrow_mut()[idx] = Some(k.clone());
        Ok(k)
    }
}

/// A kernel's scalar argument, in its storage type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KArg {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    /// A unit scalar, stored as the C backend stores it: a one-byte `bool`, zero.
    Unit(()),
}

impl KArg {
    fn write(&self, out: &mut Vec<u8>, at: usize) {
        let bytes: Vec<u8> = match *self {
            KArg::I8(x) => x.to_le_bytes().to_vec(),
            KArg::I16(x) => x.to_le_bytes().to_vec(),
            KArg::I32(x) => x.to_le_bytes().to_vec(),
            KArg::I64(x) => x.to_le_bytes().to_vec(),
            KArg::F32(x) => x.to_le_bytes().to_vec(),
            KArg::F64(x) => x.to_le_bytes().to_vec(),
            KArg::Bool(x) => vec![x as u8],
            KArg::Unit(()) => vec![0],
        };
        if out.len() < at + bytes.len() {
            out.resize(at + bytes.len(), 0);
        }
        out[at..at + bytes.len()].copy_from_slice(&bytes);
    }

    /// A shared memory size argument, read as the C runtime reads it (`*(int32_t *) args[i]`).
    fn as_i32(&self) -> Result<i32, Error> {
        match *self {
            KArg::I32(x) => Ok(x),
            other => Err(Error::new(format!("our software has a bug: a shared memory size argument {other:?}"))),
        }
    }
}

/// Launch kernel `idx` of the program: `grid` blocks of `block` threads. A launch with any dimension 0 does nothing,
/// as the C kernel wrapper skips it. `vals` are the shared memory sizes then the scalars; `mems` the memory, in the
/// kernel's binding order.
pub fn launch(ctx: &FutharkCtx, prog: &Program, idx: usize, grid: [u32; 3], block: [u32; 3], vals: &[KArg], mems: &[&DevMem]) -> Result<(), Error> {
    // The C wrapper's `grid_x * grid_y * grid_z * block_x * block_y * block_z != 0`, in unsigned arithmetic.
    if grid.iter().chain(block.iter()).fold(1u32, |a, &b| a.wrapping_mul(b)) == 0 {
        ctx.count(|s| s.skipped_launches += 1);
        return Ok(());
    }
    let info = &prog.desc.kernels[idx];
    let k = prog.kernel(ctx, idx)?;
    let nsh = info.shared_mem_overrides.len();
    let nsc = info.scalar_offsets.len();
    if vals.len() != nsh + nsc || mems.len() != info.binding_indices.len() {
        return Err(Error::new(format!("Kernel {} called with num_args not maching its info", info.name)));
    }
    let mut scalars = vec![0u8; info.scalars_size as usize];
    for (i, &off) in info.scalar_offsets.iter().enumerate() {
        vals[nsh + i].write(&mut scalars, off as usize);
    }
    if scalars.len() as u64 != info.scalars_size {
        return Err(Error::new(format!("the scalars of kernel {} overrun its {} bytes", info.name, info.scalars_size)));
    }
    let blocks: Vec<std::rc::Rc<crate::mem::Block>> = mems.iter().map(|m| m.block()).collect::<Result<_, _>>()?;

    let pipeline = if info.dynamic_block_dims.is_empty() && nsh == 0 {
        let cached = k.static_pipeline.borrow().clone();
        match cached {
            Some(p) => p,
            None => {
                let p = ctx.pipeline(&k.module, &k.layouts, info.name, &k.base)?;
                *k.static_pipeline.borrow_mut() = Some(p.clone());
                p
            }
        }
    } else {
        let mut constants = k.base.clone();
        for &(dim, name) in info.dynamic_block_dims {
            constants.push((name, block[dim as usize] as f64));
        }
        for (i, &name) in info.shared_mem_overrides.iter().enumerate() {
            constants.push((name, vals[i].as_i32()? as f64));
        }
        ctx.pipeline(&k.module, &k.layouts, info.name, &constants)?
    };

    let device = ctx.device();
    let scope = crate::compat::push_scope(device, wgpu::ErrorFilter::Validation);
    ctx.queue().write_buffer(&k.scalars, 0, &scalars);
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: info.scalars_binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &k.scalars, offset: 0, size: None }),
    }];
    for (b, &binding) in blocks.iter().zip(info.binding_indices) {
        entries.push(wgpu::BindGroupEntry { binding, resource: b.buffer().as_entire_binding() });
    }
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(info.name), layout: &k.layouts.bind_group, entries: &entries });
    let timed = ctx.timing().then(|| crate::copy::PassTimer::new(device));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(info.name) });
    {
        let writes = timed.as_ref().map(|t| t.writes());
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(info.name), timestamp_writes: writes });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
    }
    ctx.queue().submit(Some(enc.finish()));
    if let Some(e) = scope.pop() {
        return Err(Error::new(format!("the launch of kernel {} failed: {e}", info.name)));
    }
    ctx.count(|s| {
        s.launches += 1;
        s.submits += 1;
    });
    if let Some(t) = timed {
        let ns = t.read(ctx)?;
        ctx.record_pass(ns);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specialisation_fills_each_marker_from_its_macro() {
        let w = "const k_rx_spec: i32 = /*futhark-specialise:k_rx_lo*/ 0;\nvar x: array<f32, (k_rx_spec * k_rx_spec)>;";
        let got = specialise(w, &[("other", 9.0), ("k_rx_lo", 2.0)]).expect("specialised");
        assert_eq!(got, "const k_rx_spec: i32 = 2;\nvar x: array<f32, (k_rx_spec * k_rx_spec)>;");
        assert!(specialise(w, &[("other", 9.0)]).is_err(), "a marker of no macro");
        assert!(specialise(w, &[("k_rx_lo", 0.0)]).is_err(), "a size of 0");
        assert_eq!(specialise("no markers", &[]).expect("unchanged"), "no markers");
    }
}
