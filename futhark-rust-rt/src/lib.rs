//! THE RUNTIME of Futhark's `rust-wgpu` backend (the fork's `src/Futhark/CodeGen/Backends/RustWGPU.hs`,
//! `docs/PLAN_futhark_rust_wgpu.md`).
//!
//! A generated module is one Futhark program: WGSL kernels, a kernel table, and host code as Rust functions over this
//! crate. It runs on a `wgpu::Device` and `wgpu::Queue` the caller gives ([`FutharkCtx::new`]), on buffers the caller
//! may own ([`Array::from_buffer`]). Its host semantics are those of Futhark's C runtime for the `webgpu` backend
//! (`rts/c/gpu.h`, `rts/c/backends/webgpu.h`), reimplemented here: the same allocation with a free list, the same
//! scalar layout of a launch, the same override constants, the same copies and transpositions. The differences, each
//! named where it is made:
//!
//! * a kernel's pipeline is created on its first launch, not when the context is made, and every pipeline is cached
//!   (the C runtime creates a pipeline at every launch of a kernel with dynamic block sizes);
//! * shader modules are created once per text per context, and the WGSL is ported for naga first ([`port`]);
//! * a scalar read or write at an offset that is not a multiple of 4 is done on the aligned word around it (WebGPU's
//!   copies need 4-byte alignment; the C runtime issues the unaligned copy, which WebGPU rejects).

#[cfg(all(feature = "wgpu27", feature = "wgpu30"))]
compile_error!("futhark-rust-rt: choose one wgpu, the feature wgpu30 (the default) or wgpu27 (with default-features = false)");
#[cfg(not(any(feature = "wgpu27", feature = "wgpu30")))]
compile_error!("futhark-rust-rt: choose a wgpu, the feature wgpu30 (the default) or wgpu27");
/// The wgpu this runtime is built on (27.0.1, UltraDim's), under the name the runtime's code uses, for a caller that
/// must name the same types.
#[cfg(feature = "wgpu27")]
pub extern crate wgpu27 as wgpu;

mod compat;
pub mod copy;
pub mod ctx;
pub mod host;
pub mod mem;
pub mod port;
pub mod program;
pub mod sc;
pub mod values;

pub use copy::{lmad_copy_dev2dev, lmad_copy_dev2host, lmad_copy_host2dev, lmad_copy_host2host, read_scalar, write_scalar, Lmad};
pub use ctx::{CtxStats, FutharkCtx};
pub use mem::{DevMem, HostMem};
pub use program::{launch, BuiltinSource, KArg, KernelInfo, Program, ProgramDesc, TuningParam};
pub use values::{check_arity, check_dim, Array, Elem, ElemType, EntryDesc, IntoValues, Opaque, TypeDesc, Value};
/// The wgpu this runtime is built on (30.0.1), for a caller that must name the same types.
#[cfg(feature = "wgpu30")]
pub use wgpu;

/// A scalar variable of the generated host code: set and read, never borrowed.
pub type Var<T> = std::cell::Cell<T>;

/// Everything that goes wrong: a failed assertion of the program, a device or validation error, a misuse of the API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    msg: String,
    device: bool,
}

impl Error {
    pub fn new(msg: impl Into<String>) -> Error {
        Error { msg: msg.into(), device: false }
    }
    /// A failure of the device itself (a wait past its bound, a lost device): the device should not be used again.
    pub fn device(msg: impl Into<String>) -> Error {
        Error { msg: msg.into(), device: true }
    }
    pub fn is_device_failure(&self) -> bool {
        self.device
    }
    pub fn message(&self) -> &str {
        &self.msg
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Error {}
