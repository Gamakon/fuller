//! What a C handle points at.
//!
//! Every `WGPU*` handle is a `Box::into_raw` of one of these. They are opaque
//! to C (`cbindgen` writes `typedef struct WGPUBufferImpl WGPUBufferImpl;`).
//!
//! Futhark's runtime **never releases a handle** — the surface has no
//! `wgpu*Release` — so a handle lives until the process ends. The GPU memory
//! behind a buffer is freed by `wgpuBufferDestroy`, which the runtime does
//! call; what stays behind per kernel launch is the small `Box` of an encoder,
//! a pass, a command buffer and a bind group. That is the runtime's choice,
//! and it is recorded in `README.md`, not hidden here.

use crate::futures::Registry;
use crate::report::TimedPass;
use std::cell::RefCell;
use std::rc::Rc;

pub struct WGPUInstanceImpl {
    pub instance: wgpu::Instance,
    pub registry: Rc<Registry>,
}

pub struct WGPUAdapterImpl {
    pub adapter: wgpu::Adapter,
    pub registry: Rc<Registry>,
}

pub struct WGPUDeviceImpl {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub registry: Rc<Registry>,
    /// Whether this device times every compute pass on the GPU
    /// (`FUTHARK_WGPU_SHIM_TIMESTAMPS=1`).
    pub timestamps: bool,
    /// The queue handle, made once: `wgpuDeviceGetQueue` returns the same one
    /// every time.
    pub queue_handle: RefCell<Option<*mut WGPUQueueImpl>>,
    /// The cell the uncaptured-error callback's `WGPUDevice const *` points at.
    pub self_cell: Box<*mut WGPUDeviceImpl>,
    /// THE PIPELINE CACHE: the pipeline made for each compute-pipeline descriptor, by its layout, its module, its
    /// entry point and its constants (bit for bit). Futhark's runtime creates the pipeline of a kernel with dynamic
    /// block sizes AT EVERY LAUNCH (measured 2026-10-28: 6,210 creations, 14.5 s, in one process of 12 histogram
    /// programs), always from the same few descriptors; an identical descriptor gets the pipeline already made,
    /// which is the same program with the same constants. Safe because the runtime never releases a pipeline (the
    /// surface has no `wgpuComputePipelineRelease`), so a shared handle is never freed under another user.
    pub pipelines: RefCell<std::collections::HashMap<PipelineKey, *mut WGPUComputePipelineImpl>>,
}

/// A compute-pipeline descriptor's identity: the layout and module handles, the entry point, each constant's key and
/// value bits, in the descriptor's order.
pub type PipelineKey = (usize, usize, Option<String>, Vec<(String, u64)>);

pub struct WGPUQueueImpl {
    pub queue: wgpu::Queue,
    pub device: wgpu::Device,
    pub registry: Rc<Registry>,
}

pub struct WGPUBufferImpl {
    pub buffer: wgpu::Buffer,
    pub device: wgpu::Device,
    pub registry: Rc<Registry>,
    /// The views `wgpuBufferGetConstMappedRange` handed out. The C side holds
    /// raw pointers into them, so they live until `wgpuBufferUnmap`.
    pub views: RefCell<Vec<wgpu::BufferView>>,
}

pub struct WGPUShaderModuleImpl {
    pub module: wgpu::ShaderModule,
    pub registry: Rc<Registry>,
}

pub struct WGPUBindGroupLayoutImpl {
    pub layout: wgpu::BindGroupLayout,
}

pub struct WGPUPipelineLayoutImpl {
    pub layout: wgpu::PipelineLayout,
}

pub struct WGPUComputePipelineImpl {
    pub pipeline: wgpu::ComputePipeline,
}

pub struct WGPUBindGroupImpl {
    pub group: wgpu::BindGroup,
}

pub struct WGPUCommandEncoderImpl {
    /// `None` once `wgpuCommandEncoderFinish` has taken it.
    pub encoder: RefCell<Option<wgpu::CommandEncoder>>,
    pub device: wgpu::Device,
    pub timestamps: bool,
    /// The passes recorded with GPU timestamps, waiting for the submit.
    pub timed: RefCell<Vec<TimedPass>>,
}

pub struct WGPUComputePassEncoderImpl {
    /// `None` once `wgpuComputePassEncoderEnd` has ended it.
    pub pass: RefCell<Option<wgpu::ComputePass<'static>>>,
    /// The encoder the pass records into: the timestamp resolve is recorded
    /// there after the pass ends.
    pub encoder: *mut WGPUCommandEncoderImpl,
    pub timing: RefCell<Option<TimedPass>>,
    pub dispatches: RefCell<u64>,
}

pub struct WGPUCommandBufferImpl {
    /// `None` once `wgpuQueueSubmit` has taken it.
    pub buffer: RefCell<Option<wgpu::CommandBuffer>>,
    pub timed: RefCell<Vec<TimedPass>>,
}

/// The object behind a handle, or a loud stop for a null one.
///
/// # Safety
/// `p` must be null or a pointer this crate made with `Box::into_raw` (every
/// handle is one, and none is ever freed — see the module header).
pub unsafe fn get<'a, T>(p: *mut T, what: &str) -> &'a T {
    match unsafe { p.as_ref() } {
        Some(r) => r,
        None => crate::fatal(&format!("a null {what} handle")),
    }
}

/// A new handle.
pub fn make<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}
