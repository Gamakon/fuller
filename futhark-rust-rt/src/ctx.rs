//! THE CONTEXT: a caller's device and queue, and what the runtime keeps on them: the free list, the shader modules,
//! the bind group layouts and pipelines (cached), each generated module's state, the GPU pass timings.

use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::mem::{Block, FreeList};
use crate::Error;

/// The sizes the C runtime fixes for WebGPU (`backend_context_setup`, `rts/c/backends/webgpu.h`): the spec's default
/// limits, which are the actual ones unless a larger one is requested, and "a large enough value" where WebGPU has
/// no query. `lockstep_width` is never set by that runtime (0).
const MAX_THREAD_BLOCK_SIZE: i64 = 256;
const MAX_GRID_SIZE: i64 = 65536;
const MAX_TILE_SIZE: i64 = 65536;
const MAX_THRESHOLD: i64 = 65536;
const MAX_SHARED_MEMORY: i64 = 16384;
const MAX_REGISTERS: i64 = 65536;
const MAX_CACHE: i64 = 2097152;

/// What the runtime has done on a context: for the cost tables and the tests.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CtxStats {
    pub shader_modules: u64,
    pub pipelines_created: u64,
    pub pipelines_cached: u64,
    pub pipeline_ns: u128,
    pub launches: u64,
    pub skipped_launches: u64,
    pub submits: u64,
    pub buffers_created: u64,
    pub buffers_reused: u64,
}

/// A kernel's bindings: the scalars' binding and size, the storage bindings.
type LayoutKey = (u32, u64, Vec<u32>);

pub(crate) type PipelineKey = (u64, &'static str, Vec<(&'static str, u64)>);

/// A shader module and the identity the pipeline cache knows it by.
pub(crate) struct Module {
    pub id: u64,
    pub module: wgpu::ShaderModule,
}

/// A bind group layout and the pipeline layout over it.
pub(crate) struct Layouts {
    pub bind_group: wgpu::BindGroupLayout,
    pub pipeline: wgpu::PipelineLayout,
}

/// The context of one or more generated modules on one device. Not `Send`: one thread uses it.
pub struct FutharkCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pool: Rc<RefCell<FreeList>>,
    modules: RefCell<HashMap<String, Rc<Module>>>,
    layouts: RefCell<HashMap<LayoutKey, Rc<Layouts>>>,
    pipelines: RefCell<HashMap<PipelineKey, wgpu::ComputePipeline>>,
    states: RefCell<HashMap<TypeId, Rc<dyn Any>>>,
    readback: wgpu::Buffer,
    timing: Cell<bool>,
    passes: RefCell<Vec<f64>>,
    wait_timeout: Cell<Option<Duration>>,
    stats: Cell<CtxStats>,
    trace: Cell<bool>,
}

impl FutharkCtx {
    /// A context on the caller's device and queue. The device must have `SHADER_F16` (Futhark's WGSL prelude enables
    /// `f16`, as Futhark's runtime requires it of the device it makes); `TIMESTAMP_QUERY` is needed only for
    /// [`FutharkCtx::set_timing`].
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<FutharkCtx, Error> {
        if !device.features().contains(wgpu::Features::SHADER_F16) {
            return Err(Error::new("the device has no SHADER_F16, which Futhark's WGSL prelude needs (`enable f16;`)"));
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("futhark scalar readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(FutharkCtx {
            device,
            queue,
            pool: Rc::new(RefCell::new(FreeList::default())),
            modules: RefCell::new(HashMap::new()),
            layouts: RefCell::new(HashMap::new()),
            pipelines: RefCell::new(HashMap::new()),
            states: RefCell::new(HashMap::new()),
            readback,
            timing: Cell::new(false),
            passes: RefCell::new(Vec::new()),
            wait_timeout: Cell::new(None),
            stats: Cell::new(CtxStats::default()),
            trace: Cell::new(false),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Time every compute pass with GPU timestamps, as the shim's timing mode does (`futhark-wgpu-shim`,
    /// `FUTHARK_WGPU_SHIM_TIMESTAMPS=1`): each pass's begin and end are written by the GPU, and resolved in a
    /// submission of their own after the pass's has finished. It waits for every launch, so it changes the program's
    /// overlap: a measuring mode.
    pub fn set_timing(&self, on: bool) -> Result<(), Error> {
        if on && !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(Error::new("GPU timing needs a device with TIMESTAMP_QUERY"));
        }
        self.timing.set(on);
        Ok(())
    }
    pub fn timing(&self) -> bool {
        self.timing.get()
    }

    /// Every timed pass so far, in nanoseconds, in order.
    pub fn timed_passes(&self) -> usize {
        self.passes.borrow().len()
    }
    pub fn pass_ns(&self, i: usize) -> Option<f64> {
        self.passes.borrow().get(i).copied()
    }
    pub(crate) fn record_pass(&self, ns: f64) {
        self.passes.borrow_mut().push(ns);
    }

    /// Bound every wait for the device (a sync, a read-back): a wait past it is an error, and the context should not
    /// be used again. `None` (the default) waits as long as the device takes.
    pub fn set_wait_timeout(&self, t: Option<Duration>) {
        self.wait_timeout.set(t);
    }

    /// Print the program's trace messages (`TracePrint`) on standard error.
    pub fn set_trace(&self, on: bool) {
        self.trace.set(on);
    }
    pub fn trace(&self, msg: &str) {
        if self.trace.get() {
            eprint!("{msg}");
        }
    }

    pub fn stats(&self) -> CtxStats {
        self.stats.get()
    }
    pub(crate) fn count(&self, f: impl FnOnce(&mut CtxStats)) {
        let mut s = self.stats.get();
        f(&mut s);
        self.stats.set(s);
    }

    /// `GetSizeMax` and `SizeMaxConst`: the maximum of a size class, by its printed name.
    pub fn size_max(&self, class: &str) -> i64 {
        let prefix = |p: &str| class.starts_with(p);
        if prefix("thread_block_size") {
            MAX_THREAD_BLOCK_SIZE
        } else if prefix("grid_size") {
            MAX_GRID_SIZE
        } else if prefix("tile_size") {
            MAX_TILE_SIZE
        } else if prefix("threshold") {
            MAX_THRESHOLD
        } else if prefix("shared_memory") {
            MAX_SHARED_MEMORY
        } else if prefix("registers") {
            MAX_REGISTERS
        } else if prefix("cache") {
            MAX_CACHE
        } else {
            0
        }
    }

    /// `gpu_alloc`: at least 4 bytes, rounded up to a multiple of 4, from the free list when one is acceptable.
    pub(crate) fn alloc_block(&self, size: i64, tag: &'static str) -> Result<Block, Error> {
        if size < 0 {
            return Err(Error::new(format!("an allocation of {size} bytes")));
        }
        let min = ((size.max(4) as u64) + 3) & !3;
        if let Some((buffer, got)) = self.pool.borrow_mut().find(min, tag) {
            self.count(|s| s.buffers_reused += 1);
            return Ok(Block::new(buffer, got, tag, Some(self.pool.clone())));
        }
        let buffer = self.new_storage(min)?;
        Ok(Block::new(buffer, min, tag, Some(self.pool.clone())))
    }

    /// `gpu_alloc_actual`: a storage buffer the runtime owns.
    pub(crate) fn new_storage(&self, size: u64) -> Result<wgpu::Buffer, Error> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("futhark"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(Error::new(format!("Failed to allocate {size} bytes of GPU memory: {e}")));
        }
        self.count(|s| s.buffers_created += 1);
        Ok(buffer)
    }

    /// Every buffer of the free list released (`futhark_context_clear_caches`).
    pub fn clear_caches(&self) {
        self.pool.borrow_mut().clear();
    }

    /// The shader module of a WGSL text (ported for naga first), made once per text.
    pub(crate) fn module(&self, wgsl: &str, label: &str) -> Result<Rc<Module>, Error> {
        if let Some(m) = self.modules.borrow().get(wgsl) {
            return Ok(m.clone());
        }
        let ported = crate::port::port_module(wgsl)?;
        let t = Instant::now();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(ported.text.into()),
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(Error::new(format!("the WGSL of {label} does not validate: {e}")));
        }
        let id = self.modules.borrow().len() as u64;
        let m = Rc::new(Module { id, module });
        self.modules.borrow_mut().insert(wgsl.to_string(), m.clone());
        let ns = t.elapsed().as_nanos();
        self.count(|s| {
            s.shader_modules += 1;
            s.pipeline_ns += ns;
        });
        Ok(m)
    }

    /// The program's WGSL module made (and validated) now, as the C runtime makes it when its context is made.
    pub fn module_check(&self, wgsl: &str) -> Result<(), Error> {
        self.module(wgsl, "Futhark program").map(|_| ())
    }

    /// The layouts of a kernel's bindings: one uniform buffer of scalars, then storage buffers (`gpu_create_kernel`).
    pub(crate) fn layouts(&self, scalars_binding: u32, scalars_size: u64, bindings: &[u32]) -> Rc<Layouts> {
        let key = (scalars_binding, scalars_size, bindings.to_vec());
        if let Some(l) = self.layouts.borrow().get(&key) {
            return l.clone();
        }
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: scalars_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        for &b in bindings {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let bind_group = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("futhark kernel"), entries: &entries });
        let pipeline = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("futhark kernel"),
            bind_group_layouts: &[Some(&bind_group)],
            immediate_size: 0,
        });
        let l = Rc::new(Layouts { bind_group, pipeline });
        self.layouts.borrow_mut().insert(key, l.clone());
        l
    }

    /// The pipeline of a kernel entry point with these override constants, made once per module, entry point and
    /// constants.
    pub(crate) fn pipeline(
        &self,
        module: &Module,
        layouts: &Layouts,
        entry: &'static str,
        constants: &[(&'static str, f64)],
    ) -> Result<wgpu::ComputePipeline, Error> {
        let key: PipelineKey = (module.id, entry, constants.iter().map(|(k, v)| (*k, v.to_bits())).collect());
        if let Some(p) = self.pipelines.borrow().get(&key) {
            self.count(|s| s.pipelines_cached += 1);
            return Ok(p.clone());
        }
        let t = Instant::now();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layouts.pipeline),
            module: &module.module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(Error::new(format!("the pipeline of kernel {entry} could not be made: {e}")));
        }
        let ns = t.elapsed().as_nanos();
        self.count(|s| {
            s.pipelines_created += 1;
            s.pipeline_ns += ns;
        });
        self.pipelines.borrow_mut().insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// `futhark_context_sync`: wait until the device has done everything submitted.
    pub fn sync(&self) -> Result<(), Error> {
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: self.wait_timeout.get() })
            .map(|_| ())
            .map_err(|e| Error::device(format!("Failed to wait for work to be done: {e}")))
    }

    /// Map a buffer for reading and wait for it.
    pub(crate) fn map_read(&self, buffer: &wgpu::Buffer, size: u64) -> Result<(), Error> {
        // wgpu calls the closure inside `poll`, on this thread; `map_async` asks for a `Send` closure, so the slot is an
        // `Arc<Mutex<_>>`.
        let slot: Arc<Mutex<Option<Result<(), String>>>> = Arc::new(Mutex::new(None));
        let s = slot.clone();
        buffer.slice(0..size).map_async(wgpu::MapMode::Read, move |r| {
            *s.lock().unwrap_or_else(|p| p.into_inner()) = Some(r.map_err(|e| e.to_string()));
        });
        self.sync()?;
        let done = slot.lock().unwrap_or_else(|p| p.into_inner()).take();
        match done {
            Some(Ok(())) => Ok(()),
            Some(Err(e)) => Err(Error::new(format!("mapping a buffer failed: {e}"))),
            None => Err(Error::new("a buffer was not mapped after the device's wait")),
        }
    }

    pub(crate) fn readback(&self) -> &wgpu::Buffer {
        &self.readback
    }

    /// A generated module's state in this context, made by `init` on first use. The key is the state's type, which
    /// is a distinct type in every generated module.
    pub fn module_state<S: 'static, F>(&self, init: F) -> Result<Rc<S>, Error>
    where
        F: FnOnce(&FutharkCtx) -> Result<S, Error>,
    {
        let key = TypeId::of::<S>();
        if let Some(s) = self.states.borrow().get(&key) {
            return s.clone().downcast::<S>().map_err(|_| Error::new("our software has a bug: a module state of the wrong type"));
        }
        let s = Rc::new(init(self)?);
        self.states.borrow_mut().insert(key, s.clone());
        Ok(s)
    }

    /// Every module's state dropped (their constants' memory returns to the free list). A host that unloads a
    /// generated module must call this first: the states' code lives in the module.
    pub fn clear_module_states(&self) {
        self.states.borrow_mut().clear();
    }
}
