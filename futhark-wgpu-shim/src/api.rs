//! The 34 entry points of `surface/wgpu_functions.txt`, plus
//! `emscripten_sleep`.
//!
//! Each is a marshalling of C arguments onto the `wgpu-rs` call the plan's
//! §4.1 names. The Rust names are snake_case; the C names come from
//! `#[export_name]`, which is also what `cbindgen` writes into the header.
//!
//! # Safety
//!
//! Every function here is `unsafe extern "C"`: the caller (Futhark's
//! generated C) must pass handles this crate made, descriptors that are valid
//! for the duration of the call, and data pointers valid for the sizes given —
//! the WebGPU C API's own contract. A null handle is a loud stop.

use crate::futures::{fill, slot, Completion, Deliver, ErrorCallback, Registry, UserData};
use crate::handles::*;
use crate::report;
use crate::types::*;
use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::{c_void, CStr};
use std::num::NonZeroU64;
use std::ops::Bound;
use std::rc::Rc;

// ---- helpers ---------------------------------------------------------------

/// The text of a string view: `None` for the null view.
///
/// # Safety
/// `v.data` must be null, or valid for `v.length` bytes (or NUL-terminated if
/// the length is `WGPU_STRLEN`).
unsafe fn text<'a>(v: WGPUStringView) -> Option<&'a str> {
    if v.data.is_null() {
        return None;
    }
    let bytes: &[u8] = if v.length == STRLEN {
        unsafe { CStr::from_ptr(v.data) }.to_bytes()
    } else {
        unsafe { std::slice::from_raw_parts(v.data.cast(), v.length) }
    };
    match std::str::from_utf8(bytes) {
        Ok(s) => Some(s),
        Err(e) => crate::fatal(&format!("a string view that is not UTF-8: {e}")),
    }
}

/// A label: `None` for the null view.
unsafe fn label<'a>(v: WGPUStringView) -> Option<&'a str> {
    unsafe { text(v) }
}

/// A slice from a C pointer and a count (the empty slice for a null pointer
/// with count 0).
unsafe fn slice<'a, T>(p: *const T, n: usize, what: &str) -> &'a [T] {
    if n == 0 {
        return &[];
    }
    if p.is_null() {
        crate::fatal(&format!("{what}: a null pointer with a count of {n}"));
    }
    unsafe { std::slice::from_raw_parts(p, n) }
}

/// No chained struct this surface does not know.
fn no_chain(p: *const WGPUChainedStruct, what: &str) {
    if !p.is_null() {
        crate::fatal(&format!("{what}: a nextInChain the surface does not implement"));
    }
}

fn buffer_usages(u: WGPUBufferUsage) -> wgpu::BufferUsages {
    let known = [
        (BUFFER_USAGE_MAP_READ, wgpu::BufferUsages::MAP_READ),
        (BUFFER_USAGE_MAP_WRITE, wgpu::BufferUsages::MAP_WRITE),
        (BUFFER_USAGE_COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BUFFER_USAGE_COPY_DST, wgpu::BufferUsages::COPY_DST),
        (BUFFER_USAGE_UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BUFFER_USAGE_STORAGE, wgpu::BufferUsages::STORAGE),
    ];
    let mut out = wgpu::BufferUsages::empty();
    let mut rest = u;
    for (bit, usage) in known {
        if u & bit != 0 {
            out |= usage;
            rest &= !bit;
        }
    }
    if rest != 0 {
        crate::fatal(&format!("a buffer usage the surface does not implement: {rest:#x}"));
    }
    out
}

fn shader_stages(s: WGPUShaderStage) -> wgpu::ShaderStages {
    if s & !SHADER_STAGE_COMPUTE != 0 {
        crate::fatal(&format!("a shader stage the surface does not implement: {s:#x}"));
    }
    if s & SHADER_STAGE_COMPUTE != 0 {
        wgpu::ShaderStages::COMPUTE
    } else {
        wgpu::ShaderStages::NONE
    }
}

/// A byte range of a buffer. `WGPU_WHOLE_MAP_SIZE` (all-ones) means "to the
/// end of the buffer".
fn range(offset: u64, size: u64, whole: u64) -> (Bound<u64>, Bound<u64>) {
    if size == whole {
        (Bound::Included(offset), Bound::Unbounded)
    } else {
        (Bound::Included(offset), Bound::Excluded(offset + size))
    }
}

fn user(u1: *mut c_void, u2: *mut c_void) -> UserData {
    UserData { u1, u2 }
}

// ---- instance and adapter --------------------------------------------------

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuCreateInstance"]
pub unsafe extern "C" fn wgpu_create_instance(descriptor: *const WGPUInstanceDescriptor) -> WGPUInstance {
    if let Some(d) = unsafe { descriptor.as_ref() } {
        no_chain(d.next_in_chain, "wgpuCreateInstance");
    }
    report::at_exit();
    report::instance_created();
    make(WGPUInstanceImpl { instance: wgpu::Instance::default(), registry: Rc::new(Registry::default()) })
}

unsafe fn request_adapter(
    instance: WGPUInstance,
    options: *const WGPURequestAdapterOptions,
    info: WGPURequestAdapterCallbackInfo,
) -> WGPUFuture {
    let inst = unsafe { get(instance, "WGPUInstance") };
    no_chain(info.next_in_chain, "WGPURequestAdapterCallbackInfo");
    let mut opts = wgpu::RequestAdapterOptions::default();
    if let Some(o) = unsafe { options.as_ref() } {
        no_chain(o.next_in_chain, "WGPURequestAdapterOptions");
        opts.power_preference = match o.power_preference {
            0 => wgpu::PowerPreference::None,
            1 => wgpu::PowerPreference::LowPower,
            2 => wgpu::PowerPreference::HighPerformance,
            p => crate::fatal(&format!("an unknown WGPUPowerPreference {p}")),
        };
        opts.force_fallback_adapter = o.force_fallback_adapter != 0;
    }
    let s = slot();
    let r = pollster::block_on(inst.instance.request_adapter(&opts)).map_err(|e| e.to_string());
    fill(&s, Completion::Adapter(r));
    inst.registry.start(info.mode, s, None, Deliver::Adapter(info.callback, user(info.userdata1, info.userdata2)))
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuInstanceRequestAdapter"]
pub unsafe extern "C" fn wgpu_instance_request_adapter(
    instance: WGPUInstance,
    options: *const WGPURequestAdapterOptions,
    callback_info: WGPURequestAdapterCallbackInfo,
) -> WGPUFuture {
    unsafe { request_adapter(instance, options, callback_info) }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuInstanceRequestAdapterF"]
pub unsafe extern "C" fn wgpu_instance_request_adapter_f(
    instance: WGPUInstance,
    options: *const WGPURequestAdapterOptions,
    callback_info: WGPURequestAdapterCallbackInfo,
) -> WGPUFuture {
    unsafe { request_adapter(instance, options, callback_info) }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuInstanceWaitAny"]
pub unsafe extern "C" fn wgpu_instance_wait_any(
    instance: WGPUInstance,
    future_count: usize,
    futures: *mut WGPUFutureWaitInfo,
    timeout_ns: u64,
) -> WGPUWaitStatus {
    let inst = unsafe { get(instance, "WGPUInstance") };
    if future_count == 0 {
        return WAIT_STATUS_SUCCESS;
    }
    if futures.is_null() {
        crate::fatal("wgpuInstanceWaitAny: a null futures array");
    }
    let infos = unsafe { std::slice::from_raw_parts_mut(futures, future_count) };
    inst.registry.wait_any(infos, timeout_ns)
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuAdapterGetLimits"]
pub unsafe extern "C" fn wgpu_adapter_get_limits(adapter: WGPUAdapter, limits: *mut WGPULimits) -> WGPUStatus {
    let a = unsafe { get(adapter, "WGPUAdapter") };
    let Some(out) = (unsafe { limits.as_mut() }) else {
        return STATUS_ERROR;
    };
    let l = a.adapter.limits();
    out.max_bind_groups = l.max_bind_groups;
    out.max_bindings_per_bind_group = l.max_bindings_per_bind_group;
    out.max_dynamic_uniform_buffers_per_pipeline_layout = l.max_dynamic_uniform_buffers_per_pipeline_layout;
    out.max_dynamic_storage_buffers_per_pipeline_layout = l.max_dynamic_storage_buffers_per_pipeline_layout;
    out.max_storage_buffers_per_shader_stage = l.max_storage_buffers_per_shader_stage;
    out.max_uniform_buffers_per_shader_stage = l.max_uniform_buffers_per_shader_stage;
    out.max_uniform_buffer_binding_size = l.max_uniform_buffer_binding_size;
    out.max_storage_buffer_binding_size = l.max_storage_buffer_binding_size;
    out.max_buffer_size = l.max_buffer_size;
    out.max_compute_workgroup_storage_size = l.max_compute_workgroup_storage_size;
    out.max_compute_invocations_per_workgroup = l.max_compute_invocations_per_workgroup;
    out.max_compute_workgroup_size_x = l.max_compute_workgroup_size_x;
    out.max_compute_workgroup_size_y = l.max_compute_workgroup_size_y;
    out.max_compute_workgroup_size_z = l.max_compute_workgroup_size_z;
    out.max_compute_workgroups_per_dimension = l.max_compute_workgroups_per_dimension;
    STATUS_SUCCESS
}

/// The limits a device descriptor asks for: WebGPU's defaults, with every
/// field the caller defined (not all-ones) taken from the caller.
fn required_limits(l: &WGPULimits) -> wgpu::Limits {
    let mut out = wgpu::Limits::default();
    let u32s: [(u32, &mut u32); 12] = [
        (l.max_bind_groups, &mut out.max_bind_groups),
        (l.max_bindings_per_bind_group, &mut out.max_bindings_per_bind_group),
        (l.max_dynamic_uniform_buffers_per_pipeline_layout, &mut out.max_dynamic_uniform_buffers_per_pipeline_layout),
        (l.max_dynamic_storage_buffers_per_pipeline_layout, &mut out.max_dynamic_storage_buffers_per_pipeline_layout),
        (l.max_storage_buffers_per_shader_stage, &mut out.max_storage_buffers_per_shader_stage),
        (l.max_uniform_buffers_per_shader_stage, &mut out.max_uniform_buffers_per_shader_stage),
        (l.max_compute_workgroup_storage_size, &mut out.max_compute_workgroup_storage_size),
        (l.max_compute_invocations_per_workgroup, &mut out.max_compute_invocations_per_workgroup),
        (l.max_compute_workgroup_size_x, &mut out.max_compute_workgroup_size_x),
        (l.max_compute_workgroup_size_y, &mut out.max_compute_workgroup_size_y),
        (l.max_compute_workgroup_size_z, &mut out.max_compute_workgroup_size_z),
        (l.max_compute_workgroups_per_dimension, &mut out.max_compute_workgroups_per_dimension),
    ];
    for (v, slot) in u32s {
        if v != LIMIT_U32_UNDEFINED {
            *slot = v;
        }
    }
    let u64s: [(u64, &mut u64); 3] = [
        (l.max_uniform_buffer_binding_size, &mut out.max_uniform_buffer_binding_size),
        (l.max_storage_buffer_binding_size, &mut out.max_storage_buffer_binding_size),
        (l.max_buffer_size, &mut out.max_buffer_size),
    ];
    for (v, slot) in u64s {
        if v != LIMIT_U64_UNDEFINED {
            *slot = v;
        }
    }
    out
}

unsafe fn request_device(
    adapter: WGPUAdapter,
    descriptor: *const WGPUDeviceDescriptor,
    info: WGPURequestDeviceCallbackInfo,
) -> WGPUFuture {
    let a = unsafe { get(adapter, "WGPUAdapter") };
    no_chain(info.next_in_chain, "WGPURequestDeviceCallbackInfo");
    let mut features = wgpu::Features::empty();
    let mut limits = wgpu::Limits::default();
    let mut errors = ErrorCallback { callback: None, user: user(std::ptr::null_mut(), std::ptr::null_mut()) };
    let mut device_label = None;
    if let Some(d) = unsafe { descriptor.as_ref() } {
        no_chain(d.next_in_chain, "WGPUDeviceDescriptor");
        device_label = unsafe { label(d.label) };
        for f in unsafe { slice(d.required_features, d.required_feature_count, "requiredFeatures") } {
            match *f {
                FEATURE_NAME_SHADER_F16 => features |= wgpu::Features::SHADER_F16,
                other => crate::fatal(&format!("a feature the surface does not implement: WGPUFeatureName {other:#x}")),
            }
        }
        if let Some(l) = unsafe { d.required_limits.as_ref() } {
            limits = required_limits(l);
        }
        let e = &d.uncaptured_error_callback_info;
        no_chain(e.next_in_chain, "WGPUUncapturedErrorCallbackInfo");
        errors = ErrorCallback { callback: e.callback, user: user(e.userdata1, e.userdata2) };
    }
    let timestamps = report::timestamps_wanted();
    if timestamps {
        if !a.adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            crate::fatal("FUTHARK_WGPU_SHIM_TIMESTAMPS=1, but the adapter has no TIMESTAMP_QUERY");
        }
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    let desc = wgpu::DeviceDescriptor {
        label: device_label,
        required_features: features,
        required_limits: limits,
        ..Default::default()
    };
    let s = slot();
    let r = pollster::block_on(a.adapter.request_device(&desc)).map_err(|e| e.to_string());
    report::device_ready();
    fill(&s, Completion::Device(r));
    a.registry.start(
        info.mode,
        s,
        None,
        Deliver::Device(info.callback, user(info.userdata1, info.userdata2), errors, timestamps),
    )
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuAdapterRequestDevice"]
pub unsafe extern "C" fn wgpu_adapter_request_device(
    adapter: WGPUAdapter,
    descriptor: *const WGPUDeviceDescriptor,
    callback_info: WGPURequestDeviceCallbackInfo,
) -> WGPUFuture {
    unsafe { request_device(adapter, descriptor, callback_info) }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuAdapterRequestDeviceF"]
pub unsafe extern "C" fn wgpu_adapter_request_device_f(
    adapter: WGPUAdapter,
    descriptor: *const WGPUDeviceDescriptor,
    callback_info: WGPURequestDeviceCallbackInfo,
) -> WGPUFuture {
    unsafe { request_device(adapter, descriptor, callback_info) }
}

// ---- device ----------------------------------------------------------------

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceGetQueue"]
pub unsafe extern "C" fn wgpu_device_get_queue(device: WGPUDevice) -> WGPUQueue {
    let d = unsafe { get(device, "WGPUDevice") };
    let mut cached = d.queue_handle.borrow_mut();
    *cached.get_or_insert_with(|| {
        make(WGPUQueueImpl { queue: d.queue.clone(), device: d.device.clone(), registry: Rc::clone(&d.registry) })
    })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceDestroy"]
pub unsafe extern "C" fn wgpu_device_destroy(device: WGPUDevice) {
    unsafe { get(device, "WGPUDevice") }.device.destroy();
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreateBuffer"]
pub unsafe extern "C" fn wgpu_device_create_buffer(device: WGPUDevice, descriptor: *const WGPUBufferDescriptor) -> WGPUBuffer {
    let d = unsafe { get(device, "WGPUDevice") };
    let desc = unsafe { get(descriptor.cast_mut(), "WGPUBufferDescriptor") };
    no_chain(desc.next_in_chain, "WGPUBufferDescriptor");
    let buffer = d.device.create_buffer(&wgpu::BufferDescriptor {
        label: unsafe { label(desc.label) },
        size: desc.size,
        usage: buffer_usages(desc.usage),
        mapped_at_creation: desc.mapped_at_creation != 0,
    });
    make(WGPUBufferImpl {
        buffer,
        device: d.device.clone(),
        registry: Rc::clone(&d.registry),
        views: RefCell::new(Vec::new()),
    })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreateShaderModule"]
pub unsafe extern "C" fn wgpu_device_create_shader_module(
    device: WGPUDevice,
    descriptor: *const WGPUShaderModuleDescriptor,
) -> WGPUShaderModule {
    let d = unsafe { get(device, "WGPUDevice") };
    let desc = unsafe { get(descriptor.cast_mut(), "WGPUShaderModuleDescriptor") };
    let chain = unsafe { desc.next_in_chain.as_ref() }
        .unwrap_or_else(|| crate::fatal("wgpuDeviceCreateShaderModule: no shader source in nextInChain"));
    if chain.s_type != STYPE_SHADER_SOURCE_WGSL {
        crate::fatal(&format!("a shader source the surface does not implement: WGPUSType {:#x}", chain.s_type));
    }
    no_chain(chain.next, "WGPUShaderSourceWGSL");
    // SAFETY: a chained struct of sType ShaderSourceWGSL is the `chain` field
    // at offset 0 of a WGPUShaderSourceWGSL (both are #[repr(C)]).
    let wgsl = unsafe { &*(chain as *const WGPUChainedStruct).cast::<WGPUShaderSourceWGSL>() };
    let code = unsafe { text(wgsl.code) }.unwrap_or_else(|| crate::fatal("a WGSL shader source with no code"));
    let t = std::time::Instant::now();
    let module = d.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: unsafe { label(desc.label) },
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(code.to_string())),
    });
    report::count_pipeline(t.elapsed().as_nanos());
    make(WGPUShaderModuleImpl { module, registry: Rc::clone(&d.registry) })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuShaderModuleGetCompilationInfo"]
pub unsafe extern "C" fn wgpu_shader_module_get_compilation_info(
    module: WGPUShaderModule,
    callback_info: WGPUCompilationInfoCallbackInfo,
) -> WGPUFuture {
    let m = unsafe { get(module, "WGPUShaderModule") };
    no_chain(callback_info.next_in_chain, "WGPUCompilationInfoCallbackInfo");
    let s = slot();
    fill(&s, Completion::Compilation(pollster::block_on(m.module.get_compilation_info())));
    m.registry.start(
        callback_info.mode,
        s,
        None,
        Deliver::Compilation(callback_info.callback, user(callback_info.userdata1, callback_info.userdata2)),
    )
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreateBindGroupLayout"]
pub unsafe extern "C" fn wgpu_device_create_bind_group_layout(
    device: WGPUDevice,
    descriptor: *const WGPUBindGroupLayoutDescriptor,
) -> WGPUBindGroupLayout {
    let d = unsafe { get(device, "WGPUDevice") };
    let desc = unsafe { get(descriptor.cast_mut(), "WGPUBindGroupLayoutDescriptor") };
    no_chain(desc.next_in_chain, "WGPUBindGroupLayoutDescriptor");
    let entries: Vec<wgpu::BindGroupLayoutEntry> = unsafe { slice(desc.entries, desc.entry_count, "entries") }
        .iter()
        .map(|e| {
            no_chain(e.next_in_chain, "WGPUBindGroupLayoutEntry");
            no_chain(e.buffer.next_in_chain, "WGPUBufferBindingLayout");
            let ty = match e.buffer.kind {
                BUFFER_BINDING_TYPE_UNIFORM => wgpu::BufferBindingType::Uniform,
                BUFFER_BINDING_TYPE_STORAGE => wgpu::BufferBindingType::Storage { read_only: false },
                BUFFER_BINDING_TYPE_READ_ONLY_STORAGE => wgpu::BufferBindingType::Storage { read_only: true },
                BUFFER_BINDING_TYPE_BINDING_NOT_USED => {
                    crate::fatal("a bind group layout entry with no buffer: the surface implements buffer bindings only")
                }
                other => crate::fatal(&format!("an unknown WGPUBufferBindingType {other}")),
            };
            wgpu::BindGroupLayoutEntry {
                binding: e.binding,
                visibility: shader_stages(e.visibility),
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: e.buffer.has_dynamic_offset != 0,
                    min_binding_size: NonZeroU64::new(e.buffer.min_binding_size),
                },
                count: None,
            }
        })
        .collect();
    let layout =
        d.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: unsafe { label(desc.label) }, entries: &entries });
    make(WGPUBindGroupLayoutImpl { layout })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreatePipelineLayout"]
pub unsafe extern "C" fn wgpu_device_create_pipeline_layout(
    device: WGPUDevice,
    descriptor: *const WGPUPipelineLayoutDescriptor,
) -> WGPUPipelineLayout {
    let d = unsafe { get(device, "WGPUDevice") };
    let desc = unsafe { get(descriptor.cast_mut(), "WGPUPipelineLayoutDescriptor") };
    no_chain(desc.next_in_chain, "WGPUPipelineLayoutDescriptor");
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
        unsafe { slice(desc.bind_group_layouts, desc.bind_group_layout_count, "bindGroupLayouts") }
            .iter()
            .map(|h| Some(&unsafe { get(*h, "WGPUBindGroupLayout") }.layout))
            .collect();
    let layout = d.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: unsafe { label(desc.label) },
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    make(WGPUPipelineLayoutImpl { layout })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreateComputePipeline"]
pub unsafe extern "C" fn wgpu_device_create_compute_pipeline(
    device: WGPUDevice,
    descriptor: *const WGPUComputePipelineDescriptor,
) -> WGPUComputePipeline {
    let d = unsafe { get(device, "WGPUDevice") };
    let desc = unsafe { get(descriptor.cast_mut(), "WGPUComputePipelineDescriptor") };
    no_chain(desc.next_in_chain, "WGPUComputePipelineDescriptor");
    no_chain(desc.compute.next_in_chain, "WGPUComputeState");
    let layout = unsafe { desc.layout.as_ref() }.map(|l| &l.layout);
    let module = &unsafe { get(desc.compute.module, "WGPUShaderModule") }.module;
    let constants: Vec<(&str, f64)> = unsafe { slice(desc.compute.constants, desc.compute.constant_count, "constants") }
        .iter()
        .map(|c| {
            no_chain(c.next_in_chain, "WGPUConstantEntry");
            (unsafe { text(c.key) }.unwrap_or_else(|| crate::fatal("a pipeline constant with no key")), c.value)
        })
        .collect();
    let t = std::time::Instant::now();
    let pipeline = d.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: unsafe { label(desc.label) },
        layout,
        module,
        entry_point: unsafe { text(desc.compute.entry_point) },
        compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
        cache: None,
    });
    report::count_pipeline(t.elapsed().as_nanos());
    make(WGPUComputePipelineImpl { pipeline })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreateBindGroup"]
pub unsafe extern "C" fn wgpu_device_create_bind_group(device: WGPUDevice, descriptor: *const WGPUBindGroupDescriptor) -> WGPUBindGroup {
    let d = unsafe { get(device, "WGPUDevice") };
    let desc = unsafe { get(descriptor.cast_mut(), "WGPUBindGroupDescriptor") };
    no_chain(desc.next_in_chain, "WGPUBindGroupDescriptor");
    let layout = &unsafe { get(desc.layout, "WGPUBindGroupLayout") }.layout;
    let entries: Vec<wgpu::BindGroupEntry> = unsafe { slice(desc.entries, desc.entry_count, "entries") }
        .iter()
        .map(|e| {
            no_chain(e.next_in_chain, "WGPUBindGroupEntry");
            let buffer = &unsafe { get(e.buffer, "WGPUBuffer") }.buffer;
            let size = if e.size == u64::MAX {
                None
            } else {
                Some(NonZeroU64::new(e.size).unwrap_or_else(|| crate::fatal("a bind group entry of size 0")))
            };
            wgpu::BindGroupEntry {
                binding: e.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: e.offset, size }),
            }
        })
        .collect();
    let group = d.device.create_bind_group(&wgpu::BindGroupDescriptor { label: unsafe { label(desc.label) }, layout, entries: &entries });
    make(WGPUBindGroupImpl { group })
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuDeviceCreateCommandEncoder"]
pub unsafe extern "C" fn wgpu_device_create_command_encoder(
    device: WGPUDevice,
    descriptor: *const WGPUCommandEncoderDescriptor,
) -> WGPUCommandEncoder {
    let d = unsafe { get(device, "WGPUDevice") };
    let mut lbl = None;
    if let Some(desc) = unsafe { descriptor.as_ref() } {
        no_chain(desc.next_in_chain, "WGPUCommandEncoderDescriptor");
        lbl = unsafe { label(desc.label) };
    }
    let encoder = d.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: lbl });
    make(WGPUCommandEncoderImpl {
        encoder: RefCell::new(Some(encoder)),
        device: d.device.clone(),
        timestamps: d.timestamps,
        timed: RefCell::new(Vec::new()),
    })
}

// ---- command encoder and compute pass --------------------------------------

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuCommandEncoderBeginComputePass"]
pub unsafe extern "C" fn wgpu_command_encoder_begin_compute_pass(
    encoder: WGPUCommandEncoder,
    descriptor: *const WGPUComputePassDescriptor,
) -> WGPUComputePassEncoder {
    let e = unsafe { get(encoder, "WGPUCommandEncoder") };
    let mut lbl = None;
    if let Some(desc) = unsafe { descriptor.as_ref() } {
        no_chain(desc.next_in_chain, "WGPUComputePassDescriptor");
        lbl = unsafe { label(desc.label) };
    }
    let timing = e.timestamps.then(|| report::TimedPass {
        queries: e.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("futhark-wgpu-shim pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        }),
        resolve: e.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("futhark-wgpu-shim timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        read: e.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("futhark-wgpu-shim timestamp read"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        dispatches: 0,
    });
    let mut guard = e.encoder.borrow_mut();
    let enc = guard.as_mut().unwrap_or_else(|| crate::fatal("wgpuCommandEncoderBeginComputePass on a finished encoder"));
    let writes = timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites {
        query_set: &t.queries,
        beginning_of_pass_write_index: Some(0),
        end_of_pass_write_index: Some(1),
    });
    let pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: lbl, timestamp_writes: writes }).forget_lifetime();
    make(WGPUComputePassEncoderImpl {
        pass: RefCell::new(Some(pass)),
        encoder,
        timing: RefCell::new(timing),
        dispatches: RefCell::new(0),
    })
}

fn with_pass<R>(p: &WGPUComputePassEncoderImpl, what: &str, f: impl FnOnce(&mut wgpu::ComputePass<'static>) -> R) -> R {
    let mut guard = p.pass.borrow_mut();
    let pass = guard.as_mut().unwrap_or_else(|| crate::fatal(&format!("{what} on an ended compute pass")));
    f(pass)
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuComputePassEncoderSetPipeline"]
pub unsafe extern "C" fn wgpu_compute_pass_encoder_set_pipeline(pass: WGPUComputePassEncoder, pipeline: WGPUComputePipeline) {
    let p = unsafe { get(pass, "WGPUComputePassEncoder") };
    let pl = &unsafe { get(pipeline, "WGPUComputePipeline") }.pipeline;
    with_pass(p, "wgpuComputePassEncoderSetPipeline", |pass| pass.set_pipeline(pl));
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuComputePassEncoderSetBindGroup"]
pub unsafe extern "C" fn wgpu_compute_pass_encoder_set_bind_group(
    pass: WGPUComputePassEncoder,
    group_index: u32,
    group: WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let p = unsafe { get(pass, "WGPUComputePassEncoder") };
    let g = unsafe { group.as_ref() }.map(|g| &g.group);
    let offsets = unsafe { slice(dynamic_offsets, dynamic_offset_count, "dynamicOffsets") };
    with_pass(p, "wgpuComputePassEncoderSetBindGroup", |pass| pass.set_bind_group(group_index, g, offsets));
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuComputePassEncoderDispatchWorkgroups"]
pub unsafe extern "C" fn wgpu_compute_pass_encoder_dispatch_workgroups(
    pass: WGPUComputePassEncoder,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
) {
    let p = unsafe { get(pass, "WGPUComputePassEncoder") };
    with_pass(p, "wgpuComputePassEncoderDispatchWorkgroups", |pass| {
        pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z)
    });
    *p.dispatches.borrow_mut() += 1;
    report::count_dispatch(workgroup_count_x, workgroup_count_y, workgroup_count_z);
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuComputePassEncoderEnd"]
pub unsafe extern "C" fn wgpu_compute_pass_encoder_end(pass: WGPUComputePassEncoder) {
    let p = unsafe { get(pass, "WGPUComputePassEncoder") };
    let ended = p.pass.borrow_mut().take();
    if ended.is_none() {
        crate::fatal("wgpuComputePassEncoderEnd on an ended compute pass");
    }
    drop(ended);
    if let Some(mut t) = p.timing.borrow_mut().take() {
        t.dispatches = *p.dispatches.borrow();
        let e = unsafe { get(p.encoder, "WGPUCommandEncoder") };
        e.timed.borrow_mut().push(t);
    }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuCommandEncoderCopyBufferToBuffer"]
pub unsafe extern "C" fn wgpu_command_encoder_copy_buffer_to_buffer(
    encoder: WGPUCommandEncoder,
    source: WGPUBuffer,
    source_offset: u64,
    destination: WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    let e = unsafe { get(encoder, "WGPUCommandEncoder") };
    let src = &unsafe { get(source, "WGPUBuffer") }.buffer;
    let dst = &unsafe { get(destination, "WGPUBuffer") }.buffer;
    let mut guard = e.encoder.borrow_mut();
    let enc = guard.as_mut().unwrap_or_else(|| crate::fatal("wgpuCommandEncoderCopyBufferToBuffer on a finished encoder"));
    enc.copy_buffer_to_buffer(src, source_offset, dst, destination_offset, if size == u64::MAX { None } else { Some(size) });
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuCommandEncoderFinish"]
pub unsafe extern "C" fn wgpu_command_encoder_finish(
    encoder: WGPUCommandEncoder,
    descriptor: *const WGPUCommandBufferDescriptor,
) -> WGPUCommandBuffer {
    let e = unsafe { get(encoder, "WGPUCommandEncoder") };
    if let Some(desc) = unsafe { descriptor.as_ref() } {
        no_chain(desc.next_in_chain, "WGPUCommandBufferDescriptor");
    }
    let enc = e.encoder.borrow_mut().take().unwrap_or_else(|| crate::fatal("wgpuCommandEncoderFinish on a finished encoder"));
    make(WGPUCommandBufferImpl {
        buffer: RefCell::new(Some(enc.finish())),
        timed: RefCell::new(std::mem::take(&mut *e.timed.borrow_mut())),
    })
}

// ---- queue -----------------------------------------------------------------

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuQueueWriteBuffer"]
pub unsafe extern "C" fn wgpu_queue_write_buffer(queue: WGPUQueue, buffer: WGPUBuffer, buffer_offset: u64, data: *const c_void, size: usize) {
    let q = unsafe { get(queue, "WGPUQueue") };
    let b = &unsafe { get(buffer, "WGPUBuffer") }.buffer;
    let bytes = unsafe { slice(data.cast::<u8>(), size, "wgpuQueueWriteBuffer data") };
    q.queue.write_buffer(b, buffer_offset, bytes);
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuQueueSubmit"]
pub unsafe extern "C" fn wgpu_queue_submit(queue: WGPUQueue, command_count: usize, commands: *const WGPUCommandBuffer) {
    let q = unsafe { get(queue, "WGPUQueue") };
    let mut buffers = Vec::with_capacity(command_count);
    let mut timed = Vec::new();
    for h in unsafe { slice(commands, command_count, "commands") } {
        let cb = unsafe { get(*h, "WGPUCommandBuffer") };
        buffers.push(cb.buffer.borrow_mut().take().unwrap_or_else(|| crate::fatal("a command buffer submitted twice")));
        timed.append(&mut cb.timed.borrow_mut());
    }
    q.queue.submit(buffers);
    report::count_submit();
    if timed.is_empty() {
        return;
    }
    // Timing mode only: read every pass's timestamps back now. This waits for
    // the submission, so it changes the program's overlap — it is a
    // diagnostic mode, never the mode results are taken in.
    //
    // The counters are resolved in a submission of their own, after the
    // program's submission has finished: resolved in the same command buffer
    // as the pass, Metal on this machine returned zero for both timestamps
    // in 3 of 4 runs (measured 2026-10-27), and a zero pair is no evidence.
    if let Err(e) = q.device.poll(wgpu::PollType::wait_indefinitely()) {
        crate::fatal(&format!("the device's wait before the pass timestamps failed: {e}"));
    }
    let mut enc = q.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("futhark-wgpu-shim timestamps") });
    for t in &timed {
        enc.resolve_query_set(&t.queries, 0..2, &t.resolve, 0);
        enc.copy_buffer_to_buffer(&t.resolve, 0, &t.read, 0, Some(16));
    }
    q.queue.submit(Some(enc.finish()));
    let period = f64::from(q.queue.get_timestamp_period());
    for t in timed {
        let s = slot();
        let s2 = s.clone();
        t.read.slice(..).map_async(wgpu::MapMode::Read, move |r| fill(&s2, Completion::Map(r.map_err(|e| e.to_string()))));
        if let Err(e) = q.device.poll(wgpu::PollType::wait_indefinitely()) {
            crate::fatal(&format!("the device's wait for the pass timestamps failed: {e}"));
        }
        match s.lock().unwrap_or_else(|p| p.into_inner()).take() {
            Some(Completion::Map(Ok(()))) => {}
            Some(Completion::Map(Err(e))) => crate::fatal(&format!("mapping the pass timestamps failed: {e}")),
            _ => crate::fatal("the pass timestamps were not mapped after the device's wait"),
        }
        let ticks: [u64; 2] = {
            let view = t.read.get_mapped_range(..).unwrap_or_else(|e| crate::fatal(&format!("the pass timestamps' view: {e}")));
            [u64::from_le_bytes(view[0..8].try_into().unwrap_or([0; 8])), u64::from_le_bytes(view[8..16].try_into().unwrap_or([0; 8]))]
        };
        t.read.unmap();
        if ticks[1] < ticks[0] {
            crate::fatal(&format!("a pass ended before it began: timestamps {} then {}", ticks[0], ticks[1]));
        }
        report::record_pass(ticks[0], ticks[1], (ticks[1] - ticks[0]) as f64 * period, t.dispatches);
        t.read.destroy();
        t.resolve.destroy();
    }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuQueueOnSubmittedWorkDone"]
pub unsafe extern "C" fn wgpu_queue_on_submitted_work_done(queue: WGPUQueue, callback_info: WGPUQueueWorkDoneCallbackInfo) -> WGPUFuture {
    unsafe { on_submitted_work_done(queue, callback_info) }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuQueueOnSubmittedWorkDoneF"]
pub unsafe extern "C" fn wgpu_queue_on_submitted_work_done_f(
    queue: WGPUQueue,
    callback_info: WGPUQueueWorkDoneCallbackInfo,
) -> WGPUFuture {
    unsafe { on_submitted_work_done(queue, callback_info) }
}

unsafe fn on_submitted_work_done(queue: WGPUQueue, info: WGPUQueueWorkDoneCallbackInfo) -> WGPUFuture {
    let q = unsafe { get(queue, "WGPUQueue") };
    no_chain(info.next_in_chain, "WGPUQueueWorkDoneCallbackInfo");
    let s = slot();
    let s2 = s.clone();
    q.queue.on_submitted_work_done(move || fill(&s2, Completion::WorkDone));
    q.registry.start(info.mode, s, Some(q.device.clone()), Deliver::WorkDone(info.callback, user(info.userdata1, info.userdata2)))
}

// ---- buffer ----------------------------------------------------------------

unsafe fn map_async(buffer: WGPUBuffer, mode: WGPUMapMode, offset: usize, size: usize, info: WGPUBufferMapCallbackInfo) -> WGPUFuture {
    let b = unsafe { get(buffer, "WGPUBuffer") };
    no_chain(info.next_in_chain, "WGPUBufferMapCallbackInfo");
    let map_mode = match mode {
        MAP_MODE_READ => wgpu::MapMode::Read,
        MAP_MODE_WRITE => wgpu::MapMode::Write,
        other => crate::fatal(&format!("a map mode the surface does not implement: {other:#x}")),
    };
    let s = slot();
    let s2 = s.clone();
    b.buffer
        .slice(range(offset as u64, size as u64, usize::MAX as u64))
        .map_async(map_mode, move |r| fill(&s2, Completion::Map(r.map_err(|e| e.to_string()))));
    b.registry.start(info.mode, s, Some(b.device.clone()), Deliver::Map(info.callback, user(info.userdata1, info.userdata2)))
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuBufferMapAsync"]
pub unsafe extern "C" fn wgpu_buffer_map_async(
    buffer: WGPUBuffer,
    mode: WGPUMapMode,
    offset: usize,
    size: usize,
    callback_info: WGPUBufferMapCallbackInfo,
) -> WGPUFuture {
    unsafe { map_async(buffer, mode, offset, size, callback_info) }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuBufferMapAsyncF"]
pub unsafe extern "C" fn wgpu_buffer_map_async_f(
    buffer: WGPUBuffer,
    mode: WGPUMapMode,
    offset: usize,
    size: usize,
    callback_info: WGPUBufferMapCallbackInfo,
) -> WGPUFuture {
    unsafe { map_async(buffer, mode, offset, size, callback_info) }
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuBufferGetConstMappedRange"]
pub unsafe extern "C" fn wgpu_buffer_get_const_mapped_range(buffer: WGPUBuffer, offset: usize, size: usize) -> *const c_void {
    let b = unsafe { get(buffer, "WGPUBuffer") };
    let view = b
        .buffer
        .get_mapped_range(range(offset as u64, size as u64, usize::MAX as u64))
        .unwrap_or_else(|e| crate::fatal(&format!("wgpuBufferGetConstMappedRange: {e}")));
    let p = view.as_ptr().cast::<c_void>();
    b.views.borrow_mut().push(view);
    p
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuBufferUnmap"]
pub unsafe extern "C" fn wgpu_buffer_unmap(buffer: WGPUBuffer) {
    let b = unsafe { get(buffer, "WGPUBuffer") };
    b.views.borrow_mut().clear();
    b.buffer.unmap();
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuBufferGetSize"]
pub unsafe extern "C" fn wgpu_buffer_get_size(buffer: WGPUBuffer) -> u64 {
    unsafe { get(buffer, "WGPUBuffer") }.buffer.size()
}

/// # Safety
/// The WebGPU C API's contract (the module header).
#[export_name = "wgpuBufferDestroy"]
pub unsafe extern "C" fn wgpu_buffer_destroy(buffer: WGPUBuffer) {
    let b = unsafe { get(buffer, "WGPUBuffer") };
    b.views.borrow_mut().clear();
    b.buffer.destroy();
}

// ---- emscripten.h ----------------------------------------------------------

/// The native stand-in for `emscripten_sleep` (`include/emscripten.h`). It
/// yields, and it is counted: the default wait path should never call it
/// (the plan's §4.5, rule 4).
#[export_name = "emscripten_sleep"]
pub extern "C" fn emscripten_sleep(ms: u32) {
    report::count_sleep();
    if ms == 0 {
        std::thread::yield_now();
    } else {
        std::thread::sleep(std::time::Duration::from_millis(u64::from(ms)));
    }
}
