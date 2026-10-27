//! The C types of the surface: every `WGPU*` name in `surface/wgpu_types.txt`,
//! plus the few the C needs to spell them (the chained struct, the callback
//! typedefs, the descriptors the runtime passes as `NULL`).
//!
//! `cbindgen` writes `webgpu/webgpu.h` from these definitions, so the C side
//! and the Rust side share one layout. The Rust names follow Rust's case
//! rules and `cbindgen.toml` gives the C names: fields and arguments are
//! renamed to camelCase, and the constants (C's `WGPUBufferUsage_MapRead`,
//! Rust's `BUFFER_USAGE_MAP_READ`) through the rename table.
//!
//! **Every enum is a `u32` (or `u64`) typedef with constants, never a Rust
//! `enum`.** A value that comes in from C may be anything, and a Rust enum
//! holding a value outside its variants is undefined behaviour; a `u32` is
//! checked against the constants instead, and an unknown one is a loud stop.
//!
//! The values are those of the WebGPU C header (webgpu-headers), so the
//! numbers Futhark's runtime prints in its panics mean what they say there.
//! Only what the surface uses is defined: this is a bridge, not a permanent
//! API (the plan's §10 question 4, decided: keep it small).

use crate::handles::{
    WGPUAdapterImpl, WGPUBindGroupImpl, WGPUBindGroupLayoutImpl, WGPUBufferImpl, WGPUCommandBufferImpl,
    WGPUCommandEncoderImpl, WGPUComputePassEncoderImpl, WGPUComputePipelineImpl, WGPUDeviceImpl, WGPUInstanceImpl,
    WGPUPipelineLayoutImpl, WGPUQueueImpl, WGPUShaderModuleImpl,
};
use std::ffi::{c_char, c_void};

// ---- handles ---------------------------------------------------------------

pub type WGPUInstance = *mut WGPUInstanceImpl;
pub type WGPUAdapter = *mut WGPUAdapterImpl;
pub type WGPUDevice = *mut WGPUDeviceImpl;
pub type WGPUQueue = *mut WGPUQueueImpl;
pub type WGPUBuffer = *mut WGPUBufferImpl;
pub type WGPUShaderModule = *mut WGPUShaderModuleImpl;
pub type WGPUBindGroupLayout = *mut WGPUBindGroupLayoutImpl;
pub type WGPUPipelineLayout = *mut WGPUPipelineLayoutImpl;
pub type WGPUComputePipeline = *mut WGPUComputePipelineImpl;
pub type WGPUBindGroup = *mut WGPUBindGroupImpl;
pub type WGPUCommandEncoder = *mut WGPUCommandEncoderImpl;
pub type WGPUComputePassEncoder = *mut WGPUComputePassEncoderImpl;
pub type WGPUCommandBuffer = *mut WGPUCommandBufferImpl;

// ---- scalars and flags -----------------------------------------------------

pub type WGPUBool = u32;
pub type WGPUFlags = u64;
pub type WGPUBufferUsage = WGPUFlags;
pub type WGPUMapMode = WGPUFlags;
pub type WGPUShaderStage = WGPUFlags;

pub const BUFFER_USAGE_MAP_READ: WGPUBufferUsage = 0x1;
pub const BUFFER_USAGE_MAP_WRITE: WGPUBufferUsage = 0x2;
pub const BUFFER_USAGE_COPY_SRC: WGPUBufferUsage = 0x4;
pub const BUFFER_USAGE_COPY_DST: WGPUBufferUsage = 0x8;
pub const BUFFER_USAGE_UNIFORM: WGPUBufferUsage = 0x40;
pub const BUFFER_USAGE_STORAGE: WGPUBufferUsage = 0x80;

pub const MAP_MODE_READ: WGPUMapMode = 0x1;
pub const MAP_MODE_WRITE: WGPUMapMode = 0x2;

pub const SHADER_STAGE_COMPUTE: WGPUShaderStage = 0x4;

// ---- enums (u32 typedefs) --------------------------------------------------

pub type WGPUSType = u32;
pub const STYPE_SHADER_SOURCE_WGSL: WGPUSType = 0x2;

pub type WGPUCallbackMode = u32;
pub const CALLBACK_MODE_WAIT_ANY_ONLY: WGPUCallbackMode = 0x1;
pub const CALLBACK_MODE_ALLOW_PROCESS_EVENTS: WGPUCallbackMode = 0x2;
pub const CALLBACK_MODE_ALLOW_SPONTANEOUS: WGPUCallbackMode = 0x3;

pub type WGPUStatus = u32;
pub const STATUS_SUCCESS: WGPUStatus = 0x1;
pub const STATUS_ERROR: WGPUStatus = 0x2;

pub type WGPUWaitStatus = u32;
pub const WAIT_STATUS_SUCCESS: WGPUWaitStatus = 0x1;
pub const WAIT_STATUS_TIMED_OUT: WGPUWaitStatus = 0x2;
pub const WAIT_STATUS_ERROR: WGPUWaitStatus = 0x3;

pub type WGPUMapAsyncStatus = u32;
pub const MAP_ASYNC_STATUS_SUCCESS: WGPUMapAsyncStatus = 0x1;
pub const MAP_ASYNC_STATUS_ERROR: WGPUMapAsyncStatus = 0x3;

pub type WGPURequestAdapterStatus = u32;
pub const REQUEST_ADAPTER_STATUS_SUCCESS: WGPURequestAdapterStatus = 0x1;
pub const REQUEST_ADAPTER_STATUS_UNAVAILABLE: WGPURequestAdapterStatus = 0x3;

pub type WGPURequestDeviceStatus = u32;
pub const REQUEST_DEVICE_STATUS_SUCCESS: WGPURequestDeviceStatus = 0x1;
pub const REQUEST_DEVICE_STATUS_ERROR: WGPURequestDeviceStatus = 0x3;

pub type WGPUQueueWorkDoneStatus = u32;
pub const QUEUE_WORK_DONE_STATUS_SUCCESS: WGPUQueueWorkDoneStatus = 0x1;

pub type WGPUCompilationInfoRequestStatus = u32;
pub const COMPILATION_INFO_REQUEST_STATUS_SUCCESS: WGPUCompilationInfoRequestStatus = 0x1;

pub type WGPUCompilationMessageType = u32;
pub const COMPILATION_MESSAGE_TYPE_ERROR: WGPUCompilationMessageType = 0x1;
pub const COMPILATION_MESSAGE_TYPE_WARNING: WGPUCompilationMessageType = 0x2;
pub const COMPILATION_MESSAGE_TYPE_INFO: WGPUCompilationMessageType = 0x3;

pub type WGPUErrorType = u32;
pub const ERROR_TYPE_VALIDATION: WGPUErrorType = 0x2;
pub const ERROR_TYPE_OUT_OF_MEMORY: WGPUErrorType = 0x3;
pub const ERROR_TYPE_INTERNAL: WGPUErrorType = 0x4;

pub type WGPUBufferBindingType = u32;
pub const BUFFER_BINDING_TYPE_BINDING_NOT_USED: WGPUBufferBindingType = 0x0;
pub const BUFFER_BINDING_TYPE_UNIFORM: WGPUBufferBindingType = 0x2;
pub const BUFFER_BINDING_TYPE_STORAGE: WGPUBufferBindingType = 0x3;
pub const BUFFER_BINDING_TYPE_READ_ONLY_STORAGE: WGPUBufferBindingType = 0x4;

pub type WGPUFeatureName = u32;
pub const FEATURE_NAME_SHADER_F16: WGPUFeatureName = 0xB;

/// `WGPU_STRLEN`: a string view whose length is this is NUL-terminated.
pub const STRLEN: usize = usize::MAX;
/// `WGPU_LIMIT_U32_UNDEFINED`.
pub const LIMIT_U32_UNDEFINED: u32 = u32::MAX;
/// `WGPU_LIMIT_U64_UNDEFINED`.
pub const LIMIT_U64_UNDEFINED: u64 = u64::MAX;

// ---- plain structs ---------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WGPUStringView {
    pub data: *const c_char,
    pub length: usize,
}

#[repr(C)]
pub struct WGPUChainedStruct {
    pub next: *const WGPUChainedStruct,
    pub s_type: WGPUSType,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WGPUFuture {
    pub id: u64,
}

#[repr(C)]
pub struct WGPUFutureWaitInfo {
    pub future: WGPUFuture,
    pub completed: WGPUBool,
}

#[repr(C)]
pub struct WGPUInstanceDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
}

#[repr(C)]
pub struct WGPURequestAdapterOptions {
    pub next_in_chain: *const WGPUChainedStruct,
    pub power_preference: u32,
    pub force_fallback_adapter: WGPUBool,
}

/// The limits the runtime reads and requests. Futhark's runtime fills the
/// struct with `0xff` bytes and then sets these fields, so an all-ones field
/// means "undefined" ([`LIMIT_U32_UNDEFINED`], [`LIMIT_U64_UNDEFINED`]).
#[repr(C)]
pub struct WGPULimits {
    pub next_in_chain: *mut WGPUChainedStruct,
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

#[repr(C)]
pub struct WGPUUncapturedErrorCallbackInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub callback: WGPUUncapturedErrorCallback,
    pub userdata1: *mut c_void,
    pub userdata2: *mut c_void,
}

#[repr(C)]
pub struct WGPUDeviceDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
    pub required_feature_count: usize,
    pub required_features: *const WGPUFeatureName,
    pub required_limits: *const WGPULimits,
    pub uncaptured_error_callback_info: WGPUUncapturedErrorCallbackInfo,
}

#[repr(C)]
pub struct WGPUBufferDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
    pub usage: WGPUBufferUsage,
    pub size: u64,
    pub mapped_at_creation: WGPUBool,
}

#[repr(C)]
pub struct WGPUShaderSourceWGSL {
    pub chain: WGPUChainedStruct,
    pub code: WGPUStringView,
}

#[repr(C)]
pub struct WGPUShaderModuleDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
}

/// cbindgen:field-names=[nextInChain, message, type, lineNum, linePos, offset, length]
#[repr(C)]
pub struct WGPUCompilationMessage {
    pub next_in_chain: *const WGPUChainedStruct,
    pub message: WGPUStringView,
    pub kind: WGPUCompilationMessageType,
    pub line_num: u64,
    pub line_pos: u64,
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
pub struct WGPUCompilationInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub message_count: usize,
    pub messages: *const WGPUCompilationMessage,
}

/// cbindgen:field-names=[nextInChain, type, hasDynamicOffset, minBindingSize]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WGPUBufferBindingLayout {
    pub next_in_chain: *const WGPUChainedStruct,
    pub kind: WGPUBufferBindingType,
    pub has_dynamic_offset: WGPUBool,
    pub min_binding_size: u64,
}

#[repr(C)]
pub struct WGPUBindGroupLayoutEntry {
    pub next_in_chain: *const WGPUChainedStruct,
    pub binding: u32,
    pub visibility: WGPUShaderStage,
    pub buffer: WGPUBufferBindingLayout,
}

#[repr(C)]
pub struct WGPUBindGroupLayoutDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
    pub entry_count: usize,
    pub entries: *const WGPUBindGroupLayoutEntry,
}

#[repr(C)]
pub struct WGPUPipelineLayoutDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
    pub bind_group_layout_count: usize,
    pub bind_group_layouts: *const WGPUBindGroupLayout,
}

#[repr(C)]
pub struct WGPUConstantEntry {
    pub next_in_chain: *const WGPUChainedStruct,
    pub key: WGPUStringView,
    pub value: f64,
}

#[repr(C)]
pub struct WGPUComputeState {
    pub next_in_chain: *const WGPUChainedStruct,
    pub module: WGPUShaderModule,
    pub entry_point: WGPUStringView,
    pub constant_count: usize,
    pub constants: *const WGPUConstantEntry,
}

#[repr(C)]
pub struct WGPUComputePipelineDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
    pub layout: WGPUPipelineLayout,
    pub compute: WGPUComputeState,
}

#[repr(C)]
pub struct WGPUBindGroupEntry {
    pub next_in_chain: *const WGPUChainedStruct,
    pub binding: u32,
    pub buffer: WGPUBuffer,
    pub offset: u64,
    pub size: u64,
}

#[repr(C)]
pub struct WGPUBindGroupDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
    pub layout: WGPUBindGroupLayout,
    pub entry_count: usize,
    pub entries: *const WGPUBindGroupEntry,
}

#[repr(C)]
pub struct WGPUCommandEncoderDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
}

#[repr(C)]
pub struct WGPUComputePassDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
}

#[repr(C)]
pub struct WGPUCommandBufferDescriptor {
    pub next_in_chain: *const WGPUChainedStruct,
    pub label: WGPUStringView,
}

// ---- callbacks -------------------------------------------------------------

pub type WGPUBufferMapCallback =
    Option<unsafe extern "C" fn(status: WGPUMapAsyncStatus, message: WGPUStringView, userdata1: *mut c_void, userdata2: *mut c_void)>;
pub type WGPUQueueWorkDoneCallback = Option<
    unsafe extern "C" fn(status: WGPUQueueWorkDoneStatus, message: WGPUStringView, userdata1: *mut c_void, userdata2: *mut c_void),
>;
pub type WGPURequestAdapterCallback = Option<
    unsafe extern "C" fn(
        status: WGPURequestAdapterStatus,
        adapter: WGPUAdapter,
        message: WGPUStringView,
        userdata1: *mut c_void,
        userdata2: *mut c_void,
    ),
>;
pub type WGPURequestDeviceCallback = Option<
    unsafe extern "C" fn(
        status: WGPURequestDeviceStatus,
        device: WGPUDevice,
        message: WGPUStringView,
        userdata1: *mut c_void,
        userdata2: *mut c_void,
    ),
>;
pub type WGPUCompilationInfoCallback = Option<
    unsafe extern "C" fn(
        status: WGPUCompilationInfoRequestStatus,
        compilation_info: *const WGPUCompilationInfo,
        userdata1: *mut c_void,
        userdata2: *mut c_void,
    ),
>;
pub type WGPUUncapturedErrorCallback = Option<
    unsafe extern "C" fn(
        device: *const WGPUDevice,
        kind: WGPUErrorType,
        message: WGPUStringView,
        userdata1: *mut c_void,
        userdata2: *mut c_void,
    ),
>;

#[repr(C)]
pub struct WGPUBufferMapCallbackInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub mode: WGPUCallbackMode,
    pub callback: WGPUBufferMapCallback,
    pub userdata1: *mut c_void,
    pub userdata2: *mut c_void,
}

#[repr(C)]
pub struct WGPUQueueWorkDoneCallbackInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub mode: WGPUCallbackMode,
    pub callback: WGPUQueueWorkDoneCallback,
    pub userdata1: *mut c_void,
    pub userdata2: *mut c_void,
}

#[repr(C)]
pub struct WGPURequestAdapterCallbackInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub mode: WGPUCallbackMode,
    pub callback: WGPURequestAdapterCallback,
    pub userdata1: *mut c_void,
    pub userdata2: *mut c_void,
}

#[repr(C)]
pub struct WGPURequestDeviceCallbackInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub mode: WGPUCallbackMode,
    pub callback: WGPURequestDeviceCallback,
    pub userdata1: *mut c_void,
    pub userdata2: *mut c_void,
}

#[repr(C)]
pub struct WGPUCompilationInfoCallbackInfo {
    pub next_in_chain: *const WGPUChainedStruct,
    pub mode: WGPUCallbackMode,
    pub callback: WGPUCompilationInfoCallback,
    pub userdata1: *mut c_void,
    pub userdata2: *mut c_void,
}
