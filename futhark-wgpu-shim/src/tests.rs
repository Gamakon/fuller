//! The shim's own tests: the header contract, and the future/callback model
//! of the plan's §4.5 driven through the C entry points.
//!
//! The device tests touch the GPU; run them under the GPU lock
//! (`/private/tmp/claude-501/locks/gpu`), one GPU process at a time.

use crate::api::*;
use crate::types::*;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

const GENERATED: &str = include_str!(concat!(env!("OUT_DIR"), "/include/webgpu/webgpu.h"));
const COMMITTED: &str = include_str!("../include/webgpu/webgpu.h");
const SURFACE: &str = include_str!("../surface/wgpu_functions.txt");

/// Acceptance criterion 4: the header is generated, and the committed copy
/// the C compiles against is exactly what this build generated.
#[test]
fn the_committed_header_is_the_generated_one() {
    assert!(
        GENERATED == COMMITTED,
        "include/webgpu/webgpu.h is not the header this build generated; after a deliberate change to the \
         surface, refresh it with FUTHARK_WGPU_SHIM_BLESS=1 cargo build --release"
    );
}

/// Acceptance criterion 3, the shim's half: the header declares exactly the
/// functions S1 extracted from the generated artefact — no more, no fewer.
/// (The other half, that the generated C needs nothing outside them, is
/// checked on every link by `build_futhark.sh`.)
#[test]
fn the_header_declares_exactly_the_surface() {
    let declared: BTreeSet<&str> = GENERATED
        .lines()
        .filter_map(|l| {
            let at = l.find("wgpu")?;
            let rest = &l[at..];
            let end = rest.find('(')?;
            let name = &rest[..end];
            (name.chars().all(|c| c.is_ascii_alphanumeric()) && name.len() > 4 && name.as_bytes()[4].is_ascii_uppercase())
                .then_some(name)
        })
        .collect();
    let surface: BTreeSet<&str> = SURFACE.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    assert_eq!(surface.len(), 34, "surface/wgpu_functions.txt should hold the 34 functions S1 extracted");
    assert_eq!(declared, surface, "the header's functions and the extracted surface differ");
}

static CALLS: AtomicU32 = AtomicU32::new(0);
static STATUS: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn on_adapter(
    status: WGPURequestAdapterStatus,
    adapter: WGPUAdapter,
    _message: WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    CALLS.fetch_add(1, Ordering::SeqCst);
    STATUS.store(status, Ordering::SeqCst);
    unsafe { *(userdata1 as *mut WGPUAdapter) = adapter };
}

fn adapter_info(mode: WGPUCallbackMode, out: &mut WGPUAdapter) -> WGPURequestAdapterCallbackInfo {
    WGPURequestAdapterCallbackInfo {
        next_in_chain: std::ptr::null(),
        mode,
        callback: Some(on_adapter),
        userdata1: (out as *mut WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    }
}

static MAPS: AtomicU32 = AtomicU32::new(0);
static MAP_STATUS: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn on_map(status: WGPUMapAsyncStatus, _m: WGPUStringView, _u1: *mut c_void, _u2: *mut c_void) {
    MAPS.fetch_add(1, Ordering::SeqCst);
    MAP_STATUS.store(status, Ordering::SeqCst);
}

static DEVICES: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn on_device(
    status: WGPURequestDeviceStatus,
    device: WGPUDevice,
    _m: WGPUStringView,
    userdata1: *mut c_void,
    _u2: *mut c_void,
) {
    DEVICES.fetch_add(1, Ordering::SeqCst);
    assert_eq!(status, REQUEST_DEVICE_STATUS_SUCCESS, "the device request failed");
    unsafe { *(userdata1 as *mut WGPUDevice) = device };
}

/// The model's rules 3 and 4 through the C entry points, on the GPU:
///
/// * `WaitAnyOnly`: the callback does not run when the request starts; it
///   runs once, inside `wgpuInstanceWaitAny`, which marks the future
///   completed; a second wait on the delivered future is an error.
/// * `AllowSpontaneous`: the callback has run before the starting call
///   returns.
/// * A buffer mapped in `WaitAnyOnly` mode completes through the runtime's
///   own loop — `WaitAny` with a timeout of 0 — and its bytes are the ones
///   written.
///
/// One test, because the counters are process-wide.
#[test]
fn the_future_model_holds_through_the_c_entry_points() {
    unsafe {
        let instance = wgpu_create_instance(std::ptr::null());

        // WaitAnyOnly: nothing until WaitAny.
        let mut adapter: WGPUAdapter = std::ptr::null_mut();
        let before = CALLS.load(Ordering::SeqCst);
        let f = wgpu_instance_request_adapter_f(instance, std::ptr::null(), adapter_info(CALLBACK_MODE_WAIT_ANY_ONLY, &mut adapter));
        assert_eq!(CALLS.load(Ordering::SeqCst), before, "a WaitAnyOnly callback ran outside wgpuInstanceWaitAny");
        let mut info = [WGPUFutureWaitInfo { future: f, completed: 0 }];
        let w = wgpu_instance_wait_any(instance, 1, info.as_mut_ptr(), 0);
        assert_eq!(w, WAIT_STATUS_SUCCESS);
        assert_eq!(info[0].completed, 1);
        assert_eq!(CALLS.load(Ordering::SeqCst), before + 1, "the callback should have run exactly once");
        assert_eq!(STATUS.load(Ordering::SeqCst), REQUEST_ADAPTER_STATUS_SUCCESS);
        assert!(!adapter.is_null());
        let again = wgpu_instance_wait_any(instance, 1, info.as_mut_ptr(), 0);
        assert_eq!(again, WAIT_STATUS_ERROR, "a delivered future must not be waited on again");
        assert_eq!(CALLS.load(Ordering::SeqCst), before + 1, "a delivered callback ran a second time");

        // AllowSpontaneous: done before the call returns.
        let mut adapter2: WGPUAdapter = std::ptr::null_mut();
        let _ = wgpu_instance_request_adapter(instance, std::ptr::null(), adapter_info(CALLBACK_MODE_ALLOW_SPONTANEOUS, &mut adapter2));
        assert_eq!(CALLS.load(Ordering::SeqCst), before + 2, "an AllowSpontaneous callback had not run when the call returned");
        assert!(!adapter2.is_null());

        // A device, then a buffer mapped in WaitAnyOnly mode.
        let mut device: WGPUDevice = std::ptr::null_mut();
        let desc = WGPUDeviceDescriptor {
            next_in_chain: std::ptr::null(),
            label: WGPUStringView { data: std::ptr::null(), length: 0 },
            required_feature_count: 0,
            required_features: std::ptr::null(),
            required_limits: std::ptr::null(),
            uncaptured_error_callback_info: WGPUUncapturedErrorCallbackInfo {
                next_in_chain: std::ptr::null(),
                callback: None,
                userdata1: std::ptr::null_mut(),
                userdata2: std::ptr::null_mut(),
            },
        };
        let _ = wgpu_adapter_request_device(
            adapter,
            &desc,
            WGPURequestDeviceCallbackInfo {
                next_in_chain: std::ptr::null(),
                mode: CALLBACK_MODE_ALLOW_SPONTANEOUS,
                callback: Some(on_device),
                userdata1: (&mut device as *mut WGPUDevice).cast(),
                userdata2: std::ptr::null_mut(),
            },
        );
        assert_eq!(DEVICES.load(Ordering::SeqCst), 1);
        assert!(!device.is_null());
        let queue = wgpu_device_get_queue(device);
        assert_eq!(queue, wgpu_device_get_queue(device), "wgpuDeviceGetQueue must return the same queue");
        let buffer = wgpu_device_create_buffer(
            device,
            &WGPUBufferDescriptor {
                next_in_chain: std::ptr::null(),
                label: WGPUStringView { data: std::ptr::null(), length: 0 },
                usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
                size: 16,
                mapped_at_creation: 0,
            },
        );
        let data: [u32; 4] = [0xdead_beef, 1, 2, 0x8000_0001];
        wgpu_queue_write_buffer(queue, buffer, 0, data.as_ptr().cast(), 16);
        wgpu_queue_submit(queue, 0, std::ptr::null());
        let f = wgpu_buffer_map_async_f(
            buffer,
            MAP_MODE_READ,
            0,
            16,
            WGPUBufferMapCallbackInfo {
                next_in_chain: std::ptr::null(),
                mode: CALLBACK_MODE_WAIT_ANY_ONLY,
                callback: Some(on_map),
                userdata1: std::ptr::null_mut(),
                userdata2: std::ptr::null_mut(),
            },
        );
        assert_eq!(MAPS.load(Ordering::SeqCst), 0, "a WaitAnyOnly map callback ran outside wgpuInstanceWaitAny");
        let mut info = [WGPUFutureWaitInfo { future: f, completed: 0 }];
        let mut spins = 0u64;
        while info[0].completed == 0 {
            let w = wgpu_instance_wait_any(instance, 1, info.as_mut_ptr(), 0);
            assert!(w == WAIT_STATUS_SUCCESS || w == WAIT_STATUS_TIMED_OUT, "WaitAny returned {w}");
            spins += 1;
            assert!(spins < 50_000_000, "the map never completed");
        }
        assert_eq!(MAPS.load(Ordering::SeqCst), 1);
        assert_eq!(MAP_STATUS.load(Ordering::SeqCst), MAP_ASYNC_STATUS_SUCCESS);
        let p = wgpu_buffer_get_const_mapped_range(buffer, 0, 16).cast::<u32>();
        let got = std::slice::from_raw_parts(p, 4);
        assert_eq!(got, &data, "the mapped bytes are not the written ones");
        wgpu_buffer_unmap(buffer);
        assert_eq!(wgpu_buffer_get_size(buffer), 16);
        wgpu_buffer_destroy(buffer);
        wgpu_device_destroy(device);
    }
}
