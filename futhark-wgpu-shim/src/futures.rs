//! The future/callback model: `docs/PLAN_futhark_wgpu_host.md` §4.5.
//!
//! A `WGPUFuture` is an id into the instance's [`Registry`]. The `wgpu-rs`
//! callback of an operation only **fills a slot**; the C callback runs only in
//! [`Registry::deliver`], which is called from exactly two places:
//!
//! * `wgpuInstanceWaitAny`, for a `WGPUCallbackMode_WaitAnyOnly` future, and
//! * the call that started an `WGPUCallbackMode_AllowSpontaneous` request,
//!   which drives it to completion before it returns ([`Registry::start`]).
//!
//! `wgpu` fires its own closures inside `Device::poll`; a C callback never
//! runs there.

use crate::handles::{make, WGPUAdapterImpl, WGPUDeviceImpl};
use crate::report;
use crate::types::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What an operation produced. Filled by a `wgpu-rs` callback, so it is
/// `Send`.
pub enum Completion {
    Map(Result<(), String>),
    WorkDone,
    Adapter(Result<wgpu::Adapter, String>),
    Device(Result<(wgpu::Device, wgpu::Queue), String>),
    Compilation(wgpu::CompilationInfo),
}

/// The slot an operation's `wgpu-rs` callback fills.
pub type Slot = Arc<Mutex<Option<Completion>>>;

/// A new, empty slot.
pub fn slot() -> Slot {
    Arc::new(Mutex::new(None))
}

/// Fill a slot. A poisoned lock means a panic elsewhere already stopped the
/// process's logic; the value is stored regardless.
pub fn fill(slot: &Slot, value: Completion) {
    let mut guard = slot.lock().unwrap_or_else(|p| p.into_inner());
    *guard = Some(value);
}

fn is_full(slot: &Slot) -> bool {
    slot.lock().unwrap_or_else(|p| p.into_inner()).is_some()
}

fn take(slot: &Slot) -> Option<Completion> {
    slot.lock().unwrap_or_else(|p| p.into_inner()).take()
}

/// The two userdata pointers C hands in and gets back.
#[derive(Clone, Copy)]
pub struct UserData {
    pub u1: *mut c_void,
    pub u2: *mut c_void,
}

/// The device's uncaptured-error callback, as the device descriptor gave it.
#[derive(Clone, Copy)]
pub struct ErrorCallback {
    pub callback: WGPUUncapturedErrorCallback,
    pub user: UserData,
}

/// The C callback a future delivers to.
pub enum Deliver {
    Map(WGPUBufferMapCallback, UserData),
    WorkDone(WGPUQueueWorkDoneCallback, UserData),
    Adapter(WGPURequestAdapterCallback, UserData),
    Device(WGPURequestDeviceCallback, UserData, ErrorCallback, bool),
    Compilation(WGPUCompilationInfoCallback, UserData),
}

struct Pending {
    slot: Slot,
    /// The device whose poll completes this future; `None` for a future
    /// whose slot was filled when it was started.
    device: Option<wgpu::Device>,
    deliver: Deliver,
}

/// The futures an instance has issued and not yet delivered.
pub struct Registry {
    next: AtomicU64,
    pending: RefCell<BTreeMap<u64, Pending>>,
    /// Error messages handed to C. Futhark's runtime keeps the view after the
    /// callback returns (it prints it in its panic), so the text must outlive
    /// the call: it lives as long as the instance.
    messages: RefCell<Vec<CString>>,
}

impl Default for Registry {
    fn default() -> Self {
        Registry { next: AtomicU64::new(1), pending: RefCell::new(BTreeMap::new()), messages: RefCell::new(Vec::new()) }
    }
}

const EMPTY: WGPUStringView = WGPUStringView { data: std::ptr::null(), length: 0 };

impl Registry {
    /// Start a future in `mode` (the plan's §4.5, rules 3–5).
    pub fn start(
        self: &Rc<Self>,
        mode: WGPUCallbackMode,
        slot: Slot,
        device: Option<wgpu::Device>,
        deliver: Deliver,
    ) -> WGPUFuture {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.pending.borrow_mut().insert(id, Pending { slot, device, deliver });
        match mode {
            CALLBACK_MODE_WAIT_ANY_ONLY => {
                report::count_future(false);
            }
            CALLBACK_MODE_ALLOW_SPONTANEOUS => {
                report::count_future(true);
                self.drive(id);
                self.deliver(id);
            }
            CALLBACK_MODE_ALLOW_PROCESS_EVENTS => crate::fatal(
                "a request in WGPUCallbackMode_AllowProcessEvents: the surface has no wgpuInstanceProcessEvents",
            ),
            other => crate::fatal(&format!("a request in an unknown WGPUCallbackMode {other}")),
        }
        WGPUFuture { id }
    }

    /// Poll a future's device until its slot is full.
    fn drive(&self, id: u64) {
        loop {
            let (full, device) = {
                let pending = self.pending.borrow();
                let p = pending.get(&id).unwrap_or_else(|| crate::fatal("drive: a future the registry does not hold"));
                (is_full(&p.slot), p.device.clone())
            };
            if full {
                return;
            }
            let Some(device) = device else {
                crate::fatal("a future with no device to poll was started empty");
            };
            match device.poll(wgpu::PollType::wait_indefinitely()) {
                Ok(status) => {
                    let still_empty = !is_full(&self.pending.borrow()[&id].slot);
                    if still_empty && status.is_queue_empty() {
                        crate::fatal("a request never completed: the device's queue is empty and its callback has not fired");
                    }
                }
                Err(e) => crate::fatal(&format!("the device's wait failed: {e}")),
            }
        }
    }

    /// `wgpuInstanceWaitAny`: rule 3 of the plan's §4.5.
    pub fn wait_any(self: &Rc<Self>, infos: &mut [WGPUFutureWaitInfo], timeout_ns: u64) -> WGPUWaitStatus {
        report::count_wait_any();
        // Every listed id must be one this registry issued and still holds.
        let mut devices: Vec<wgpu::Device> = Vec::new();
        {
            let pending = self.pending.borrow();
            for info in infos.iter() {
                match pending.get(&info.future.id) {
                    Some(p) => {
                        if let Some(d) = &p.device {
                            if !devices.iter().any(|x| x == d) {
                                devices.push(d.clone());
                            }
                        }
                    }
                    None => return WAIT_STATUS_ERROR,
                }
            }
        }
        let poll = if timeout_ns == 0 {
            wgpu::PollType::Poll
        } else {
            wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_nanos(timeout_ns)) }
        };
        for d in &devices {
            match d.poll(poll.clone()) {
                Ok(_) | Err(wgpu::PollError::Timeout) => {}
                Err(e) => crate::fatal(&format!("the device's poll failed: {e}")),
            }
        }
        let mut any = false;
        for info in infos.iter_mut() {
            let full = is_full(&self.pending.borrow()[&info.future.id].slot);
            if full {
                self.deliver(info.future.id);
                info.completed = 1;
                any = true;
            }
        }
        if any {
            WAIT_STATUS_SUCCESS
        } else {
            WAIT_STATUS_TIMED_OUT
        }
    }

    /// Keep an error message alive for the instance's lifetime and view it.
    fn message(&self, text: &str) -> WGPUStringView {
        let c = CString::new(text.replace('\0', " ")).unwrap_or_default();
        let view = WGPUStringView { data: c.as_ptr(), length: c.as_bytes().len() };
        self.messages.borrow_mut().push(c);
        view
    }

    /// Remove a completed future and run its C callback, once.
    fn deliver(self: &Rc<Self>, id: u64) {
        let Some(p) = self.pending.borrow_mut().remove(&id) else {
            crate::fatal("deliver: a future the registry does not hold");
        };
        let Some(done) = take(&p.slot) else {
            crate::fatal("deliver: a future whose slot is empty");
        };
        // SAFETY (all arms): the callback and the userdata are what the C
        // caller registered for this request; they are handed back exactly
        // once, with arguments of the types the header declares.
        match (p.deliver, done) {
            (Deliver::Map(cb, u), Completion::Map(r)) => {
                let (status, msg) = match r {
                    Ok(()) => (MAP_ASYNC_STATUS_SUCCESS, EMPTY),
                    Err(e) => (MAP_ASYNC_STATUS_ERROR, self.message(&e)),
                };
                if let Some(cb) = cb {
                    unsafe { cb(status, msg, u.u1, u.u2) }
                }
            }
            (Deliver::WorkDone(cb, u), Completion::WorkDone) => {
                if let Some(cb) = cb {
                    unsafe { cb(QUEUE_WORK_DONE_STATUS_SUCCESS, EMPTY, u.u1, u.u2) }
                }
            }
            (Deliver::Adapter(cb, u), Completion::Adapter(r)) => {
                let (status, handle, msg) = match r {
                    Ok(adapter) => {
                        report::set_adapter(&adapter.get_info());
                        let registry = Rc::clone(self);
                        (REQUEST_ADAPTER_STATUS_SUCCESS, make(WGPUAdapterImpl { adapter, registry }), EMPTY)
                    }
                    Err(e) => (REQUEST_ADAPTER_STATUS_UNAVAILABLE, std::ptr::null_mut(), self.message(&e)),
                };
                if let Some(cb) = cb {
                    unsafe { cb(status, handle, msg, u.u1, u.u2) }
                }
            }
            (Deliver::Device(cb, u, errors, timestamps), Completion::Device(r)) => {
                let (status, handle, msg) = match r {
                    Ok((device, queue)) => {
                        (REQUEST_DEVICE_STATUS_SUCCESS, self.new_device(device, queue, errors, timestamps), EMPTY)
                    }
                    Err(e) => (REQUEST_DEVICE_STATUS_ERROR, std::ptr::null_mut(), self.message(&e)),
                };
                if let Some(cb) = cb {
                    unsafe { cb(status, handle, msg, u.u1, u.u2) }
                }
            }
            (Deliver::Compilation(cb, u), Completion::Compilation(info)) => {
                // The views point into `info`, which outlives the call.
                let messages: Vec<WGPUCompilationMessage> = info
                    .messages
                    .iter()
                    .map(|m| {
                        let loc = m.location.as_ref();
                        WGPUCompilationMessage {
                            next_in_chain: std::ptr::null(),
                            message: WGPUStringView { data: m.message.as_ptr().cast(), length: m.message.len() },
                            kind: match m.message_type {
                                wgpu::CompilationMessageType::Error => COMPILATION_MESSAGE_TYPE_ERROR,
                                wgpu::CompilationMessageType::Warning => COMPILATION_MESSAGE_TYPE_WARNING,
                                wgpu::CompilationMessageType::Info => COMPILATION_MESSAGE_TYPE_INFO,
                            },
                            line_num: loc.map_or(0, |l| u64::from(l.line_number)),
                            line_pos: loc.map_or(0, |l| u64::from(l.line_position)),
                            offset: loc.map_or(0, |l| u64::from(l.offset)),
                            length: loc.map_or(0, |l| u64::from(l.length)),
                        }
                    })
                    .collect();
                let c_info = WGPUCompilationInfo {
                    next_in_chain: std::ptr::null(),
                    message_count: messages.len(),
                    messages: messages.as_ptr(),
                };
                if let Some(cb) = cb {
                    unsafe { cb(COMPILATION_INFO_REQUEST_STATUS_SUCCESS, &c_info, u.u1, u.u2) }
                }
            }
            _ => crate::fatal("deliver: a future's completion does not match its callback"),
        }
    }

    fn new_device(
        self: &Rc<Self>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        errors: ErrorCallback,
        timestamps: bool,
    ) -> *mut WGPUDeviceImpl {
        let handle = make(WGPUDeviceImpl {
            device: device.clone(),
            queue,
            registry: Rc::clone(self),
            timestamps,
            queue_handle: RefCell::new(None),
            self_cell: Box::new(std::ptr::null_mut()),
            pipelines: RefCell::new(std::collections::HashMap::new()),
        });
        // SAFETY: `handle` was made just above and is never freed.
        let cell: *const WGPUDevice = unsafe {
            *(*handle).self_cell = handle;
            &*(*handle).self_cell
        };
        if let Some(cb) = errors.callback {
            let sink = ErrorSink { cb, user: errors.user, device: cell };
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| sink.call(e)));
        } else {
            // No callback: an error must still not pass silently.
            device.on_uncaptured_error(Arc::new(|e: wgpu::Error| {
                crate::fatal(&format!("an uncaptured WebGPU error and no callback to report it to: {e}"))
            }));
        }
        handle
    }
}

/// The C side's uncaptured-error callback, captured for `wgpu`'s handler.
struct ErrorSink {
    cb: unsafe extern "C" fn(*const WGPUDevice, WGPUErrorType, WGPUStringView, *mut c_void, *mut c_void),
    user: UserData,
    device: *const WGPUDevice,
}

// SAFETY: the pointers are the C caller's. The shim never dereferences them;
// it hands them back to the C callback the caller registered for exactly
// this. `wgpu` requires the handler to be Send + Sync because it may be
// called from any thread that hits an error; Futhark's runtime is
// single-threaded, so in practice it is called on the runtime's thread.
unsafe impl Send for ErrorSink {}
unsafe impl Sync for ErrorSink {}

impl ErrorSink {
    fn call(&self, e: wgpu::Error) {
        let kind = match e {
            wgpu::Error::OutOfMemory { .. } => ERROR_TYPE_OUT_OF_MEMORY,
            wgpu::Error::Validation { .. } => ERROR_TYPE_VALIDATION,
            wgpu::Error::Internal { .. } => ERROR_TYPE_INTERNAL,
        };
        let text = format!("{e}");
        let view = WGPUStringView { data: text.as_ptr().cast(), length: text.len() };
        // SAFETY: see `unsafe impl Send for ErrorSink`.
        unsafe { (self.cb)(self.device, kind, view, self.user.u1, self.user.u2) }
    }
}
