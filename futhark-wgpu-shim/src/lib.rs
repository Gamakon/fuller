//! The WebGPU C API that Futhark's webgpu host runtime calls, implemented
//! over `wgpu-rs` — `docs/PLAN_futhark_wgpu_host.md` (route H2).
//!
//! Futhark's experimental `webgpu` backend writes a host program in C against
//! `<webgpu/webgpu.h>`. `wgpu-native` has no C API, so this crate is one: the
//! 34 functions the generated C calls (`surface/wgpu_functions.txt`, taken
//! from the generated artefact), each a marshalling onto the `wgpu-rs` call
//! behind it, and the header, written by `cbindgen` from these definitions
//! on every build.
//!
//! * [`types`] — the C types of the surface;
//! * [`handles`] — what a C handle points at;
//! * [`futures`] — the future/callback model (the plan's §4.5);
//! * [`api`] — the entry points;
//! * [`report`] — dispatch counts and GPU pass timestamps.

pub mod api;
pub mod futures;
pub mod handles;
pub mod report;
pub mod types;

/// Stop the process with a message naming what the shim met and does not
/// implement (or what went wrong underneath it). Never a silent fallback: a
/// missing piece of the surface is a bug to fix, and a Futhark program that
/// carried on past it would compute something nobody asked for.
pub fn fatal(msg: &str) -> ! {
    eprintln!("futhark-wgpu-shim: {msg}");
    std::process::abort()
}

#[cfg(test)]
mod tests;
