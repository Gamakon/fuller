//! THE DEVICE'S WAITS AND READS (wgpu 30, 2026-10-19): `Device::poll` and
//! `BufferSlice::get_mapped_range` return Results in wgpu 30 where wgpu 0.20
//! returned nothing and a view; every call site goes through these, so a
//! poll that fails or a view that cannot be taken is an error by name, never
//! ignored.

/// Wait until the device has finished every submission (wgpu 0.20's
/// `Maintain::Wait`).
pub fn wait(device: &wgpu::Device) -> Result<(), String> {
    device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| format!("our software has a bug: the device's wait failed: {e}"))
}

/// Wait at most `ms` for every submission: true when the device finished,
/// false when the wait timed out (the work is still on the GPU).
pub fn wait_for(device: &wgpu::Device, ms: f64) -> Result<bool, String> {
    let timeout = std::time::Duration::from_secs_f64(ms / 1e3);
    match device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(timeout) }) {
        Ok(_) => Ok(true),
        Err(wgpu::PollError::Timeout) => Ok(false),
        Err(e) => Err(format!("our software has a bug: the device's wait failed: {e}")),
    }
}

/// Run the device's maintenance without waiting (wgpu 0.20's
/// `Maintain::Poll`): it frees what finished submissions held.
pub fn drain(device: &wgpu::Device) -> Result<(), String> {
    device.poll(wgpu::PollType::Poll).map(|_| ()).map_err(|e| format!("our software has a bug: the device's poll failed: {e}"))
}

/// The mapped view of a slice whose map has completed.
pub fn mapped(slice: &wgpu::BufferSlice<'_>) -> Result<wgpu::BufferView, String> {
    slice.get_mapped_range().map_err(|e| format!("our software has a bug: a mapped buffer's view: {e}"))
}
