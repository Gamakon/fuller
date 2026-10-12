//! The adapter the WGSL kingdom's kernels run on: its compute limits (the
//! workgroup sizes and the workgroup memory a kernel may use) and whether it
//! grants the features the kingdom spells only when present (subgroups,
//! shader f16, timestamp queries). The limits are what the safe kingdom's
//! grid rows and workgroup arrays must stay within.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_adapter
fn main() -> Result<(), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok_or("no adapter")?;
    let info = adapter.get_info();
    let l = adapter.limits();
    let f = adapter.features();
    println!("adapter\t{} ({:?}, driver {})", info.name, info.backend, info.driver_info);
    println!("max_compute_invocations_per_workgroup\t{}", l.max_compute_invocations_per_workgroup);
    println!("max_compute_workgroup_size\t{} x {} x {}", l.max_compute_workgroup_size_x, l.max_compute_workgroup_size_y, l.max_compute_workgroup_size_z);
    println!("max_compute_workgroup_storage_size\t{} bytes", l.max_compute_workgroup_storage_size);
    println!("max_compute_workgroups_per_dimension\t{}", l.max_compute_workgroups_per_dimension);
    println!("max_storage_buffer_binding_size\t{} bytes", l.max_storage_buffer_binding_size);
    println!("min/max subgroup size\t{} / {}", l.min_subgroup_size, l.max_subgroup_size);
    for (name, feature) in [("SUBGROUP", wgpu::Features::SUBGROUP), ("SUBGROUP_BARRIER", wgpu::Features::SUBGROUP_BARRIER), ("SHADER_F16", wgpu::Features::SHADER_F16), ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY), ("SHADER_INT64", wgpu::Features::SHADER_INT64)] {
        println!("feature {name}\t{}", f.contains(feature));
    }
    Ok(())
}
