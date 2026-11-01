//! Run the virtual-block generator (`experiments/virtual_blocks/wgsl/vblock.wgsl`) on the GPU.
//!
//!   cargo run --release --features gpu --example vblock_wgsl -- \
//!       <kernel.wgsl> <tables.f32> <params.f32> <seed> <out.f32> [reps]
//!
//! `params.f32` holds four f32 a block (m_mean, m_std, scale, bias); block i gets the offset i. The
//! output is 64 f32 a block. With `reps`, the dispatch is repeated and the median GPU wall time printed. With
//! `bpw` (blocks per workgroup, default 1), each workgroup loops over that many consecutive blocks; the kernel reads
//! it as the fifth u32 of its uniform. A seventh argument fixes the number of workgroups instead (a kernel that
//! maps workgroups to matrix rows). `VBLOCK_INNER=k` records k dispatches in one submission for the timed runs and
//! reports the time per dispatch, so the fixed cost of a submission and its wait is shared out.

use std::borrow::Cow;
use std::time::Instant;

use wgpu::util::DeviceExt;

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 5 {
        return Err("usage: vblock_wgsl <kernel.wgsl> <tables.f32> <params.f32> <seed> <out.f32> [reps]".into());
    }
    let src = std::fs::read_to_string(&a[0]).map_err(|e| format!("{}: {e}", a[0]))?;
    let tables = read_f32(&a[1])?;
    let params = read_f32(&a[2])?;
    let seed: u32 = a[3].parse().map_err(|e| format!("seed: {e}"))?;
    let reps: usize = a.get(5).map(|r| r.parse().map_err(|e| format!("reps: {e}"))).transpose()?.unwrap_or(0);
    let bpw: u32 = a.get(6).map(|r| r.parse().map_err(|e| format!("bpw: {e}"))).transpose()?.unwrap_or(1).max(1);
    let n = (params.len() / 4) as u32;
    let groups: u32 = match a.get(7) {
        Some(g) => g.parse().map_err(|e| format!("groups: {e}"))?,
        None => n.div_ceil(bpw),
    };
    let groups_x = groups.min(65_535);
    let groups_y = groups.div_ceil(groups_x);
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("vblock"), required_features: adapter.features() & wgpu::Features::SUBGROUP, required_limits: adapter.limits(), ..Default::default() })
            .await
            .map_err(|e| format!("request_device: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("vblock"), source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vblock"),
            layout: None,
            module: &module,
            entry_point: Some("gen_block"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE })
        };
        let p_buf = storage("params", &params);
        let t_buf = storage("tables", &tables);
        let size = u64::from(n) * 64 * 4;
        let o_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("out"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let r_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let cfg = [n, seed, groups_x, 0u32, bpw, 0, 0, 0];
        let c_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("cfg"), contents: bytemuck::cast_slice(&cfg), usage: wgpu::BufferUsages::UNIFORM });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: t_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: o_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
            ],
        });
        let inner: u32 = std::env::var("VBLOCK_INNER").ok().map(|v| v.parse().map_err(|e| format!("VBLOCK_INNER: {e}"))).transpose()?.unwrap_or(1).max(1);
        let run = |copy: bool| -> Result<f64, String> {
            let k = if copy { 1 } else { inner };
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                for _ in 0..k {
                    pass.dispatch_workgroups(groups_x, groups_y, 1);
                }
            }
            if copy {
                enc.copy_buffer_to_buffer(&o_buf, 0, &r_buf, 0, size);
            }
            let t0 = Instant::now();
            queue.submit(Some(enc.finish()));
            fuller::gpu_io::wait(&device)?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / f64::from(k))
        };
        run(true)?;
        let slice = r_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        fuller::gpu_io::wait(&device)?;
        rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
        let bytes = fuller::gpu_io::mapped(&slice)?.to_vec();
        r_buf.unmap();
        std::fs::write(&a[4], &bytes).map_err(|e| format!("{}: {e}", a[4]))?;
        println!("vblock: {n} blocks written to {}", a[4]);
        if reps > 0 {
            let mut ms: Vec<f64> = (0..reps).map(|_| run(false)).collect::<Result<_, _>>()?;
            ms.sort_by(f64::total_cmp);
            println!("vblock: {n} blocks, median {:.3} ms a dispatch over {reps} submissions of {inner} (min {:.3}, max {:.3})", ms[reps / 2], ms[0], ms[reps - 1]);
        }
        Ok(())
    })
}
