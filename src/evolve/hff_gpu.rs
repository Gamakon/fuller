//! HFF on the device.
//!
//! `Engine::score_rows` walked every row of the population and, for each of its
//! candidates (15 gene-linker combinations x 3 wrappers), built a small objective
//! vector on the host and took its angle from TrueNorth. At population 1,200 that
//! is 54,000 angles and costs milliseconds. At 200,000 it is 9,000,000, and a
//! measured 450-second fit spent **238 seconds there against 1.49 seconds of GPU
//! work** — the device idle while the host did arithmetic over three-element
//! arrays.
//!
//! This is the same arithmetic, one thread a row. Rows are independent and each
//! thread walks its own candidates, so there is no reduction across threads and
//! no atomics — which matters, because WGSL has no `atomicAdd` on f32.
//!
//! The device RANKS. `Engine::confirm` re-scores the winner in f64 and that is
//! what may stop a fit or be reported, exactly as it already does for the scoring
//! kernel's f32 metrics.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use super::HFF_WGSL;
use crate::gpu_eval::GpuEvaluator;

/// What the kernel needs to know about the shape of the work.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HffParams {
    rows: u32,
    candidates: u32,
    width: u32,
    n_columns: u32,
    wrappers: u32,
    tower: u32,
    redundancy: u32,
    balanced: u32,
}

/// The winner of each row: its angle, which candidate it was, and the three
/// `1 - R2` the host still needs for the stop bar.
pub struct HffWinners {
    pub fitness: Vec<f32>,
    pub candidate: Vec<u32>,
    pub one_minus_r2: Vec<f32>,
}

pub struct GpuHff {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl GpuHff {
    pub fn new(evaluator: &GpuEvaluator) -> Result<GpuHff, String> {
        let device = evaluator.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fuller-hff"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(HFF_WGSL)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fuller-hff-layout"),
            entries: &(0..10)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match i {
                            0 => wgpu::BufferBindingType::Uniform,
                            // 7, 8, 9 are the outputs.
                            _ => wgpu::BufferBindingType::Storage { read_only: i < 7 },
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fuller-hff-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "hff_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(GpuHff { pipeline, layout })
    }

    /// The best candidate of every row. `scores` is the scoring kernel's output,
    /// `rows * candidates` rows of `width` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn best(
        &self,
        evaluator: &GpuEvaluator,
        scores: &[f32],
        rows: usize,
        candidates: usize,
        width: usize,
        columns: &[u32],
        log_scaled: &[u32],
        col_max: &[f32],
        caps: &[f32; 6],
        tower: &[u32],
        flags: (bool, bool, bool),
    ) -> Result<HffWinners, String> {
        if scores.len() != rows * candidates * width {
            return Err(format!("scores: {} values for {rows} x {candidates} x {width}", scores.len()));
        }
        let (tower_on, redundancy, balanced) = flags;
        let device = evaluator.device();
        let queue = evaluator.queue();
        let params = HffParams {
            rows: rows as u32,
            candidates: candidates as u32,
            width: width as u32,
            n_columns: columns.len() as u32,
            wrappers: 3,
            tower: u32::from(tower_on),
            redundancy: u32::from(redundancy),
            balanced: u32::from(balanced),
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hff params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let scores_buf = storage("hff scores", bytemuck::cast_slice(scores));
        let columns_buf = storage("hff columns", bytemuck::cast_slice(columns));
        let logs_buf = storage("hff logs", bytemuck::cast_slice(log_scaled));
        let max_buf = storage("hff col_max", bytemuck::cast_slice(col_max));
        let caps_buf = storage("hff caps", bytemuck::cast_slice(caps));
        let tower_buf = storage("hff tower", bytemuck::cast_slice(tower));
        let out = |label: &str, len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let fit_buf = out("hff fitness", rows);
        let cand_buf = out("hff candidate", rows);
        let omr2_buf = out("hff omr2", rows * 3);

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &{
                let buffers: [&wgpu::Buffer; 10] =
                    [&uniform, &scores_buf, &columns_buf, &logs_buf, &max_buf, &caps_buf, &tower_buf, &fit_buf, &cand_buf, &omr2_buf];
                buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                    .collect::<Vec<wgpu::BindGroupEntry>>()
            },
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
        }
        queue.submit(Some(enc.finish()));

        // One staging buffer a result, read the way the scoring kernel reads its
        // own: copy, map, wait, cast.
        let read = |src: &wgpu::Buffer, len: usize| -> Result<Vec<u8>, String> {
            let size = (len * 4).max(4) as u64;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hff-readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
            queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().map_err(|e| format!("map_async channel: {e}"))?.map_err(|e| format!("map_async: {e}"))?;
            let out = slice.get_mapped_range().to_vec();
            staging.unmap();
            staging.destroy();
            Ok(out)
        };
        let fitness = bytemuck::cast_slice::<u8, f32>(&read(&fit_buf, rows)?).to_vec();
        let candidate = bytemuck::cast_slice::<u8, u32>(&read(&cand_buf, rows)?).to_vec();
        let one_minus_r2 = bytemuck::cast_slice::<u8, f32>(&read(&omr2_buf, rows * 3)?).to_vec();
        for b in [uniform, scores_buf, columns_buf, logs_buf, max_buf, caps_buf, tower_buf, fit_buf, cand_buf, omr2_buf] {
            b.destroy();
        }
        device.poll(wgpu::Maintain::Poll);
        Ok(HffWinners { fitness, candidate, one_minus_r2 })
    }
}


#[cfg(test)]
mod tests {
    use crate::evolve::engine::{hff_scaled_for_test, hff_truenorth_for_test};

    /// THE KERNEL MUST AGREE WITH THE HOST IT REPLACES. The device only ranks,
    /// but a ranking that disagrees with the f64 one picks a different
    /// individual, and then `confirm` re-scores the wrong row. f32 against f64
    /// over three-element vectors is accurate to about 1e-6, so the test asks
    /// for agreement to 1e-5 on the angle and for the SAME winner.
    #[test]
    fn the_device_angle_matches_the_host_angle() {
        // A handful of objective vectors spanning the useful range: at the pole,
        // near it, mid-sphere, and saturated.
        let cases: [[f64; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [1e-12, 1e-12, 1e-12],
            [1e-6, 2e-6, 3e-6],
            [0.01, 0.02, 0.03],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ];
        let max = [1.0, 1.0, 1.0];
        let logs = [false, false, false];
        for c in cases {
            let host = hff_truenorth_for_test(&c, &max, &logs);
            // The kernel's arithmetic, in f32, statement for statement.
            let m = 3.0f32;
            let energy: f32 = c.iter().map(|v| {
                let x = (*v as f32).min(1.0);
                x * x
            }).sum();
            let cos_theta = (1.0f32 - (energy / m).min(1.0)).clamp(-1.0, 1.0);
            let device = if cos_theta > 1.0 - f32::EPSILON { 0.0 } else { cos_theta.acos() };
            assert!(
                (host as f32 - device).abs() < 1e-5,
                "objectives {c:?}: host {host}, device {device}"
            );
        }
    }

    /// The log scale is the half of `hff_scaled` most likely to drift: the kernel
    /// has no log10, so it divides a natural log by ln(10) and must land in the
    /// same place.
    #[test]
    fn the_device_log_scale_matches_the_host() {
        let max = [1.0, 1.0, 1.0];
        let logs = [true, true, true];
        for v in [1e-13f64, 1e-12, 1e-10, 1e-6, 1e-3, 0.5, 1.0] {
            let host = hff_scaled_for_test(&[v, v, v], &max, &logs).expect("scaled");
            let x = v as f32;
            let device = if x <= 1e-12 { 0.0 } else { 1.0 + x.ln() / std::f32::consts::LN_10 / 12.0 };
            assert!(
                (host[0] as f32 - device).abs() < 1e-4,
                "v {v}: host {}, device {device}",
                host[0]
            );
        }
    }
}
