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
    n_extrap: u32,
    /// WGSL rounds a uniform struct up to a 16-byte multiple. Nine u32 is 36
    /// bytes, so three more are uploaded whatever this says; naming them keeps
    /// what Rust writes and what the shader reads the same size by
    /// construction, rather than by luck.
    padding: [u32; 3],
}

/// The winner of each row: its TrueNorth angle, which candidate it was, the
/// three `1 - R2` the host still needs for the stop bar, and the angle that
/// CHOSE it.
///
/// `fitness` and `selection` differ only when `balanced_tournaments` is on, and
/// then the difference matters in both directions: TrueNorth is what the stop
/// bar and the hall of fame read, the balanced angle is what the tournament
/// reads. `Engine::evaluate` keeps them in `Scored::fitness` and
/// `Scored::selection`, and so does this.
pub struct HffWinners {
    pub fitness: Vec<f32>,
    pub selection: Vec<f32>,
    pub candidate: Vec<u32>,
    pub one_minus_r2: Vec<f32>,
}

/// ONE BEAT OF THE CANDIDATE WALK: the scoring kernel's output, and everything
/// needed to turn it into an angle. A struct rather than thirteen arguments,
/// because the four `bool`s at the end of a positional list are exactly the
/// kind of thing that gets transposed silently.
pub struct HffWork<'a> {
    /// `rows * candidates` blocks of `width` f32, as the scorer wrote them.
    pub scores: &'a [f32],
    pub rows: usize,
    pub candidates: usize,
    pub width: usize,
    /// Which of the nine objective columns HFF is over, and whether each is
    /// log scaled. The two are the same length.
    pub columns: &'a [u32],
    pub log_scaled: &'a [u32],
    pub col_max: &'a [f32],
    /// `var[3]` then `mad[3]`: the constant model's error on each split.
    pub caps: &'a [f32; 6],
    /// The tower depth of every candidate, `rows * candidates` of them.
    pub tower: &'a [u32],
    pub tower_on: bool,
    pub redundancy: bool,
    pub balanced: bool,
    /// How many extrapolation rows there are. Zero means that block does not
    /// exist and its 1-R2 stays zero, which is what `Caps::objectives` does.
    pub n_extrap: usize,
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
            entries: &(0..11)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match i {
                            0 => wgpu::BufferBindingType::Uniform,
                            // 7..=10 are the outputs.
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
    pub fn best(&self, evaluator: &GpuEvaluator, work: &HffWork) -> Result<HffWinners, String> {
        let HffWork { scores, rows, candidates, width, columns, log_scaled, col_max, caps, tower, tower_on, redundancy, balanced, n_extrap } = *work;
        if scores.len() != rows * candidates * width {
            return Err(format!("scores: {} values for {rows} x {candidates} x {width}", scores.len()));
        }
        if tower.len() != rows * candidates {
            return Err(format!("tower: {} values for {rows} x {candidates}", tower.len()));
        }
        if columns.len() != log_scaled.len() {
            return Err(format!("columns: {} ids against {} log flags", columns.len(), log_scaled.len()));
        }
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
            n_extrap: n_extrap as u32,
            padding: [0; 3],
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
        let sel_buf = out("hff selection", rows);

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &{
                let buffers: [&wgpu::Buffer; 11] =
                    [&uniform, &scores_buf, &columns_buf, &logs_buf, &max_buf, &caps_buf, &tower_buf, &fit_buf, &cand_buf, &omr2_buf, &sel_buf];
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

        // ONE STAGING BUFFER, ONE WAIT. This used to be four — a submit, a
        // map_async, a `Maintain::Wait` and a destroy for each of the four
        // outputs — and a round trip to the device costs the same whether it
        // carries four values or four hundred thousand. Measured on
        // strogatz_bacres1 at population 2,000, that fixed cost made the device
        // walk 11.98 s against the host's 4.65 s: the kernel was winning the
        // arithmetic and losing four times over on the journey. The four
        // results are copied into one buffer at known offsets and read back
        // together.
        let (fit_at, sel_at, cand_at, omr2_at) = (0u64, (rows * 4) as u64, (rows * 8) as u64, (rows * 12) as u64);
        let total = (rows * 24).max(4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hff-readback"),
            size: total,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let span = (rows * 4).max(4) as u64;
        enc.copy_buffer_to_buffer(&fit_buf, 0, &staging, fit_at, span);
        enc.copy_buffer_to_buffer(&sel_buf, 0, &staging, sel_at, span);
        enc.copy_buffer_to_buffer(&cand_buf, 0, &staging, cand_at, span);
        enc.copy_buffer_to_buffer(&omr2_buf, 0, &staging, omr2_at, (rows * 12).max(4) as u64);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|e| format!("map_async channel: {e}"))?.map_err(|e| format!("map_async: {e}"))?;
        let all = slice.get_mapped_range().to_vec();
        staging.unmap();
        staging.destroy();
        let at = |from: u64, len: usize| &all[from as usize..from as usize + len * 4];
        let fitness = bytemuck::cast_slice::<u8, f32>(at(fit_at, rows)).to_vec();
        let selection = bytemuck::cast_slice::<u8, f32>(at(sel_at, rows)).to_vec();
        let candidate = bytemuck::cast_slice::<u8, u32>(at(cand_at, rows)).to_vec();
        let one_minus_r2 = bytemuck::cast_slice::<u8, f32>(at(omr2_at, rows * 3)).to_vec();
        for b in [uniform, scores_buf, columns_buf, logs_buf, max_buf, caps_buf, tower_buf, fit_buf, cand_buf, omr2_buf, sel_buf] {
            b.destroy();
        }
        device.poll(wgpu::Maintain::Poll);
        Ok(HffWinners { fitness, selection, candidate, one_minus_r2 })
    }
}


#[cfg(test)]
mod tests {
    use crate::evolve::engine::{hff_scaled_for_test, hff_truenorth_for_test};

    /// THE TEST THAT ACTUALLY RUNS THE SHADER.
    ///
    /// The two tests below it check arithmetic, statement by statement, in Rust.
    /// That is worth having and it is not parity: it cannot catch a shader that
    /// fails to compile, binds the wrong buffer, walks the candidates in another
    /// order, or drops an output. The kernel shipped with exactly that last
    /// defect — it returned the angle it SELECTED by, so with
    /// `balanced_tournaments` on the balanced angle would have been handed to
    /// the stop bar as though it were TrueNorth — and both tests below passed
    /// the whole time.
    ///
    /// So: build a block of scores, dispatch, and compare against the host's own
    /// candidate walk (`host_candidate_winner`, the same code
    /// `Engine::evaluate` runs), with every flag on and off.
    #[test]
    fn the_dispatched_kernel_agrees_with_the_host_candidate_walk() {
        use crate::evolve::engine::{host_candidate_winner, hff_columns, HostWalk};
        use crate::gpu_eval::GpuEvaluator;

        const WIDTH: usize = 10;
        let (rows, per) = (7usize, 9usize);
        // Scores spanning what the scorer really emits: good fits, bad fits, an
        // unfitted candidate (NaN scale, skipped by both sides), and a couple of
        // near-perfect ones so the log scale is exercised where it is steepest.
        let mut scores = vec![0.0f32; rows * per * WIDTH];
        for r in 0..rows {
            for c in 0..per {
                let base = (r * per + c) * WIDTH;
                let k = (r * per + c) as f32;
                if c == 3 {
                    scores[base] = f32::NAN; // unfitted: not a choice
                    continue;
                }
                let err = if c == 5 { 1e-11 } else { 1e-3 * (1.0 + k) };
                scores[base] = 1.0 + 0.01 * k;          // a
                scores[base + 1] = 0.5;                  // b
                scores[base + 2] = err;                  // mse train
                scores[base + 3] = err * 1.5;            // mse val
                scores[base + 4] = err * 3.0;            // max err val
                scores[base + 5] = err * 2.0;            // mse extrap
                scores[base + 6] = err.sqrt();           // mae train
                scores[base + 7] = err.sqrt() * 1.5;     // mae val
                scores[base + 8] = err.sqrt() * 2.0;     // mae extrap
                scores[base + 9] = 0.25 + 0.01 * k;      // redundancy
            }
        }
        let caps_var = [1.0f64, 2.0, 4.0];
        let caps_mad = [0.8f64, 1.2, 1.6];
        let caps: [f32; 6] = [1.0, 2.0, 4.0, 0.8, 1.2, 1.6];
        let col_max = [1.0f64; 9];
        let col_max_f32 = [1.0f32; 9];
        let tower: Vec<u32> = (0..rows * per).map(|i| (i % 7) as u32).collect();

        // A tiny dataset only so the evaluator has a device; this kernel reads
        // none of it.
        let evaluator = GpuEvaluator::new(&[1.0, 2.0, 3.0, 4.0], 2).expect("device");
        let gpu = super::GpuHff::new(&evaluator).expect("pipeline");

        // n_extrap 0 and 3: with none, `hff_columns` drops the extrapolation
        // block entirely, which is what keeps the kernel from reading an
        // objective the host never computed.
        for n_extrap in [0usize, 3] {
            for log_scale in [[false, false, false], [true, true, true]] {
                for &(tower_on, redundancy, balanced) in &[
                    (false, false, false),
                    (true, false, false),
                    (false, true, false),
                    (false, false, true),
                    (true, true, true),
                ] {
                    let columns = hff_columns(n_extrap, false, log_scale);
                    let column_ids: Vec<u32> = columns.iter().map(|&(k, _)| k as u32).collect();
                    let logs: Vec<u32> = columns.iter().map(|&(_, l)| u32::from(l)).collect();
                    let winners = gpu
                        .best(
                            &evaluator,
                            &super::HffWork {
                                scores: &scores,
                                rows,
                                candidates: per,
                                width: WIDTH,
                                columns: &column_ids,
                                log_scaled: &logs,
                                col_max: &col_max_f32,
                                caps: &caps,
                                tower: &tower,
                                tower_on,
                                redundancy,
                                balanced,
                                n_extrap,
                            },
                        )
                        .expect("dispatch");
                    let what = format!("n_extrap {n_extrap}, log {log_scale:?}, tower {tower_on}, redundancy {redundancy}, balanced {balanced}");
                    for r in 0..rows {
                        let host_scores: Vec<f64> = scores[r * per * WIDTH..(r + 1) * per * WIDTH].iter().map(|&v| f64::from(v)).collect();
                        let host = host_candidate_winner(&HostWalk {
                            scores: &host_scores,
                            per,
                            caps_var,
                            caps_mad,
                            col_max,
                            columns: &columns,
                            tower_of: &tower[r * per..(r + 1) * per],
                            n_extrap,
                            redundancy,
                            tower_on,
                            balanced,
                        })
                        .expect("the host picks a winner");
                        assert_eq!(
                            winners.candidate[r] as usize, host.2,
                            "{what}: row {r} — the device chose candidate {} and the host chose {}",
                            winners.candidate[r], host.2
                        );
                        assert!(
                            (f64::from(winners.fitness[r]) - host.0).abs() < 1e-5,
                            "{what}: row {r} TrueNorth — device {} host {}",
                            winners.fitness[r], host.0
                        );
                        assert!(
                            (f64::from(winners.selection[r]) - host.1).abs() < 1e-5,
                            "{what}: row {r} selection — device {} host {}",
                            winners.selection[r], host.1
                        );
                    }
                }
            }
        }
    }

    /// AND THE TWO ANGLES MUST ACTUALLY DIFFER SOMEWHERE, or the test above
    /// would pass on a kernel that returned one of them twice — which is the
    /// bug it exists to catch.
    #[test]
    fn the_balanced_angle_is_not_the_truenorth_angle() {
        use crate::evolve::engine::{host_candidate_winner, hff_columns, HostWalk};
        const WIDTH: usize = 10;
        let per = 2usize;
        let mut scores = vec![0.0f64; per * WIDTH];
        for c in 0..per {
            let base = c * WIDTH;
            let err = if c == 0 { 1e-6 } else { 0.3 };
            scores[base] = 1.0;
            scores[base + 2] = err;
            scores[base + 3] = err;
            scores[base + 5] = err;
            scores[base + 6] = err;
            scores[base + 7] = err;
            scores[base + 8] = err;
        }
        let columns = hff_columns(0, false, [false, false, false]);
        let walk = |balanced: bool| {
            host_candidate_winner(&HostWalk {
                scores: &scores,
                per,
                caps_var: [1.0, 1.0, 1.0],
                caps_mad: [1.0, 1.0, 1.0],
                col_max: [1.0; 9],
                columns: &columns,
                tower_of: &[0, 0],
                n_extrap: 0,
                redundancy: false,
                tower_on: false,
                balanced,
            })
            .expect("a winner")
        };
        let w = walk(true);
        assert!(
            (w.0 - w.1).abs() > 1e-6,
            "the balanced angle {} and TrueNorth {} are the same number, so the parity test above proves nothing about which one the kernel returns",
            w.1, w.0
        );
    }

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
