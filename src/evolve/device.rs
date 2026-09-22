//! The population resident on the device.
//!
//! Buffers are created once per fit and live on the device. A generation reads
//! the CURRENT set (genome, constants, fitness, wrapper) and writes the NEXT
//! one; the two then change places. `read` is the only thing that brings a
//! population back to the host.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use super::vary::{validate, GenParams, Generation, Island};
use super::{InitParams, Layout, Population, SymbolCodes, EVOLVE_WGSL, VARY_WGSL};

/// The uniform block of `evolve.wgsl`, field for field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InitUniform {
    pop: u32,
    n_genes: u32,
    head: u32,
    tail: u32,
    n_rnc: u32,
    n_functions: u32,
    n_terminals: u32,
    seed: u32,
    generation: u32,
    rnc_lo: i32,
    rnc_span: u32,
    n_wrappers: u32,
    first_row: u32,
    n_rows: u32,
    vhead: u32,
    pad1: u32,
}

/// The uniform block of `vary.wgsl`, field for field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GenUniform {
    pop: u32,
    n_genes: u32,
    head: u32,
    tail: u32,
    n_rnc: u32,
    n_functions: u32,
    n_terminals: u32,
    n_islands: u32,
    seed: u32,
    generation: u32,
    rnc_lo: i32,
    rnc_span: u32,
    // The fourteen rates moved into the island table — they are per-lane now.
    rnc_id: u32,
    vhead: u32,
}

/// One generation's worth of resident buffers.
struct Set {
    genome: wgpu::Buffer,
    rnc: wgpu::Buffer,
    fitness: wgpu::Buffer,
    wrapper: wgpu::Buffer,
}

pub struct EvolveDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    init_pipeline: wgpu::ComputePipeline,
    init_layout: wgpu::BindGroupLayout,
    vary_pipelines: [wgpu::ComputePipeline; 4],
    vary_layout: wgpu::BindGroupLayout,
    functions_buf: wgpu::Buffer,
    terminals_buf: wgpu::Buffer,
    arity_buf: wgpu::Buffer,
    parent_buf: wgpu::Buffer,
    stage1_buf: wgpu::Buffer,
    sets: [Set; 2],
    current: usize,
    layout: Layout,
    n_functions: u32,
    n_terminals: u32,
    rnc_id: u32,
}

fn bind_layout(device: &wgpu::Device, label: &str, n: u32, read_only: impl Fn(u32) -> bool) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &(0..n)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match i {
                        0 => wgpu::BufferBindingType::Uniform,
                        _ => wgpu::BufferBindingType::Storage { read_only: read_only(i) },
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    })
}

impl EvolveDevice {
    pub fn new(layout: Layout, codes: &SymbolCodes) -> Result<Self, String> {
        pollster::block_on(Self::new_async(layout, codes))
    }

    async fn new_async(layout: Layout, codes: &SymbolCodes) -> Result<Self, String> {
        codes.validate()?;
        if layout.genome_len() == 0 || layout.rnc_len() == 0 {
            return Err("an empty population has no buffers to make".into());
        }
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("no GPU adapter")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fuller-evolve-device"),
                    required_features: wgpu::Features::empty(),
                    // The adapter's real limits: a generation binds 14 storage
                    // buffers, more than wgpu's portable default of 8.
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let pipeline = |source: &'static str, label: &str, bind: &wgpu::BindGroupLayout, entry: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[bind],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: entry,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };
        let init_layout = bind_layout(&device, "fuller-evolve-init", 6, |i| i < 3);
        let vary_layout = bind_layout(&device, "fuller-evolve-vary", 15, |i| i < 9);
        let init_pipeline = pipeline(EVOLVE_WGSL, "fuller-evolve", &init_layout, "init_main");
        let vary_pipelines = [
            pipeline(VARY_WGSL, "fuller-vary", &vary_layout, "first_main"),
            pipeline(VARY_WGSL, "fuller-vary", &vary_layout, "select_main"),
            pipeline(VARY_WGSL, "fuller-vary", &vary_layout, "mutate_main"),
            pipeline(VARY_WGSL, "fuller-vary", &vary_layout, "crossover_main"),
        ];

        let table = |v: &[u32], label| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let resident = |len: usize, label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let set = || Set {
            genome: resident(layout.genome_len(), "genome"),
            rnc: resident(layout.rnc_len(), "rnc"),
            fitness: resident(layout.pop as usize, "fitness"),
            wrapper: resident(layout.pop as usize, "wrapper_id"),
        };
        Ok(EvolveDevice {
            functions_buf: table(&codes.sample_functions, "sample_functions"),
            terminals_buf: table(&codes.sample_terminals, "sample_terminals"),
            arity_buf: table(&codes.arity, "arity"),
            parent_buf: resident(layout.pop as usize, "parent"),
            stage1_buf: resident(layout.pop as usize, "stage1"),
            sets: [set(), set()],
            current: 0,
            n_functions: codes.sample_functions.len() as u32,
            n_terminals: codes.sample_terminals.len() as u32,
            rnc_id: codes.rnc_id.unwrap_or(u32::MAX),
            device,
            queue,
            init_pipeline,
            init_layout,
            vary_pipelines,
            vary_layout,
            layout,
        })
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    fn bind(&self, layout: &wgpu::BindGroupLayout, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
    }

    fn uniform<T: bytemuck::Pod>(&self, value: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform"),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Fill rows `first_row .. first_row + n_rows` of the current population
    /// with new random individuals, on the device. The whole population at the
    /// start of a fit; later, the rows the pump refills.
    pub fn init_rows(&self, p: &InitParams, first_row: u32, n_rows: u32) -> Result<(), String> {
        if p.rnc_hi < p.rnc_lo || p.n_wrappers == 0 {
            return Err("init: need rnc_lo <= rnc_hi and n_wrappers > 0".into());
        }
        if first_row + n_rows > self.layout.pop {
            return Err(format!("rows {first_row}..{} are outside a population of {}", first_row + n_rows, self.layout.pop));
        }
        if n_rows == 0 {
            return Ok(());
        }
        let l = self.layout;
        let params_buf = self.uniform(&InitUniform {
            pop: l.pop,
            n_genes: l.n_genes,
            head: l.head,
            tail: l.tail,
            n_rnc: l.n_rnc,
            n_functions: self.n_functions,
            n_terminals: self.n_terminals,
            seed: p.seed,
            generation: p.generation,
            rnc_lo: p.rnc_lo,
            rnc_span: (p.rnc_hi - p.rnc_lo + 1) as u32,
            n_wrappers: p.n_wrappers,
            first_row,
            n_rows,
            vhead: super::virtual_head(p.vhead, l)?,
            pad1: 0,
        });
        let now = &self.sets[self.current];
        let bind = self.bind(
            &self.init_layout,
            &[&params_buf, &self.functions_buf, &self.terminals_buf, &now.genome, &now.rnc, &now.wrapper],
        );
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n_rows.div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        params_buf.destroy();
        self.device.poll(wgpu::Maintain::Poll);
        Ok(())
    }

    /// The whole population, initialised on the device.
    pub fn init(&self, p: &InitParams) -> Result<(), String> {
        self.init_rows(p, 0, self.layout.pop)
    }

    /// Upload the current population's fitness (until scoring runs on the
    /// device too, this is how fitness arrives).
    pub fn write_fitness(&self, fitness: &[f32]) -> Result<(), String> {
        if fitness.len() != self.layout.pop as usize {
            return Err(format!("{} fitness values for a population of {}", fitness.len(), self.layout.pop));
        }
        self.queue.write_buffer(&self.sets[self.current].fitness, 0, bytemuck::cast_slice(fitness));
        Ok(())
    }

    /// Replace the current population's genome, constants and wrapper ids (the
    /// pump rearranges rows on the host until it is a kernel).
    pub fn write_population(&self, pop: &Population) -> Result<(), String> {
        if pop.layout != self.layout {
            return Err("write_population: a population of another shape".into());
        }
        let now = &self.sets[self.current];
        self.queue.write_buffer(&now.genome, 0, bytemuck::cast_slice(&pop.genome));
        self.queue.write_buffer(&now.rnc, 0, bytemuck::cast_slice(&pop.rnc));
        self.queue.write_buffer(&now.wrapper, 0, bytemuck::cast_slice(&pop.wrapper_id));
        Ok(())
    }

    /// One generation's variation phase on the device: the two tournaments, clone +
    /// mutate, recombine — four dispatches in one submission, nothing read back. The
    /// next population becomes the current one.
    pub fn vary(&mut self, islands: &[Island], p: &GenParams) -> Result<(), String> {
        validate(self.layout, islands)?;
        if p.rnc_hi < p.rnc_lo {
            return Err("vary: need rnc_lo <= rnc_hi".into());
        }
        let l = self.layout;
        let params_buf = self.uniform(&GenUniform {
            pop: l.pop,
            n_genes: l.n_genes,
            head: l.head,
            tail: l.tail,
            n_rnc: l.n_rnc,
            n_functions: self.n_functions,
            n_terminals: self.n_terminals,
            n_islands: islands.len() as u32,
            seed: p.seed,
            generation: p.generation,
            rnc_lo: p.rnc_lo,
            rnc_span: (p.rnc_hi - p.rnc_lo + 1) as u32,
            rnc_id: self.rnc_id,
            vhead: super::virtual_head(p.vhead, self.layout)?,
        });
        // The island table, and each island's own fourteen rates after it — the
        // swim lane's rules, uploaded with the table the dispatch already sends.
        // The order matches `struct Island` in vary.wgsl exactly.
        let flat: Vec<u32> = islands
            .iter()
            .flat_map(|i| {
                let r = i.rates;
                [
                    i.lo, i.hi, i.elites, i.tournsize,
                    r.mut_point, r.invert, r.is_transpose, r.ris_transpose, r.gene_transpose,
                    r.dc_point, r.invert_dc, r.transpose_dc, r.rnc_point,
                    r.cx_one_point, r.cx_two_point, r.cx_gene, r.cleanse, r.cleanse_collapse,
                ]
            })
            .collect();
        let islands_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("islands"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let (now, next) = (&self.sets[self.current], &self.sets[1 - self.current]);
        let bind = self.bind(
            &self.vary_layout,
            &[
                &params_buf,
                &self.functions_buf,
                &self.terminals_buf,
                &self.arity_buf,
                &islands_buf,
                &now.genome,
                &now.rnc,
                &now.fitness,
                &now.wrapper,
                &self.parent_buf,
                &next.genome,
                &next.rnc,
                &next.fitness,
                &next.wrapper,
                &self.stage1_buf,
            ],
        );
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // One pass per kernel: a pass boundary is the barrier between a kernel
        // that writes a buffer and the next that reads it.
        for pipeline in &self.vary_pipelines {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(l.pop.div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        params_buf.destroy();
        islands_buf.destroy();
        self.device.poll(wgpu::Maintain::Poll);
        self.current = 1 - self.current;
        Ok(())
    }

    /// Wait until the device has finished everything submitted so far.
    pub fn finish(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Bring the current population back to the host.
    pub fn read(&self) -> Result<Population, String> {
        let now = &self.sets[self.current];
        Ok(Population {
            layout: self.layout,
            genome: self.read_buffer::<u32>(&now.genome, self.layout.genome_len())?,
            rnc: self.read_buffer::<f32>(&now.rnc, self.layout.rnc_len())?,
            wrapper_id: self.read_buffer::<u32>(&now.wrapper, self.layout.pop as usize)?,
        })
    }

    /// THE LINEAGE EDGE the selection kernel just wrote: for every row of the
    /// generation `vary` produced, the row of the one before it that row was
    /// cloned from. `select_main` computes it every generation whatever else is
    /// on; this is the only thing that reads it back.
    pub fn read_parent(&self) -> Result<Vec<u32>, String> {
        self.read_buffer::<u32>(&self.parent_buf, self.layout.pop as usize)
    }

    /// The current population with its fitness.
    pub fn read_generation(&self) -> Result<Generation, String> {
        let fitness = self.read_buffer::<f32>(&self.sets[self.current].fitness, self.layout.pop as usize)?;
        Ok(Generation { pop: self.read()?, fitness })
    }

    fn read_buffer<T: bytemuck::Pod>(&self, source: &wgpu::Buffer, len: usize) -> Result<Vec<T>, String> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("map_async channel: {e}"))?
            .map_err(|e| format!("map_async: {e}"))?;
        let out = bytemuck::cast_slice::<u8, T>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        staging.destroy();
        self.device.poll(wgpu::Maintain::Poll);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{codes, params};
    use super::super::vary::tests::{islands, start};
    use super::super::vary::{vary, GenParams, Island, Rates};
    use super::super::{below, draw, init, Layout};
    use super::EvolveDevice;

    /// The kernel and the CPU reference are the same function: same seed, same
    /// population, bit for bit — which also proves the WGSL generator against
    /// the Rust one over every draw a population uses.
    #[test]
    fn the_device_population_is_the_cpu_reference_bit_for_bit() {
        let layout = Layout::for_arity(800, 3, 48, 2, 10);
        let dev = EvolveDevice::new(layout, &codes()).expect("device");
        for seed in [1, 7008, u32::MAX] {
            dev.init(&params(seed)).expect("init");
            let on_device = dev.read().expect("read");
            assert_eq!(on_device, init(layout, &codes(), &params(seed)).unwrap(), "seed {seed}");
            on_device.check(&codes()).unwrap();
        }
    }

    /// The pump refills a range of rows and leaves the rest alone.
    #[test]
    fn refilling_some_rows_touches_only_those_rows() {
        let layout = Layout::for_arity(100, 3, 16, 2, 10);
        let dev = EvolveDevice::new(layout, &codes()).expect("device");
        dev.init(&params(5)).expect("init");
        let before = dev.read().expect("read");
        let mut later = params(5);
        later.generation = 40;
        dev.init_rows(&later, 20, 80).expect("refill");
        let after = dev.read().expect("read");
        let width = (layout.n_genes * layout.gene_width()) as usize;
        assert_eq!(after.genome[..20 * width], before.genome[..20 * width]);
        assert_ne!(after.genome[20 * width..], before.genome[20 * width..]);
        let mut reference = init(layout, &codes(), &later).unwrap();
        reference.genome[..20 * width].copy_from_slice(&before.genome[..20 * width]);
        assert_eq!(after.genome, reference.genome);
    }

    /// The same, under a VIRTUAL HEAD that grows as the run goes (8, then 12): the
    /// init kernel and every head operator on the device agree with the CPU
    /// reference bit for bit, and no function is ever written past it.
    #[test]
    fn a_growing_virtual_head_matches_the_cpu_reference_bit_for_bit() {
        let codes = codes();
        let layout = Layout::for_arity(800, 3, 48, 2, 10);
        let born = crate::evolve::InitParams { seed: 13, generation: 0, rnc_lo: -100, rnc_hi: 100, n_wrappers: 3, vhead: 8 };
        let pop = crate::evolve::init(layout, &codes, &born).unwrap();
        let mut cpu = crate::evolve::vary::Generation { pop, fitness: (0..800).map(|r| below(draw(13, 0, r, 0, 99), 1_000_000) as f32).collect() };
        // every operator on, the cleanse included — carried by the ISLANDS, which
        // is what both the CPU reference and the device read.
        let rates = Rates::with_cleanse(layout, 0.5);
        let isl: Vec<Island> = islands().into_iter().map(|i| Island { rates, ..i }).collect();
        let mut dev = EvolveDevice::new(layout, &codes).expect("device");
        dev.init(&born).expect("init");
        assert_eq!(dev.read_generation().expect("read").pop, cpu.pop, "the init kernel under a virtual head");
        dev.write_fitness(&cpu.fitness).expect("fitness");
        for generation in 1..=20u32 {
            let vhead = if generation <= 10 { 8 } else { 12 };
            let p = GenParams { seed: 13, generation, rnc_lo: -100, rnc_hi: 100, vhead };
            cpu = vary(&cpu, &isl, &codes, &p).unwrap();
            dev.vary(&isl, &p).expect("vary");
            let on_device = dev.read_generation().expect("read");
            assert_eq!(on_device.pop, cpu.pop, "generation {generation}");
            let width = layout.gene_width() as usize;
            let past = on_device.pop.genome.chunks(width).any(|g| g[vhead as usize..48].iter().any(|&id| codes.arity[id as usize] > 0));
            assert!(!past, "generation {generation}: a function past virtual head {vhead}");
            let scored: Vec<f32> = (0..800).map(|r| below(draw(13, generation, r, 1, 99), 1_000_000) as f32).collect();
            cpu.fitness.copy_from_slice(&scored);
            dev.write_fitness(&scored).expect("fitness");
        }
        assert!(dev.vary(&isl, &GenParams { seed: 13, generation: 21, rnc_lo: -100, rnc_hi: 100, vhead: 49 }).is_err());
    }

    /// Twenty generations of select + mutate + crossover on the device, each
    /// compared with the CPU reference: genome, constants, wrapper and fitness
    /// (by bits — an unevaluated row is NaN) identical every generation.
    #[test]
    fn twenty_generations_of_variation_match_the_cpu_reference_bit_for_bit() {
        let (codes, isl) = (codes(), islands());
        let mut cpu = start(11);
        // every operator on, the cleansing mutation at a rate that exercises it —
        // carried by the ISLANDS now, which is what both sides read.
        let rates = Rates::with_cleanse(cpu.pop.layout, 0.5);
        let isl: Vec<Island> = isl.into_iter().map(|i| Island { rates, ..i }).collect();
        let mut dev = EvolveDevice::new(cpu.pop.layout, &codes).expect("device");
        dev.init(&params(11)).expect("init");
        dev.write_fitness(&cpu.fitness).expect("fitness");
        for generation in 1..=20u32 {
            let p = GenParams { seed: 11, generation, rnc_lo: -100, rnc_hi: 100, vhead: 0 };
            cpu = vary(&cpu, &isl, &codes, &p).unwrap();
            dev.vary(&isl, &p).expect("vary");
            let on_device = dev.read_generation().expect("read");
            assert_eq!(on_device.pop, cpu.pop, "generation {generation}");
            let bits = |f: &[f32]| f.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
            assert_eq!(bits(&on_device.fitness), bits(&cpu.fitness), "generation {generation}");
            on_device.pop.check(&codes).unwrap();
            // a stand-in for scoring, identical on both sides
            let scored: Vec<f32> = (0..800).map(|r| below(draw(11, generation, r, 1, 99), 1_000_000) as f32).collect();
            cpu.fitness.copy_from_slice(&scored);
            dev.write_fitness(&scored).expect("fitness");
        }
    }
}
