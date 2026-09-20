//! The population resident on the device.
//!
//! Buffers are created once per fit and live on the device; a kernel reads and
//! writes them in place, and `read` is the only thing that brings a population
//! back to the host.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use super::{InitParams, Layout, Population, SymbolCodes, EVOLVE_WGSL};

/// The uniform block of `evolve.wgsl`, field for field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
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
    pad0: u32,
    pad1: u32,
}

pub struct EvolveDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    init_pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
    functions_buf: wgpu::Buffer,
    terminals_buf: wgpu::Buffer,
    genome_buf: wgpu::Buffer,
    rnc_buf: wgpu::Buffer,
    wrapper_buf: wgpu::Buffer,
    layout: Layout,
    n_functions: u32,
    n_terminals: u32,
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
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fuller-evolve"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EVOLVE_WGSL)),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fuller-evolve-layout"),
            entries: &(0..6)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match i {
                            0 => wgpu::BufferBindingType::Uniform,
                            _ => wgpu::BufferBindingType::Storage { read_only: i < 3 },
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
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fuller-evolve-init"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "init_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

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
        Ok(EvolveDevice {
            functions_buf: table(&codes.sample_functions, "sample_functions"),
            terminals_buf: table(&codes.sample_terminals, "sample_terminals"),
            genome_buf: resident(layout.genome_len(), "genome"),
            rnc_buf: resident(layout.rnc_len(), "rnc"),
            wrapper_buf: resident(layout.pop as usize, "wrapper_id"),
            n_functions: codes.sample_functions.len() as u32,
            n_terminals: codes.sample_terminals.len() as u32,
            device,
            queue,
            init_pipeline,
            bind_layout,
            layout,
        })
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Fill rows `first_row .. first_row + n_rows` with new random individuals,
    /// on the device. The whole population at the start of a fit; later, the
    /// rows the pump refills.
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
        let params = Params {
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
            pad0: 0,
            pad1: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &params_buf,
            &self.functions_buf,
            &self.terminals_buf,
            &self.genome_buf,
            &self.rnc_buf,
            &self.wrapper_buf,
        ];
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_layout,
            entries: &entries,
        });
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

    /// Bring the population back to the host.
    pub fn read(&self) -> Result<Population, String> {
        Ok(Population {
            layout: self.layout,
            genome: self.read_buffer::<u32>(&self.genome_buf, self.layout.genome_len())?,
            rnc: self.read_buffer::<f32>(&self.rnc_buf, self.layout.rnc_len())?,
            wrapper_id: self.read_buffer::<u32>(&self.wrapper_buf, self.layout.pop as usize)?,
        })
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
    use super::super::{init, Layout};
    use super::EvolveDevice;

    /// The kernel and the CPU reference are the same function: same seed, same
    /// population, bit for bit — which also proves the WGSL generator (pcg,
    /// mulhi) against the Rust one over every draw a population uses.
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
}
