//! ST9: the image harness. A compute kernel that draws one picture into a write-only `rgba16float` storage texture from
//! uniforms (the cuneus visual shaders, `~/Dev/gamakon/cuneus/examples/shaders`) is run headless at fixed uniforms and a
//! fixed resolution, the texture is read back, saved as a PNG and timed.
//!
//! Why here and not in phylu: the device, the Metal error scopes and the work guard (`work_guard`) are fuller's, the
//! kernels this runs are rendered by fuller's kingdom (`true_kingdom`), and phylu depends on fuller (never the other
//! way). phylu's fitness calls [`ImageKernel`] and [`compare`].
//!
//! Every device run goes through the work guard: the caller passes the kernel's [`Fuel`] to [`ImageKernel::compile`] and
//! [`ImageKernel::run`] checks `fuel.check(..)` with the dispatch's invocation count before it encodes or submits
//! anything. The shipped original is `Fuel::Proven(..)`; an evolved candidate takes the kingdom's fuel.
//!
//! The host-side pieces (half-float decode, picture comparison, PNG writing) are not gated on the device.

/// A binding the case supplies. The bind group layouts are built from these, not from the kernel, so a candidate that
/// drops a binding still runs against the same layout as the original.
#[derive(Debug, Clone)]
pub enum BindingData {
    /// A uniform block with its bytes.
    Uniform { group: u32, binding: u32, bytes: Vec<u8> },
    /// A read-write storage buffer of zeroes (the host clears it before the pass).
    ZeroStorage { group: u32, binding: u32, bytes: u64 },
    /// The write-only `rgba16float` output texture.
    Output { group: u32, binding: u32 },
}

impl BindingData {
    fn group(&self) -> u32 {
        match self {
            BindingData::Uniform { group, .. } | BindingData::ZeroStorage { group, .. } | BindingData::Output { group, .. } => *group,
        }
    }
}

/// Little-endian bytes of a list of `f32`/`u32`/`i32` words.
pub fn words_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A picture: width, height, RGBA as `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<f32>,
}

/// IEEE half to single.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// How two pictures of one size differ: every statistic is over the RGB channels of every pixel (the alpha is 1 in
/// the cuneus shaders and is compared too, in `max_abs`).
#[derive(Debug, Clone, PartialEq)]
pub struct Difference {
    pub pixels: usize,
    /// Channels whose half-float bits differ (the two pictures are the same picture when this is 0).
    pub differing_channels: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    /// Pixels whose largest channel difference exceeds 1/255 (one 8-bit step of the saved PNG).
    pub pixels_over_one_step: usize,
    /// A channel is non-finite in one picture and not the same in the other.
    pub non_finite_mismatch: usize,
}

/// The per-pixel error statistics of `got` against `want`. A size mismatch is an error.
pub fn compare(got: &Picture, want: &Picture) -> Result<Difference, String> {
    if got.width != want.width || got.height != want.height {
        return Err(format!("pictures of different sizes: {}x{} against {}x{}", got.width, got.height, want.width, want.height));
    }
    let n = (got.width * got.height) as usize;
    let (mut differing, mut max_abs, mut sum, mut sq, mut over, mut nonfinite) = (0usize, 0f64, 0f64, 0f64, 0usize, 0usize);
    for p in 0..n {
        let mut worst = 0f64;
        for c in 0..4 {
            let (a, b) = (got.rgba[p * 4 + c], want.rgba[p * 4 + c]);
            if a.to_bits() != b.to_bits() {
                differing += 1;
            }
            if !a.is_finite() || !b.is_finite() {
                if a.to_bits() != b.to_bits() {
                    nonfinite += 1;
                }
                continue;
            }
            let d = f64::from((a - b).abs());
            max_abs = max_abs.max(d);
            worst = worst.max(d);
            if c < 3 {
                sum += d;
                sq += d * d;
            }
        }
        if worst > 1.0 / 255.0 {
            over += 1;
        }
    }
    let m = (n * 3).max(1) as f64;
    Ok(Difference { pixels: n, differing_channels: differing, max_abs, mean_abs: sum / m, rmse: (sq / m).sqrt(), pixels_over_one_step: over, non_finite_mismatch: nonfinite })
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut c = 0xffff_ffffu32;
    for b in data {
        c = table[((c ^ u32::from(*b)) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xffff_ffff
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for x in data {
        a = (a + u32::from(*x)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let mut crc_in = kind.to_vec();
    crc_in.extend_from_slice(body);
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

/// The picture as an 8-bit RGB PNG (stored, uncompressed deflate blocks): each channel clamped to [0, 1] and rounded.
/// No gamma is applied: the cuneus shaders write their display value (`pow(c, 1/gamma)`), which is what their host shows.
pub fn png_bytes(p: &Picture) -> Vec<u8> {
    let (w, h) = (p.width as usize, p.height as usize);
    let mut raw = Vec::with_capacity(h * (w * 3 + 1));
    for y in 0..h {
        raw.push(0u8);
        for x in 0..w {
            for c in 0..3 {
                let v = p.rgba[(y * w + x) * 4 + c];
                let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                raw.push((v * 255.0 + 0.5) as u8);
            }
        }
    }
    let mut z = vec![0x78u8, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(u8::from(chunks.peek().is_none()));
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&p.width.to_be_bytes());
    ihdr.extend_from_slice(&p.height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

pub fn write_png(path: &std::path::Path, p: &Picture) -> Result<(), String> {
    std::fs::write(path, png_bytes(p)).map_err(|e| format!("{}: {e}", path.display()))
}

/// The median of a list (the mean of the middle two for an even count).
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match s.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => s[n / 2],
        n => (s[n / 2 - 1] + s[n / 2]) / 2.0,
    }
}

#[cfg(feature = "gpu")]
pub use device::{ImageKernel, ImageRun};

#[cfg(feature = "gpu")]
mod device {
    use super::{half_to_f32, median, BindingData, Picture};
    use crate::wgsl::oracle::device::{Device, ErrorScopes};
    use crate::wgsl::work_guard::Fuel;
    use std::time::Instant;

    /// A compiled image kernel, its bind groups and the texture it draws into.
    pub struct ImageKernel<'a> {
        dev: &'a Device,
        pipeline: wgpu::ComputePipeline,
        bind_groups: Vec<(u32, wgpu::BindGroup)>,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
        workgroup: (u32, u32),
        fuel: Fuel,
        name: String,
        zero_buffers: Vec<wgpu::Buffer>,
    }

    /// One measured run: the picture of the last pass and the milliseconds of every timed pass.
    #[derive(Debug, Clone)]
    pub struct ImageRun {
        pub picture: Picture,
        /// Wall milliseconds, submit to completion, of each timed pass.
        pub wall_ms: Vec<f64>,
        /// GPU timestamp milliseconds of each timed pass.
        pub gpu_ms: Vec<f64>,
    }

    impl ImageRun {
        pub fn median_wall_ms(&self) -> f64 {
            median(&self.wall_ms)
        }
        pub fn median_gpu_ms(&self) -> f64 {
            median(&self.gpu_ms)
        }
    }

    impl<'a> ImageKernel<'a> {
        /// Compile `source`'s entry point `entry` against the case's bindings at `width` x `height`. `fuel` is the
        /// kernel's work guard (never a default: the caller names it).
        pub fn compile(dev: &'a Device, name: &str, source: &str, entry: &str, bindings: &[BindingData], width: u32, height: u32, fuel: Fuel) -> Result<ImageKernel<'a>, String> {
            fuel.refuse_unported()?;
            let module = crate::wgsl::naga_shape::parse_unscanned(source).map_err(|e| format!("naga: {e}"))?;
            let ep = module.entry_points.iter().find(|e| e.name == entry).ok_or_else(|| format!("no entry point {entry}"))?;
            let workgroup = (ep.workgroup_size[0], ep.workgroup_size[1]);
            if ep.workgroup_size[2] != 1 {
                return Err(format!("workgroup z is {}, not 1", ep.workgroup_size[2]));
            }
            let device = &dev.device;
            let groups = bindings.iter().map(BindingData::group).max().map_or(0, |g| g + 1);
            let mut layouts = Vec::new();
            for g in 0..groups {
                let mut entries = Vec::new();
                for b in bindings.iter().filter(|b| b.group() == g) {
                    let (binding, ty) = match b {
                        BindingData::Uniform { binding, .. } => (*binding, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                        BindingData::ZeroStorage { binding, .. } => (*binding, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }),
                        BindingData::Output { binding, .. } => (*binding, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }),
                    };
                    entries.push(wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None });
                }
                layouts.push(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("image bgl"), entries: &entries }));
            }
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("image output"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut zero_buffers = Vec::new();
            let mut bind_groups = Vec::new();
            for g in 0..groups {
                let mut buffers: Vec<(u32, wgpu::Buffer)> = Vec::new();
                for b in bindings.iter().filter(|b| b.group() == g) {
                    match b {
                        BindingData::Uniform { binding, bytes, .. } => {
                            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image uniform"), size: (bytes.len() as u64).next_multiple_of(16), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                            dev.queue.write_buffer(&buf, 0, bytes);
                            buffers.push((*binding, buf));
                        }
                        BindingData::ZeroStorage { binding, bytes, .. } => {
                            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image zero storage"), size: (*bytes).next_multiple_of(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                            buffers.push((*binding, buf));
                        }
                        BindingData::Output { .. } => {}
                    }
                }
                let mut entries: Vec<wgpu::BindGroupEntry> = buffers.iter().map(|(binding, buf)| wgpu::BindGroupEntry { binding: *binding, resource: buf.as_entire_binding() }).collect();
                for b in bindings.iter().filter(|b| b.group() == g) {
                    if let BindingData::Output { binding, .. } = b {
                        entries.push(wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::TextureView(&view) });
                    }
                }
                bind_groups.push((g, device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("image bind"), layout: &layouts[g as usize], entries: &entries })));
                zero_buffers.extend(buffers.into_iter().map(|(_, b)| b));
            }
            let scopes = ErrorScopes::push(device);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("image kernel"), source: wgpu::ShaderSource::Wgsl(source.into()) });
            let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("image layout"), bind_group_layouts: &layout_refs, immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("image pipeline"), layout: Some(&pl), module: &shader, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
            if let Some(e) = scopes.pop() {
                return Err(format!("pipeline: {e}"));
            }
            Ok(ImageKernel { dev, pipeline, bind_groups, texture, width, height, workgroup, fuel, name: name.to_string(), zero_buffers })
        }

        /// Workgroups of the dispatch and the invocations they launch.
        pub fn dispatch(&self) -> (u32, u32, u64) {
            let (gx, gy) = (self.width.div_ceil(self.workgroup.0), self.height.div_ceil(self.workgroup.1));
            (gx, gy, u64::from(gx) * u64::from(gy) * u64::from(self.workgroup.0) * u64::from(self.workgroup.1))
        }

        /// `warmup` untimed passes, then `repeats` timed ones; the picture is read back after the last. The work
        /// guard is checked, with the invocation count of one pass, before anything is encoded or submitted.
        pub fn run(&self, warmup: usize, repeats: usize) -> Result<ImageRun, String> {
            let (gx, gy, invocations) = self.dispatch();
            self.fuel.check(&format!("an image run of {}", self.name), invocations)?;
            let device = &self.dev.device;
            let queue = &self.dev.queue;
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("image timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image stamps"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let stamps = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image stamps read"), size: 16, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let period = f64::from(queue.get_timestamp_period());
            let (mut wall, mut gpu) = (Vec::new(), Vec::new());
            for i in 0..warmup + repeats {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image pass") });
                for b in &self.zero_buffers {
                    if b.usage().contains(wgpu::BufferUsages::STORAGE) {
                        enc.clear_buffer(b, 0, None);
                    }
                }
                {
                    let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("image"), timestamp_writes });
                    pass.set_pipeline(&self.pipeline);
                    for (g, bg) in &self.bind_groups {
                        pass.set_bind_group(*g, bg, &[]);
                    }
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                enc.resolve_query_set(&query_set, 0..2, &resolve, 0);
                enc.copy_buffer_to_buffer(&resolve, 0, &stamps, 0, 16);
                let t = Instant::now();
                queue.submit(Some(enc.finish()));
                let (tx, rx) = std::sync::mpsc::channel();
                stamps.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).expect("the receiver waits for the mapping");
                });
                crate::gpu_io::wait(device)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
                let ns = {
                    let mapped = crate::gpu_io::mapped(&stamps.slice(..))?;
                    let s: &[u64] = bytemuck::cast_slice(&mapped);
                    s[1].wrapping_sub(s[0]) as f64 * period
                };
                stamps.unmap();
                if i >= warmup {
                    wall.push(ms);
                    gpu.push(ns / 1e6);
                }
            }
            let picture = self.read_back()?;
            Ok(ImageRun { picture, wall_ms: wall, gpu_ms: gpu })
        }

        fn read_back(&self) -> Result<Picture, String> {
            let device = &self.dev.device;
            let row_bytes = (self.width * 8).next_multiple_of(256);
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image read"), size: u64::from(row_bytes) * u64::from(self.height), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image read") });
            enc.copy_texture_to_buffer(
                self.texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(self.height) } },
                wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
            );
            self.dev.queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).expect("the receiver waits for the mapping");
            });
            crate::gpu_io::wait(device)?;
            rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
            let mut rgba = Vec::with_capacity((self.width * self.height * 4) as usize);
            {
                let mapped = crate::gpu_io::mapped(&buf.slice(..))?;
                for y in 0..self.height as usize {
                    let row = &mapped[y * row_bytes as usize..y * row_bytes as usize + self.width as usize * 8];
                    for h in row.chunks_exact(2) {
                        rgba.push(half_to_f32(u16::from_le_bytes([h[0], h[1]])));
                    }
                }
            }
            buf.unmap();
            Ok(Picture { width: self.width, height: self.height, rgba })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_decode() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert!(half_to_f32(0x7e00).is_nan());
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn crc_and_adler_match_their_known_values() {
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    #[test]
    fn a_png_has_the_signature_and_the_three_chunks() {
        let p = Picture { width: 2, height: 2, rgba: vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0] };
        let b = png_bytes(&p);
        assert_eq!(&b[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(&b[12..16], b"IHDR");
        assert!(b.windows(4).any(|w| w == b"IDAT") && b.ends_with(&[0xae, 0x42, 0x60, 0x82]));
    }

    #[test]
    fn comparison_counts_the_differing_channels_and_steps() {
        let a = Picture { width: 2, height: 1, rgba: vec![0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0] };
        let mut b = a.clone();
        assert_eq!(compare(&a, &b).expect("same size").differing_channels, 0);
        b.rgba[4] = 0.75;
        let d = compare(&a, &b).expect("same size");
        assert_eq!((d.differing_channels, d.pixels_over_one_step), (1, 1));
        assert!((d.max_abs - 0.25).abs() < 1e-9 && (d.mean_abs - 0.25 / 6.0).abs() < 1e-9);
        assert!(compare(&a, &Picture { width: 1, height: 2, rgba: vec![0.0; 8] }).is_err());
    }

    #[test]
    fn the_median_of_an_even_and_an_odd_list() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
