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
    /// The bind group this binding is in.
    pub fn group(&self) -> u32 {
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

/// The picture's 8-bit RGB scanlines as a zlib stream of stored (uncompressed) deflate blocks: each channel clamped
/// to [0, 1] and rounded, NaN as 0.
fn zlib_rgb(p: &Picture) -> Vec<u8> {
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
    z
}

fn png_head(out: &mut Vec<u8>, width: u32, height: u32) {
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(out, b"IHDR", &ihdr);
}

/// The picture as an 8-bit RGB PNG (stored, uncompressed deflate blocks): each channel clamped to [0, 1] and rounded.
/// No gamma is applied: the cuneus shaders write their display value (`pow(c, 1/gamma)`), which is what their host shows.
pub fn png_bytes(p: &Picture) -> Vec<u8> {
    let mut out = Vec::new();
    png_head(&mut out, p.width, p.height);
    chunk(&mut out, b"IDAT", &zlib_rgb(p));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// An animated PNG (APNG) of `frames`, all of one size, each shown `delay_ms`, looping forever.
pub fn apng_bytes(frames: &[Picture], delay_ms: u16) -> Result<Vec<u8>, String> {
    let first = frames.first().ok_or("an animation of no frames")?;
    if frames.iter().any(|f| f.width != first.width || f.height != first.height) {
        return Err("the frames of an animation differ in size".into());
    }
    let mut out = Vec::new();
    png_head(&mut out, first.width, first.height);
    let mut actl = (frames.len() as u32).to_be_bytes().to_vec();
    actl.extend_from_slice(&0u32.to_be_bytes());
    chunk(&mut out, b"acTL", &actl);
    let mut seq = 0u32;
    for (i, f) in frames.iter().enumerate() {
        let mut fctl = seq.to_be_bytes().to_vec();
        seq += 1;
        for v in [f.width, f.height, 0, 0] {
            fctl.extend_from_slice(&v.to_be_bytes());
        }
        fctl.extend_from_slice(&delay_ms.to_be_bytes());
        fctl.extend_from_slice(&1000u16.to_be_bytes());
        fctl.extend_from_slice(&[0, 0]);
        chunk(&mut out, b"fcTL", &fctl);
        let z = zlib_rgb(f);
        if i == 0 {
            chunk(&mut out, b"IDAT", &z);
        } else {
            let mut fdat = seq.to_be_bytes().to_vec();
            seq += 1;
            fdat.extend_from_slice(&z);
            chunk(&mut out, b"fdAT", &fdat);
        }
    }
    chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// 3 x 5 glyphs, one row of three bits each, top row first.
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c {
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [7, 1, 7, 4, 7],
        '3' => [7, 1, 7, 1, 7],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 7, 1, 7],
        '6' => [7, 4, 7, 5, 7],
        '7' => [7, 1, 1, 1, 1],
        '8' => [7, 5, 7, 5, 7],
        '9' => [7, 5, 7, 1, 7],
        '.' => [0, 0, 0, 0, 2],
        '#' => [5, 7, 5, 7, 5],
        'm' => [0, 0, 7, 7, 5],
        's' => [0, 3, 6, 3, 6],
        'x' => [0, 5, 2, 2, 5],
        ' ' => [0, 0, 0, 0, 0],
        _ => return None,
    })
}

/// Stamp `text` (digits, `.`, `#`, `m`, `s`, `x`, space) at `(x, y)` in white on a black box, each glyph pixel
/// `scale` x `scale`; text that runs off the picture is clipped. A character without a glyph is an error, and then
/// nothing is drawn.
pub fn stamp(p: &mut Picture, text: &str, x: u32, y: u32, scale: u32) -> Result<(), String> {
    let glyphs = text.chars().map(|c| glyph(c).ok_or_else(|| format!("no glyph for {c:?}"))).collect::<Result<Vec<_>, _>>()?;
    let s = scale.max(1);
    let n = glyphs.len() as u32;
    let (bw, bh) = ((4 * n + 1) * s, 7 * s);
    let mut put = |px: u32, py: u32, v: f32| {
        if px < p.width && py < p.height {
            let i = ((py * p.width + px) * 4) as usize;
            p.rgba[i..i + 3].copy_from_slice(&[v, v, v]);
            p.rgba[i + 3] = 1.0;
        }
    };
    for yy in 0..bh {
        for xx in 0..bw {
            put(x + xx, y + yy, 0.0);
        }
    }
    for (i, g) in glyphs.iter().enumerate() {
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3u32 {
                if bits >> (2 - col) & 1 == 1 {
                    for dy in 0..s {
                        for dx in 0..s {
                            put(x + (1 + 4 * i as u32 + col) * s + dx, y + (1 + row as u32) * s + dy, 1.0);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Pictures of one height side by side, left to right.
pub fn side_by_side(pictures: &[Picture]) -> Result<Picture, String> {
    let h = pictures.first().ok_or("no pictures")?.height;
    if pictures.iter().any(|p| p.height != h) {
        return Err("pictures of different heights".into());
    }
    let w: u32 = pictures.iter().map(|p| p.width).sum();
    let mut rgba = vec![0f32; (w * h * 4) as usize];
    let mut x0 = 0u32;
    for p in pictures {
        for y in 0..h {
            let src = &p.rgba[(y * p.width * 4) as usize..((y + 1) * p.width * 4) as usize];
            let at = ((y * w + x0) * 4) as usize;
            rgba[at..at + src.len()].copy_from_slice(src);
        }
        x0 += p.width;
    }
    Ok(Picture { width: w, height: h, rgba })
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

/// The bind group of the tiled dispatch's uniform (`DispatchTile { origin: vec4<u32>, grid: vec4<u32> }`, the image
/// interfaces' prologue, `kingdoms/wgsl_true/interface.py`): the harness binds it for every kernel, a shader that does not
/// declare it ignores it.
pub const TILE_GROUP: u32 = 3;

/// The submits of a dispatch of `gx` x `gy` workgroups when one submit may hold at most `per_submit` whole
/// workgroups: rectangles `(x0, y0, w, h)` in workgroups, row by row, covering every workgroup exactly once. `None`
/// (a proven kernel) is one rectangle.
pub fn tiles(gx: u32, gy: u32, per_submit: Option<u64>) -> Vec<(u32, u32, u32, u32)> {
    let Some(n) = per_submit else { return vec![(0, 0, gx, gy)] };
    let n = n.max(1);
    let mut out = Vec::new();
    if n >= u64::from(gx) {
        let rows = u32::try_from((n / u64::from(gx.max(1))).min(u64::from(gy))).unwrap_or(gy).max(1);
        let mut y = 0;
        while y < gy {
            let h = rows.min(gy - y);
            out.push((0, y, gx, h));
            y += h;
        }
    } else {
        let w = u32::try_from(n).unwrap_or(gx).max(1);
        for y in 0..gy {
            let mut x = 0;
            while x < gx {
                let ww = w.min(gx - x);
                out.push((x, y, ww, 1));
                x += ww;
            }
        }
    }
    out
}

#[cfg(feature = "gpu")]
pub use device::{ImageKernel, ImageRun};

#[cfg(feature = "gpu")]
mod device {
    use super::{half_to_f32, median, tiles, BindingData, Picture, TILE_GROUP};
    use crate::wgsl::oracle::device::{Device, ErrorScopes};
    use crate::wgsl::work_guard::{Fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT};
    use std::time::Instant;

    /// One submit of the tiled dispatch: its rectangle of workgroups and the bind group of its `DispatchTile`.
    struct Tile {
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        bind: wgpu::BindGroup,
        _uniform: wgpu::Buffer,
    }

    /// A compiled image kernel, its bind groups and the texture it draws into.
    pub struct ImageKernel<'a> {
        dev: &'a Device,
        pipeline: wgpu::ComputePipeline,
        layouts: Vec<wgpu::BindGroupLayout>,
        bind_groups: Vec<(u32, wgpu::BindGroup)>,
        /// The case's uniform buffers by `(group, binding)`, so a pass at another time rewrites them in place.
        uniforms: Vec<((u32, u32), wgpu::Buffer)>,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
        workgroup: (u32, u32),
        fuel: Fuel,
        /// The per-submit cap (the machine's, or lower in a test that forces a split).
        cap: u128,
        tiles: Vec<Tile>,
        name: String,
        zero_buffers: Vec<wgpu::Buffer>,
    }

    /// One measured run: the picture of the last pass and the milliseconds of every timed pass.
    #[derive(Debug, Clone)]
    pub struct ImageRun {
        pub picture: Picture,
        /// Wall milliseconds, first submit to the last completion, of each timed pass.
        pub wall_ms: Vec<f64>,
        /// GPU timestamp milliseconds of each timed pass (the sum over its submits).
        pub gpu_ms: Vec<f64>,
        /// Submits a pass took.
        pub submits: usize,
    }

    impl ImageRun {
        pub fn median_wall_ms(&self) -> f64 {
            median(&self.wall_ms)
        }
        pub fn median_gpu_ms(&self) -> f64 {
            median(&self.gpu_ms)
        }
    }

    /// What a size and a set of bindings need on the device: the texture, the bind groups, the uniform buffers and the
    /// tiles of the dispatch.
    struct Target {
        bind_groups: Vec<(u32, wgpu::BindGroup)>,
        uniforms: Vec<((u32, u32), wgpu::Buffer)>,
        texture: wgpu::Texture,
        tiles: Vec<Tile>,
        zero_buffers: Vec<wgpu::Buffer>,
    }

    fn target(dev: &Device, layouts: &[wgpu::BindGroupLayout], bindings: &[BindingData], width: u32, height: u32, workgroup: (u32, u32), fuel: &Fuel, cap: u128) -> Result<Target, String> {
        if let Some(b) = bindings.iter().find(|b| b.group() >= TILE_GROUP) {
            return Err(format!("our software has a bug: a case binding in group {} (group {TILE_GROUP} is the dispatch tile's)", b.group()));
        }
        let device = &dev.device;
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
        let mut uniforms = Vec::new();
        let mut bind_groups = Vec::new();
        for g in 0..TILE_GROUP {
            let mut buffers: Vec<(u32, wgpu::Buffer)> = Vec::new();
            for b in bindings.iter().filter(|b| b.group() == g) {
                match b {
                    BindingData::Uniform { binding, bytes, .. } => {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image uniform"), size: (bytes.len() as u64).next_multiple_of(16), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                        dev.queue.write_buffer(&buf, 0, bytes);
                        uniforms.push(((g, *binding), buf.clone()));
                        buffers.push((*binding, buf));
                    }
                    BindingData::ZeroStorage { binding, bytes, .. } => {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image zero storage"), size: (*bytes).next_multiple_of(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                        zero_buffers.push(buf.clone());
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
        }
        // the submits: whole workgroups, as many as the kernel's bound lets one submit hold under the cap
        let (gx, gy) = (width.div_ceil(workgroup.0), height.div_ceil(workgroup.1));
        let per_submit = fuel.workgroups_per_submit(u64::from(workgroup.0) * u64::from(workgroup.1), cap)?;
        let mut tile_list = Vec::new();
        for (x0, y0, w, h) in tiles(gx, gy, per_submit) {
            let words = [x0, y0, 0, 0, gx, gy, 1, 0];
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dispatch tile"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            dev.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&words));
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dispatch tile"), layout: &layouts[TILE_GROUP as usize], entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
            tile_list.push(Tile { x0, y0, w, h, bind, _uniform: buf });
        }
        Ok(Target { bind_groups, uniforms, texture, tiles: tile_list, zero_buffers })
    }

    impl<'a> ImageKernel<'a> {
        /// Compile `source`'s entry point `entry` against the case's bindings at `width` x `height`. `fuel` is the
        /// kernel's work guard (never a default: the caller names it); the dispatch is split into submits of whole
        /// workgroups, each bounded by at most the machine's cap.
        pub fn compile(dev: &'a Device, name: &str, source: &str, entry: &str, bindings: &[BindingData], width: u32, height: u32, fuel: Fuel) -> Result<ImageKernel<'a>, String> {
            Self::compile_under(dev, name, source, entry, bindings, width, height, fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)
        }

        /// [`ImageKernel::compile`] with a per-submit cap at most the machine's (a test lowers it to force a split).
        pub fn compile_under(dev: &'a Device, name: &str, source: &str, entry: &str, bindings: &[BindingData], width: u32, height: u32, fuel: Fuel, cap: u128) -> Result<ImageKernel<'a>, String> {
            fuel.refuse_unported()?;
            let cap = cap.min(CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT);
            let module = crate::wgsl::naga_shape::parse_unscanned(source).map_err(|e| format!("naga: {e}"))?;
            let ep = module.entry_points.iter().find(|e| e.name == entry).ok_or_else(|| format!("no entry point {entry}"))?;
            let workgroup = (ep.workgroup_size[0], ep.workgroup_size[1]);
            if ep.workgroup_size[2] != 1 {
                return Err(format!("workgroup z is {}, not 1", ep.workgroup_size[2]));
            }
            let device = &dev.device;
            let tile_entry = wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None };
            let mut layouts = Vec::new();
            for g in 0..=TILE_GROUP {
                let mut entries = Vec::new();
                if g == TILE_GROUP {
                    entries.push(tile_entry);
                }
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
            let t = target(dev, &layouts, bindings, width, height, workgroup, &fuel, cap)?;
            let scopes = ErrorScopes::push(device);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("image kernel"), source: wgpu::ShaderSource::Wgsl(source.into()) });
            let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("image layout"), bind_group_layouts: &layout_refs, immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("image pipeline"), layout: Some(&pl), module: &shader, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
            if let Some(e) = scopes.pop() {
                return Err(format!("pipeline: {e}"));
            }
            Ok(ImageKernel { dev, pipeline, layouts, bind_groups: t.bind_groups, uniforms: t.uniforms, texture: t.texture, width, height, workgroup, fuel, cap, tiles: t.tiles, name: name.to_string(), zero_buffers: t.zero_buffers })
        }

        /// The same compiled pipeline drawing at another size with another case's bindings (no recompile).
        pub fn resized(&self, bindings: &[BindingData], width: u32, height: u32) -> Result<ImageKernel<'a>, String> {
            let t = target(self.dev, &self.layouts, bindings, width, height, self.workgroup, &self.fuel, self.cap)?;
            Ok(ImageKernel {
                dev: self.dev,
                pipeline: self.pipeline.clone(),
                layouts: self.layouts.clone(),
                bind_groups: t.bind_groups,
                uniforms: t.uniforms,
                texture: t.texture,
                width,
                height,
                workgroup: self.workgroup,
                fuel: self.fuel.clone(),
                cap: self.cap,
                tiles: t.tiles,
                name: self.name.clone(),
                zero_buffers: t.zero_buffers,
            })
        }

        /// Rewrite the bytes of the case's uniform at `(group, binding)` (a pass at another time).
        pub fn set_uniform(&self, group: u32, binding: u32, bytes: &[u8]) -> Result<(), String> {
            let (_, buf) = self.uniforms.iter().find(|(k, _)| *k == (group, binding)).ok_or_else(|| format!("our software has a bug: no uniform at group {group} binding {binding}"))?;
            if bytes.len() as u64 > buf.size() {
                return Err(format!("our software has a bug: {} bytes into a uniform of {}", bytes.len(), buf.size()));
            }
            self.dev.queue.write_buffer(buf, 0, bytes);
            Ok(())
        }

        /// Workgroups of the whole dispatch and the invocations they launch.
        pub fn dispatch(&self) -> (u32, u32, u64) {
            let (gx, gy) = (self.width.div_ceil(self.workgroup.0), self.height.div_ceil(self.workgroup.1));
            (gx, gy, u64::from(gx) * u64::from(gy) * u64::from(self.workgroup.0) * u64::from(self.workgroup.1))
        }

        /// The submits one pass takes.
        pub fn submits(&self) -> usize {
            self.tiles.len()
        }

        /// Each submit's rectangle `(x0, y0, w, h)` in workgroups.
        pub fn tile_rects(&self) -> Vec<(u32, u32, u32, u32)> {
            self.tiles.iter().map(|t| (t.x0, t.y0, t.w, t.h)).collect()
        }

        /// `warmup` untimed passes, then `repeats` timed ones; the picture is read back after the last. Each pass is
        /// one submit per tile; the work guard is checked, with that tile's invocation count, before each submit.
        pub fn run(&self, warmup: usize, repeats: usize) -> Result<ImageRun, String> {
            let (wall_ms, gpu_ms) = self.passes(warmup, repeats)?;
            let picture = self.read_back()?;
            Ok(ImageRun { picture, wall_ms, gpu_ms, submits: self.tiles.len() })
        }

        /// The timed passes of [`ImageKernel::run`] without the read-back: `(wall ms, GPU ms)` of each timed pass.
        pub fn passes(&self, warmup: usize, repeats: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
            let wg_inv = u64::from(self.workgroup.0) * u64::from(self.workgroup.1);
            for t in &self.tiles {
                self.fuel.check_under(&format!("an image run of {}", self.name), u64::from(t.w) * u64::from(t.h) * wg_inv, self.cap)?;
            }
            let device = &self.dev.device;
            let queue = &self.dev.queue;
            let n = self.tiles.len() as u32;
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("image timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * n });
            let bytes = 16 * u64::from(n);
            let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image stamps"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let stamps = device.create_buffer(&wgpu::BufferDescriptor { label: Some("image stamps read"), size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let period = f64::from(queue.get_timestamp_period());
            let (mut wall, mut gpu) = (Vec::new(), Vec::new());
            for i in 0..warmup + repeats {
                let t = Instant::now();
                for (c, tile) in self.tiles.iter().enumerate() {
                    let tile_invocations = u64::from(tile.w) * u64::from(tile.h) * wg_inv;
                    self.fuel.check_under(&format!("an image run of {}", self.name), tile_invocations, self.cap)?;
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image pass") });
                    if c == 0 {
                        for b in &self.zero_buffers {
                            if b.usage().contains(wgpu::BufferUsages::STORAGE) {
                                enc.clear_buffer(b, 0, None);
                            }
                        }
                    }
                    {
                        let at = 2 * c as u32;
                        let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(at), end_of_pass_write_index: Some(at + 1) });
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("image"), timestamp_writes });
                        pass.set_pipeline(&self.pipeline);
                        for (g, bg) in &self.bind_groups {
                            pass.set_bind_group(*g, bg, &[]);
                        }
                        pass.set_bind_group(TILE_GROUP, &tile.bind, &[]);
                        pass.dispatch_workgroups(tile.w, tile.h, 1);
                    }
                    if c + 1 == self.tiles.len() {
                        enc.resolve_query_set(&query_set, 0..2 * n, &resolve, 0);
                        enc.copy_buffer_to_buffer(&resolve, 0, &stamps, 0, bytes);
                    }
                    queue.submit(Some(enc.finish()));
                    crate::gpu_io::wait(device)?;
                }
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
                    s.chunks_exact(2).map(|p| p[1].wrapping_sub(p[0]) as f64 * period).sum::<f64>()
                };
                stamps.unmap();
                if i >= warmup {
                    wall.push(ms);
                    gpu.push(ns / 1e6);
                }
            }
            Ok((wall, gpu))
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

    /// An APNG has the PNG signature, one acTL with the frame count, an fcTL per frame, the first frame's IDAT and an
    /// fdAT for each later frame, sequence numbers counting up from 0; frames of different sizes are an error.
    #[test]
    fn an_apng_holds_every_frame_in_sequence() {
        let a = Picture { width: 2, height: 1, rgba: vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] };
        let mut b = a.clone();
        b.rgba[0] = 0.0;
        let bytes = apng_bytes(&[a.clone(), b.clone(), a.clone()], 500).expect("apng");
        assert_eq!(&bytes[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let count = |k: &[u8]| bytes.windows(4).filter(|w| *w == k).count();
        assert_eq!((count(b"acTL"), count(b"fcTL"), count(b"IDAT"), count(b"fdAT")), (1, 3, 1, 2));
        let at = bytes.windows(4).position(|w| w == b"acTL").expect("acTL") + 4;
        assert_eq!(u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]), 3);
        let seqs: Vec<u32> = bytes.windows(4).enumerate().filter(|(_, w)| *w == b"fcTL" || *w == b"fdAT").map(|(i, _)| u32::from_be_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])).collect();
        assert_eq!(seqs, vec![0, 1, 2, 3, 4]);
        assert!(apng_bytes(&[a, Picture { width: 1, height: 1, rgba: vec![0.0; 4] }], 10).is_err());
    }

    /// A stamp draws its glyphs in white inside a black box and clips at the edge; side by side keeps each picture.
    #[test]
    fn a_stamp_writes_white_glyphs_on_black_and_pictures_go_side_by_side() {
        let mut p = Picture { width: 20, height: 8, rgba: vec![0.5; 20 * 8 * 4] };
        stamp(&mut p, "1", 0, 0, 1).expect("stamp");
        let at = |p: &Picture, x: u32, y: u32| p.rgba[((y * p.width + x) * 4) as usize];
        assert_eq!(at(&p, 2, 1), 1.0, "the top of the 1");
        assert_eq!(at(&p, 1, 1), 0.0, "the box");
        assert_eq!(at(&p, 10, 1), 0.5, "outside the box");
        stamp(&mut p, "88888", 15, 5, 2).expect("clipped, not an error");
        let before = p.clone();
        assert!(stamp(&mut p, "?", 0, 0, 1).is_err());
        assert_eq!(p, before, "an error draws nothing");
        let q = Picture { width: 3, height: 8, rgba: vec![0.25; 3 * 8 * 4] };
        let s = side_by_side(&[p.clone(), q]).expect("same height");
        assert_eq!((s.width, s.height), (23, 8));
        assert_eq!(at(&s, 2, 1), 1.0);
        assert_eq!(at(&s, 21, 3), 0.25);
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

    /// The submits of a split dispatch cover every workgroup exactly once, each within `per_submit` workgroups, for a
    /// grid that the split divides and for one it does not.
    #[test]
    fn tiles_cover_every_workgroup_once_within_the_submit_size() {
        for (gx, gy, per) in [(32u32, 32u32, Some(75u64)), (32, 32, Some(5)), (7, 3, Some(1)), (7, 3, Some(14)), (7, 3, Some(1000)), (5, 4, None)] {
            let t = tiles(gx, gy, per);
            let mut seen = vec![0u32; (gx * gy) as usize];
            for (x0, y0, w, h) in &t {
                assert!(per.is_none_or(|p| u64::from(w * h) <= p), "{gx}x{gy} per {per:?}: a tile of {w}x{h}");
                for y in *y0..y0 + h {
                    for x in *x0..x0 + w {
                        seen[(y * gx + x) as usize] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|c| *c == 1), "{gx}x{gy} per {per:?}");
        }
        assert_eq!(tiles(5, 4, None), vec![(0, 0, 5, 4)]);
    }

    #[test]
    fn the_median_of_an_even_and_an_odd_list() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
