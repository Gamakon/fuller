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

/// The display value of a linear colour channel: clamped to [0, 1] and sRGB-encoded, as the screen shows it. The
/// cuneus host draws into an sRGB surface (`Bgra8UnormSrgb`, cuneus `src/renderkit.rs`), so the hardware encodes the
/// shader's stored value on the way to the screen; a linear 0.03 is shown as about 48/255, not 8/255. A non-finite
/// value is returned unchanged (the comparison counts it as a non-finite mismatch). The alpha channel is not encoded.
pub fn srgb_display(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    // In f64, so the ends are exact (1.055 - 0.055 is not 1 in f32).
    let l = f64::from(x.clamp(0.0, 1.0));
    (if l <= 0.003_130_8 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }) as f32
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
    /// A channel is non-finite in one picture and not the same in the other. Such a channel also counts as an
    /// infinite difference in `max_abs`, `mean_abs`, `rmse` (its colour channels) and `pixels_over_one_step`.
    pub non_finite_mismatch: usize,
}

/// The per-pixel error statistics of `got` against `want`, measured in DISPLAY space: each colour channel of both
/// pictures goes through [`srgb_display`] first (identically), so the error is the one a person sees on the cuneus
/// host's sRGB surface; the alpha channel is compared as stored. Before 2026-10-27 the statistics were on the stored
/// (linear) values. A size mismatch is an error.
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
            let (a, b) = if c < 3 { (srgb_display(a), srgb_display(b)) } else { (a, b) };
            if a.to_bits() != b.to_bits() {
                differing += 1;
            }
            // a channel non-finite in one picture and not the same in the other is as wrong as a channel can be: an
            // infinite difference (the fitness's unit clamp makes it 1), never skipped, so a picture of NaN is not a
            // near match (volumepassage run45: champions storing NaN scored rmse 0 with 75 % of channels differing)
            let d = if !a.is_finite() || !b.is_finite() {
                if a.to_bits() == b.to_bits() {
                    continue;
                }
                nonfinite += 1;
                f64::INFINITY
            } else {
                f64::from((a - b).abs())
            };
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
/// to [0, 1], sRGB-encoded as the screen shows it ([`srgb_display`]) and rounded, NaN as 0. The picture's own `rgba`
/// keeps the stored (linear) values.
fn zlib_rgb(p: &Picture) -> Vec<u8> {
    let (w, h) = (p.width as usize, p.height as usize);
    let mut raw = Vec::with_capacity(h * (w * 3 + 1));
    for y in 0..h {
        raw.push(0u8);
        for x in 0..w {
            for c in 0..3 {
                let v = p.rgba[(y * w + x) * 4 + c];
                let v = if v.is_nan() { 0.0 } else { srgb_display(v.clamp(0.0, 1.0)) };
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
/// The sRGB transfer is applied ([`srgb_display`]), as the cuneus host's sRGB surface applies it on the way to the screen.
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

/// The linear value of an sRGB display value in [0, 1] (the inverse of [`srgb_display`] on [0, 1]).
pub fn srgb_linear(v: f32) -> f32 {
    let v = f64::from(v.clamp(0.0, 1.0));
    (if v <= 0.040_45 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }) as f32
}

/// A picture file as a target: a binary PPM (`P6`, 8-bit), its bytes taken as sRGB display values (as a PNG's are)
/// and stored as the linear values a shader would write, so [`compare`] and the PNG writer show the file's own colours.
pub fn read_ppm(path: &std::path::Path) -> Result<Picture, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while i < raw.len() && raw[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < raw.len() && raw[i] == b'#' {
            while i < raw.len() && raw[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < raw.len() && !raw[i].is_ascii_whitespace() {
            i += 1;
        }
        if start == i {
            return Err(format!("{}: the PPM header ends early", path.display()));
        }
        fields.push(String::from_utf8_lossy(&raw[start..i]).to_string());
    }
    i += 1;
    let num = |s: &str| s.parse::<u32>().map_err(|e| format!("{}: the PPM header {s:?}: {e}", path.display()));
    if fields[0] != "P6" || num(&fields[3])? != 255 {
        return Err(format!("{}: not an 8-bit binary PPM ({} {})", path.display(), fields[0], fields[3]));
    }
    let (w, h) = (num(&fields[1])?, num(&fields[2])?);
    let px = raw.get(i..i + (w * h * 3) as usize).ok_or_else(|| format!("{}: the PPM holds fewer than {w}x{h} pixels", path.display()))?;
    let rgba = px.chunks_exact(3).flat_map(|c| [srgb_linear(f32::from(c[0]) / 255.0), srgb_linear(f32::from(c[1]) / 255.0), srgb_linear(f32::from(c[2]) / 255.0), 1.0]).collect();
    Ok(Picture { width: w, height: h, rgba })
}

/// The share of the picture's pixels in its most common colour, the display colour quantised to `levels` steps a
/// channel (4 steps: 64 bins), and the number of bins used.
pub fn largest_colour_share(p: &Picture, levels: u32) -> (f64, usize) {
    let mut bins: std::collections::HashMap<(u32, u32, u32), usize> = std::collections::HashMap::new();
    let q = |v: f32| ((srgb_display(v).clamp(0.0, 1.0) * levels as f32) as u32).min(levels - 1);
    for c in p.rgba.chunks_exact(4) {
        *bins.entry((q(c[0]), q(c[1]), q(c[2]))).or_default() += 1;
    }
    let n = (p.width * p.height).max(1) as f64;
    (bins.values().copied().max().unwrap_or(0) as f64 / n, bins.len())
}

/// The picture-quality inputs of a fitness (one HFF group), each in [0, 1], 0 best, all in display space: the
/// differing channel fraction; max |d|, mean |d| and rmse, each divided by the original's own max, mean and rms (so
/// drawing nothing is an error of 1) and mapped e -> e / (1 + e) (drawing nothing scores 0.5, a picture far off the
/// original's scale approaches 1, and there is a gradient at every magnitude); the fraction of pixels over 1/255;
/// the fraction of non-finite mismatches.
pub fn quality_inputs(got: &Picture, want: &Picture) -> Result<([f64; 6], Difference), String> {
    let d = compare(got, want)?;
    let c: Vec<f64> = want.rgba.chunks_exact(4).flat_map(|px| px[..3].iter().map(|v| f64::from(srgb_display(*v)))).filter(|v| v.is_finite()).collect();
    let n = c.len().max(1) as f64;
    let o_max = c.iter().cloned().fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    let o_mean = (c.iter().map(|v| v.abs()).sum::<f64>() / n).max(f64::MIN_POSITIVE);
    let o_rms = (c.iter().map(|v| v * v).sum::<f64>() / n).sqrt().max(f64::MIN_POSITIVE);
    let rel = |e: f64| if e.is_finite() { e / (1.0 + e) } else { 1.0 };
    let channels = (d.pixels * 4) as f64;
    let q = [d.differing_channels as f64 / channels, rel(d.max_abs / o_max), rel(d.mean_abs / o_mean), rel(d.rmse / o_rms), d.pixels_over_one_step as f64 / d.pixels as f64, d.non_finite_mismatch as f64 / channels];
    Ok((q, d))
}

/// The picture enlarged by the smallest whole factor that makes it at least `min_width` pixels wide (each pixel a
/// square block), for a reader such as OCR that needs letters a few dozen pixels tall.
pub fn upscaled(p: &Picture, min_width: u32) -> Picture {
    let f = min_width.div_ceil(p.width.max(1)).max(1);
    let (w, h) = (p.width * f, p.height * f);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let at = (((y / f) * p.width + x / f) * 4) as usize;
            rgba.extend_from_slice(&p.rgba[at..at + 4]);
        }
    }
    Picture { width: w, height: h, rgba }
}

/// What OCR read in a picture: the words, tesseract's mean word confidence (0 to 100; -1 when it read no word),
/// whether the letters equal the word wanted (case-insensitive, anything but letters and digits ignored), and the
/// edit distance (Levenshtein, on those letters) from the word wanted.
#[derive(Debug, Clone, PartialEq)]
pub struct Ocr {
    pub text: String,
    pub confidence: f64,
    pub matches: bool,
    pub distance: usize,
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Read the text of a PNG with `tesseract` (`psm` its page segmentation: 7 one line, 8 one word) and compare it with
/// `want`. A report measurement, not a fitness input.
pub fn ocr(png: &std::path::Path, want: &str, psm: u32) -> Result<Ocr, String> {
    let out = std::process::Command::new("tesseract")
        .arg(png)
        .arg("-")
        .args(["--psm", &psm.to_string(), "tsv"])
        .output()
        .map_err(|e| format!("tesseract: {e}"))?;
    if !out.status.success() {
        return Err(format!("tesseract failed on {}: {}", png.display(), String::from_utf8_lossy(&out.stderr)));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut words = Vec::new();
    let mut confs = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() == 12 && f[0] == "5" {
            let w = f[11].trim();
            if !w.is_empty() {
                words.push(w.to_string());
                confs.push(f[10].parse::<f64>().map_err(|e| format!("a tesseract confidence {:?}: {e}", f[10]))?);
            }
        }
    }
    let joined = words.join(" ");
    let letters: Vec<char> = joined.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_uppercase().chars().collect();
    let wanted: Vec<char> = want.to_uppercase().chars().collect();
    let confidence = if confs.is_empty() { -1.0 } else { confs.iter().sum::<f64>() / confs.len() as f64 };
    Ok(Ocr { text: joined, confidence, matches: letters == wanted, distance: levenshtein(&letters, &wanted) })
}

/// The picture prepared for OCR of bright text: the display grey (`0.299 R + 0.587 G + 0.114 B` of the sRGB
/// values) thresholded at `threshold` (bright = ink, drawn black on white), cropped to the ink's bounding box with a
/// margin of a tenth of the picture's height, and enlarged to at least `min_width` pixels. `None` when no pixel is
/// that bright.
pub fn ocr_prepared(p: &Picture, threshold: f32, min_width: u32) -> Option<Picture> {
    let grey = |i: usize| {
        let c = |k: usize| srgb_display(p.rgba[i * 4 + k]);
        0.299 * c(0) + 0.587 * c(1) + 0.114 * c(2)
    };
    let (w, h) = (p.width as usize, p.height as usize);
    let ink: Vec<bool> = (0..w * h).map(|i| grey(i) > threshold).collect();
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if ink[y * w + x] {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
    }
    if x0 == usize::MAX {
        return None;
    }
    let m = (h / 10).max(2);
    let (cx0, cy0) = (x0.saturating_sub(m), y0.saturating_sub(m));
    let (cx1, cy1) = ((x1 + m).min(w - 1), (y1 + m).min(h - 1));
    let (cw, ch) = (cx1 - cx0 + 1, cy1 - cy0 + 1);
    let mut rgba = Vec::with_capacity(cw * ch * 4);
    for y in cy0..=cy1 {
        for x in cx0..=cx1 {
            // Linear 0 and 1 are display 0 and 1, so the PNG holds pure black ink on white.
            let v = if ink[y * w + x] { 0.0 } else { 1.0 };
            rgba.extend_from_slice(&[v, v, v, 1.0]);
        }
    }
    Some(upscaled(&Picture { width: cw as u32, height: ch as u32, rgba }, min_width))
}

/// The OCR reading of a picture for the report: prepared ([`ocr_prepared`], threshold 200/255, at least 1024 wide),
/// written to `png`, and read as one word. A picture with no bright pixel reads as nothing.
pub fn ocr_word(p: &Picture, png: &std::path::Path, want: &str) -> Result<Ocr, String> {
    match ocr_prepared(p, 200.0 / 255.0, 1024) {
        Some(q) => {
            write_png(png, &q)?;
            ocr(png, want, 8)
        }
        None => Ok(Ocr { text: String::new(), confidence: -1.0, matches: false, distance: want.len() }),
    }
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

/// The image interfaces' pipeline-overridable constant (`override DISPATCH_TILED: bool = true;`, the prologue's, never a
/// row's): the harness sets it when it creates a pipeline, false when a pass is one submit (the entry then reads the
/// builtins and never the tile uniform, so Apple's compiler folds the offset away), true for a pass of several submits.
pub const TILED_OVERRIDE: &str = "DISPATCH_TILED";

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

/// A submit rectangle `(x0, y0, w, h)`: in workgroups for [`tiles`], in pixels for a [`SubmitPlan`].
pub type Rect = (u32, u32, u32, u32);

/// THE SUBMIT PLAN OF A PASS: each submit's rectangle in pixels, clipped to the picture. Two kernels with one plan launch
/// the same submits over the same pixels whatever their workgroup shapes, so a timed pair under one plan compares the
/// programs, never the splitting (ST9: a candidate timed in 16 submits against an original in one measured the split).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmitPlan {
    pub rects: Vec<Rect>,
}

impl SubmitPlan {
    /// The plan of workgroup rectangles `tiles` of a `workgroup` dispatch over `width` x `height`.
    pub fn of_tiles(tiles: &[Rect], workgroup: (u32, u32), width: u32, height: u32) -> SubmitPlan {
        let (wx, wy) = (workgroup.0.max(1), workgroup.1.max(1));
        let rects = tiles
            .iter()
            .map(|&(x0, y0, w, h)| {
                let (px, py) = ((x0 * wx).min(width), (y0 * wy).min(height));
                (px, py, ((x0 + w) * wx).min(width) - px, ((y0 + h) * wy).min(height) - py)
            })
            .collect();
        SubmitPlan { rects }
    }

    /// One submit over the whole picture.
    pub fn single(width: u32, height: u32) -> SubmitPlan {
        SubmitPlan { rects: vec![(0, 0, width, height)] }
    }

    pub fn submits(&self) -> usize {
        self.rects.len()
    }

    /// The workgroup rectangles of a `workgroup` dispatch that launch this plan: each pixel rectangle's workgroups, from
    /// the one holding its first pixel to the one holding its last. With the workgroup the plan was cut by this is the
    /// original tiling; with another workgroup shape it is the same submits over the same pixels (a rectangle edge that is
    /// not on that shape's grid is covered by the workgroup holding it, see [`SubmitPlan::aligned`]).
    pub fn tiles(&self, workgroup: (u32, u32)) -> Vec<Rect> {
        let (wx, wy) = (workgroup.0.max(1), workgroup.1.max(1));
        self.rects.iter().map(|&(x, y, w, h)| (x / wx, y / wy, (x + w).div_ceil(wx) - x / wx, (y + h).div_ceil(wy) - y / wy)).collect()
    }

    /// Whether this plan (a reference's, dispatched by [`SubmitPlan::tiles`] of `other`) launches `other`'s submits: the
    /// same count, in order, each of its rectangles holding the matching one of `other` (equal when both workgroup shapes
    /// are on the plan's grid, larger by the edge workgroups otherwise).
    pub fn launches(&self, other: &SubmitPlan) -> bool {
        self.rects.len() == other.rects.len() && self.rects.iter().zip(&other.rects).all(|(a, b)| a.0 <= b.0 && a.1 <= b.1 && a.0 + a.2 >= b.0 + b.2 && a.1 + a.3 >= b.1 + b.3)
    }

    /// Whether every rectangle of the plan is on the grid of `workgroup` (or ends at the picture's edge), so a dispatch of
    /// that shape covers exactly the plan's pixels.
    pub fn aligned(&self, workgroup: (u32, u32), width: u32, height: u32) -> bool {
        let (wx, wy) = (workgroup.0.max(1), workgroup.1.max(1));
        self.rects.iter().all(|&(x, y, w, h)| x % wx == 0 && y % wy == 0 && ((x + w) % wx == 0 || x + w == width) && ((y + h) % wy == 0 || y + h == height))
    }
}

/// THE TIMING SIZE: the largest power-of-two square side at most `size` at which a kernel of `fuel` and `workgroup` runs a
/// whole pass in ONE submit under `cap` (the machine's cap at most), so a timed pass is the setting the shipped shader runs
/// in. None when not even a 1 x 1 picture is one submit. A proven kernel is one submit at every size: `size` itself when
/// it is a power of two, else the power of two below it.
pub fn single_submit_size(fuel: &crate::wgsl::work_guard::Fuel, workgroup: (u32, u32), size: u32, cap: u128) -> Result<Option<u32>, String> {
    let wg = u64::from(workgroup.0.max(1)) * u64::from(workgroup.1.max(1));
    let mut s = if size == 0 { return Ok(None) } else { 1u32 << (31 - size.leading_zeros()) };
    loop {
        let (gx, gy) = (s.div_ceil(workgroup.0.max(1)), s.div_ceil(workgroup.1.max(1)));
        let fits = match fuel.workgroups_per_submit(wg, cap) {
            Ok(per) => tiles(gx, gy, per).len() == 1,
            Err(_) => false,
        };
        if fits {
            return Ok(Some(s));
        }
        if s == 1 {
            return Ok(None);
        }
        s /= 2;
    }
}

/// One dispatch in a timed pair's command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairStep {
    /// The reference's warm-up dispatch of `submit`, so the GPU is clocked up before the timed two. Its compute pass
    /// writes timestamps `query` (begin) and `query + 1` (end), read only to prove the passes ran one after another.
    WarmReference { submit: usize, query: u32 },
    /// A timed dispatch of `submit` (the candidate's or the reference's), its compute pass writing timestamps
    /// `query` (begin) and `query + 1` (end).
    Timed { candidate: bool, submit: usize, query: u32 },
}

impl PairStep {
    /// The begin timestamp of this step's compute pass (its end is the next query).
    pub fn query(&self) -> u32 {
        match *self {
            PairStep::WarmReference { query, .. } | PairStep::Timed { query, .. } => query,
        }
    }
}

/// The reference's warm-up dispatches that lead each command buffer of a timed pair.
pub const PAIR_WARM_DISPATCHES: usize = 4;

/// Timestamp queries of one submit of a timed pair: a begin and an end for every compute pass of its command buffer.
pub const PAIR_QUERIES_PER_SUBMIT: u32 = 2 * (PAIR_WARM_DISPATCHES as u32 + 2);

/// THE COMMAND BUFFERS OF ONE TIMED PAIR over a plan of `submits` submits: one command buffer a submit, holding the
/// reference's warm-ups of that submit, then the candidate's and the reference's timed dispatches of it back to back
/// (`candidate_first` picks the order). With `b = PAIR_QUERIES_PER_SUBMIT * c`, warm-up `i` writes timestamps
/// `b + 2i, b + 2i + 1`, the candidate `b + 2W, b + 2W + 1` and the reference `b + 2W + 2, b + 2W + 3` (`W` the warm-ups).
pub fn pair_schedule(submits: usize, candidate_first: bool) -> Vec<Vec<PairStep>> {
    (0..submits)
        .map(|c| {
            let base = PAIR_QUERIES_PER_SUBMIT * c as u32;
            let warm = 2 * PAIR_WARM_DISPATCHES as u32;
            let cand = PairStep::Timed { candidate: true, submit: c, query: base + warm };
            let refr = PairStep::Timed { candidate: false, submit: c, query: base + warm + 2 };
            let (a, b) = if candidate_first { (cand, refr) } else { (refr, cand) };
            let mut v: Vec<PairStep> = (0..PAIR_WARM_DISPATCHES as u32).map(|i| PairStep::WarmReference { submit: c, query: base + 2 * i }).collect();
            v.extend([a, b]);
            v
        })
        .collect()
}

/// THE FENCE KERNEL of a timed pair's command buffer (ST12): one invocation that reads and writes a word of the timing
/// harness's own buffer. It is the first dispatch of every compute pass in the command buffer, so every pass writes that
/// buffer, and Metal starts a pass only after the pass before it that wrote it has ended, whatever the timed kernel
/// itself touches.
pub const FENCE_WGSL: &str = "@group(0) @binding(0) var<storage, read_write> fence: array<u32, 4>;\n@compute @workgroup_size(1, 1, 1)\nfn main() {\n    fence[0] = fence[0] + 1u;\n}\n";

#[cfg(feature = "gpu")]
pub use device::{timed_pairs, ImageKernel, ImageRun, PairTimes};

#[cfg(feature = "gpu")]
mod device {
    use super::{half_to_f32, median, tiles, BindingData, Picture, SubmitPlan, TILED_OVERRIDE, TILE_GROUP};
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
        /// The module and layout the pipeline was made from, so a plan of the other kind (one submit, several) makes the
        /// pipeline with the other value of [`TILED_OVERRIDE`].
        shader: wgpu::ShaderModule,
        pipeline_layout: wgpu::PipelineLayout,
        entry: String,
        /// Whether the module declares [`TILED_OVERRIDE`].
        tiled_override: bool,
        /// The value of [`TILED_OVERRIDE`] the pipeline was made with (false when the module does not declare it).
        pipeline_tiled: bool,
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
        /// Wall milliseconds, first submit to the last tile's completion, of each timed pass.
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

    /// `plan` (None: the kernel's own split, as many whole workgroups a submit as its bound allows under the cap) is a
    /// [`SubmitPlan`] imposed on the dispatch: its tiles for this workgroup shape, each checked under the guard here, before
    /// anything is encoded, and again before its submit.
    fn target(dev: &Device, layouts: &[wgpu::BindGroupLayout], bindings: &[BindingData], width: u32, height: u32, workgroup: (u32, u32), fuel: &Fuel, cap: u128, plan: Option<&SubmitPlan>) -> Result<Target, String> {
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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
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
        let rects = match plan {
            None => tiles(gx, gy, fuel.workgroups_per_submit(u64::from(workgroup.0) * u64::from(workgroup.1), cap)?),
            Some(p) => {
                let t = p.tiles(workgroup);
                let wg_inv = u64::from(workgroup.0) * u64::from(workgroup.1);
                for (x0, y0, w, h) in &t {
                    if x0 + w > gx || y0 + h > gy {
                        return Err(format!("our software has a bug: a submit plan rectangle of workgroups ({x0}, {y0}, {w}, {h}) outside the {gx} x {gy} grid"));
                    }
                    fuel.check_under("a submit of the imposed plan", u64::from(*w) * u64::from(*h) * wg_inv, cap)?;
                }
                t
            }
        };
        let mut tile_list = Vec::new();
        for (x0, y0, w, h) in rects {
            let words = [x0, y0, 0, 0, gx, gy, 1, 0];
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dispatch tile"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            dev.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&words));
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dispatch tile"), layout: &layouts[TILE_GROUP as usize], entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
            tile_list.push(Tile { x0, y0, w, h, bind, _uniform: buf });
        }
        Ok(Target { bind_groups, uniforms, texture, tiles: tile_list, zero_buffers })
    }

    /// The compute pipeline of `shader`'s `entry`, with [`TILED_OVERRIDE`] set to `tiled` when the module declares it.
    fn make_pipeline(dev: &Device, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, entry: &str, tiled_override: bool, tiled: bool) -> Result<wgpu::ComputePipeline, String> {
        let constants: &[(&str, f64)] = match (tiled_override, tiled) {
            (false, _) => &[],
            (true, false) => &[(TILED_OVERRIDE, 0.0)],
            (true, true) => &[(TILED_OVERRIDE, 1.0)],
        };
        let scopes = ErrorScopes::push(&dev.device);
        let compilation_options = wgpu::PipelineCompilationOptions { constants, ..Default::default() };
        let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("image pipeline"), layout: Some(layout), module: shader, entry_point: Some(entry), compilation_options, cache: None });
        if let Some(e) = scopes.pop() {
            return Err(format!("pipeline: {e}"));
        }
        Ok(pipeline)
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
            let tiled_override = module.overrides.iter().any(|(_, o)| o.name.as_deref() == Some(TILED_OVERRIDE));
            let t = target(dev, &layouts, bindings, width, height, workgroup, &fuel, cap, None)?;
            let scopes = ErrorScopes::push(device);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("image kernel"), source: wgpu::ShaderSource::Wgsl(source.into()) });
            let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("image layout"), bind_group_layouts: &layout_refs, immediate_size: 0 });
            if let Some(e) = scopes.pop() {
                return Err(format!("shader: {e}"));
            }
            let pipeline_tiled = tiled_override && t.tiles.len() > 1;
            let pipeline = make_pipeline(dev, &shader, &pipeline_layout, entry, tiled_override, pipeline_tiled)?;
            Ok(ImageKernel {
                dev,
                pipeline,
                shader,
                pipeline_layout,
                entry: entry.to_string(),
                tiled_override,
                pipeline_tiled,
                layouts,
                bind_groups: t.bind_groups,
                uniforms: t.uniforms,
                texture: t.texture,
                width,
                height,
                workgroup,
                fuel,
                cap,
                tiles: t.tiles,
                name: name.to_string(),
                zero_buffers: t.zero_buffers,
            })
        }

        /// This kernel's compiled program over `t` (another size, bindings or plan): the same pipeline, or, when `t`'s
        /// submits are of the other kind (one, several) and the module declares [`TILED_OVERRIDE`], the pipeline made
        /// with the other value of it.
        fn over(&self, t: Target, width: u32, height: u32) -> Result<ImageKernel<'a>, String> {
            let tiled = self.tiled_override && t.tiles.len() > 1;
            let pipeline = if tiled == self.pipeline_tiled { self.pipeline.clone() } else { make_pipeline(self.dev, &self.shader, &self.pipeline_layout, &self.entry, self.tiled_override, tiled)? };
            Ok(ImageKernel {
                dev: self.dev,
                pipeline,
                shader: self.shader.clone(),
                pipeline_layout: self.pipeline_layout.clone(),
                entry: self.entry.clone(),
                tiled_override: self.tiled_override,
                pipeline_tiled: tiled,
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

        /// Whether the pipeline reads the tile uniform: the module does not declare [`TILED_OVERRIDE`] (the uniform is
        /// always read), or it does and the pass takes several submits.
        pub fn reads_tile_uniform(&self) -> bool {
            !self.tiled_override || self.pipeline_tiled
        }

        /// The same compiled pipeline drawing at another size with another case's bindings (no recompile).
        pub fn resized(&self, bindings: &[BindingData], width: u32, height: u32) -> Result<ImageKernel<'a>, String> {
            let t = target(self.dev, &self.layouts, bindings, width, height, self.workgroup, &self.fuel, self.cap, None)?;
            self.over(t, width, height)
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

        /// The kernel's workgroup shape `(x, y)`.
        pub fn workgroup(&self) -> (u32, u32) {
            self.workgroup
        }

        /// The picture size `(width, height)`.
        pub fn size(&self) -> (u32, u32) {
            (self.width, self.height)
        }

        /// The submits of one pass as pixel rectangles.
        pub fn plan(&self) -> SubmitPlan {
            SubmitPlan::of_tiles(&self.tile_rects(), self.workgroup, self.width, self.height)
        }

        /// The same compiled pipeline at its own size with the case's bindings, dispatched by `plan` (the submits and the
        /// pixels each covers) instead of its own split: the reference of a timed pair takes the candidate's plan. Every
        /// submit of the plan is checked under the work guard before anything is encoded.
        pub fn with_plan(&self, bindings: &[BindingData], plan: &SubmitPlan) -> Result<ImageKernel<'a>, String> {
            let t = target(self.dev, &self.layouts, bindings, self.width, self.height, self.workgroup, &self.fuel, self.cap, Some(plan))?;
            self.over(t, self.width, self.height)
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
            // The timestamp queries of a pass (two per submit), one query set for this call (Metal holds few counter
            // sample buffers at once: one per kernel exhausted them at 32). They are resolved by a command buffer of
            // their own after the pass's last submit has completed: resolved in the pass's own command buffer they read
            // the values of the pass before (0 ns for the first) on this machine (M3 Max, Metal).
            let bytes = 16 * u64::from(n);
            let query_set = &device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("image timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * n });
            let resolve = &device.create_buffer(&wgpu::BufferDescriptor { label: Some("image stamps"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let stamps = &device.create_buffer(&wgpu::BufferDescriptor { label: Some("image stamps read"), size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let period = f64::from(queue.get_timestamp_period());
            let (mut wall, mut gpu) = (Vec::new(), Vec::new());
            for i in 0..warmup + repeats {
                let t = Instant::now();
                for (c, tile) in self.tiles.iter().enumerate() {
                    let tile_invocations = u64::from(tile.w) * u64::from(tile.h) * wg_inv;
                    self.fuel.check_under(&format!("an image run of {}", self.name), tile_invocations, self.cap)?;
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image pass") });
                    self.encode_submit(&mut enc, c, Some((query_set, 2 * c as u32)), None);
                    queue.submit(Some(enc.finish()));
                    crate::gpu_io::wait(device)?;
                }
                let ms_pass = t.elapsed().as_secs_f64() * 1e3;
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image timestamps") });
                enc.resolve_query_set(query_set, 0..2 * n, resolve, 0);
                enc.copy_buffer_to_buffer(resolve, 0, stamps, 0, bytes);
                queue.submit(Some(enc.finish()));
                let (tx, rx) = std::sync::mpsc::channel();
                stamps.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).expect("the receiver waits for the mapping");
                });
                crate::gpu_io::wait(device)?;
                let ms = ms_pass;
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

        /// Encode submit `c` of a pass into `enc`: the zero storage cleared first when `c` is the pass's first submit,
        /// then one compute pass dispatching tile `c`, its begin and end written to `stamps` at `(query, query + 1)`;
        /// with a `fence`, the pass's first dispatch is the fence kernel's (see [`super::FENCE_WGSL`]).
        fn encode_submit(&self, enc: &mut wgpu::CommandEncoder, c: usize, stamps: Option<(&wgpu::QuerySet, u32)>, fence: Option<&Fence>) {
            if c == 0 {
                for b in &self.zero_buffers {
                    if b.usage().contains(wgpu::BufferUsages::STORAGE) {
                        enc.clear_buffer(b, 0, None);
                    }
                }
            }
            let tile = &self.tiles[c];
            let timestamp_writes = stamps.map(|(query_set, at)| wgpu::ComputePassTimestampWrites { query_set, beginning_of_pass_write_index: Some(at), end_of_pass_write_index: Some(at + 1) });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("image"), timestamp_writes });
            if let Some(f) = fence {
                pass.set_pipeline(&f.pipeline);
                pass.set_bind_group(0, &f.bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            pass.set_pipeline(&self.pipeline);
            for (g, bg) in &self.bind_groups {
                pass.set_bind_group(*g, bg, &[]);
            }
            pass.set_bind_group(TILE_GROUP, &tile.bind, &[]);
            pass.dispatch_workgroups(tile.w, tile.h, 1);
        }

        /// The work-guard bound of submit `c` (0 for a proven kernel), checked under the kernel's cap.
        fn submit_bound(&self, c: usize) -> Result<u128, String> {
            let t = &self.tiles[c];
            let wg_inv = u64::from(self.workgroup.0) * u64::from(self.workgroup.1);
            self.fuel.check_under(&format!("an image run of {}", self.name), u64::from(t.w) * u64::from(t.h) * wg_inv, self.cap)
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

    /// The fence kernel's pipeline and its buffer (see [`super::FENCE_WGSL`]).
    struct Fence {
        pipeline: wgpu::ComputePipeline,
        bind: wgpu::BindGroup,
        _buffer: wgpu::Buffer,
    }

    impl Fence {
        fn new(dev: &Device) -> Result<Fence, String> {
            let device = &dev.device;
            let scopes = ErrorScopes::push(device);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pair fence"), source: wgpu::ShaderSource::Wgsl(super::FENCE_WGSL.into()) });
            let entry = wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("pair fence"), entries: &[entry] });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pair fence"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("pair fence"), layout: Some(&layout), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pair fence"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("pair fence"), layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }] });
            if let Some(e) = scopes.pop() {
                return Err(format!("our software has a bug: the pair fence kernel: {e}"));
            }
            Ok(Fence { pipeline, bind, _buffer: buffer })
        }
    }

    /// The timed pairs of a candidate against a reference and the plan they shared.
    #[derive(Debug, Clone)]
    pub struct PairTimes {
        /// `(candidate GPU ms, reference GPU ms)` of each pair (each the sum over the plan's submits).
        pub times: Vec<(f64, f64)>,
        /// Whether the candidate's pass came first in pair `i` (alternating, the first pair candidate first).
        pub candidate_first: Vec<bool>,
        /// The command buffer of every timestamp of pair `i`, submit by submit: `[candidate begin, candidate end,
        /// reference begin, reference end]` as the index of the pair's command buffer that wrote it.
        pub stamp_buffers: Vec<Vec<[usize; 4]>>,
        /// Submits a pass took, the same for both.
        pub submits: usize,
        /// Whether the plan's rectangles are on both kernels' workgroup grids (so both cover exactly the plan's pixels).
        pub aligned: bool,
    }

    /// THE TIMED PAIRS (ST9, the clock-state fix): `warmup` untimed passes of each, then `pairs` pairs. A pair is ONE
    /// command buffer a submit of the plan (one at the timing size): an untimed warm-up dispatch of the reference, then
    /// the two timed dispatches back to back, the candidate first in even pairs and the reference first in odd ones, each
    /// timed by the timestamps of its own compute pass. Both times of a pair are taken in one clock state, so their ratio
    /// measures the programs and not the GPU's clock (2026-10-22: passes timed in submits of their own read 0.05 to 0.18
    /// ms with the clock state, and HFF rewarded candidates whose noisy ratios happened to agree).
    ///
    /// The two kernels must have ONE submit plan (the same submits over the same pixels; the caller gives the reference
    /// the candidate's plan with [`ImageKernel::with_plan`]): a pair under two plans would time the splitting, so it is
    /// our software's bug and nothing is timed. A command buffer holds the warm-up and both passes of one submit, so the
    /// sum of their work-guard bounds (a proven kernel's is 0) must be under the cap before anything is encoded.
    ///
    /// THE ORDER (ST12): Metal runs compute passes of one command buffer that share no resource at the same time, and a
    /// kernel shares only what its compiled program reads or writes: a kernel whose stores the compiler removes shares
    /// nothing, so a one-texel copy between the pictures (to 2026-10-22) did not order it (measured: 24 kernels of the
    /// planted-win run, 70 of 70 replays, and an empty kernel). Every compute pass of the command buffer, warm-ups
    /// included, therefore begins with the fence kernel's dispatch, which reads and writes the harness's own buffer: each
    /// pass waits for the pass before it. Every pass writes its begin and end timestamps, and a pass that begins before
    /// the pass before it in its command buffer ends is our software's bug (the two overlapped and neither time would be a
    /// kernel's), so the order is proved on every pair, never assumed.
    pub fn timed_pairs(candidate: &ImageKernel<'_>, reference: &ImageKernel<'_>, warmup: usize, pairs: usize) -> Result<PairTimes, String> {
        let (cp, rp) = (candidate.plan(), reference.plan());
        if !rp.launches(&cp) || candidate.size() != reference.size() {
            return Err(format!("our software has a bug: a timed pair with two submit plans: candidate {:?} at {:?}, reference {:?} at {:?}", cp.rects, candidate.size(), rp.rects, reference.size()));
        }
        if !std::ptr::eq(candidate.dev, reference.dev) {
            return Err("our software has a bug: a timed pair of kernels on two devices cannot share a command buffer".into());
        }
        let aligned = cp == rp;
        let n = cp.submits();
        let cap = candidate.cap.min(reference.cap).min(CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT);
        for c in 0..n {
            let bound = candidate.submit_bound(c)?.saturating_add(reference.submit_bound(c)?.saturating_mul(2));
            if bound > cap {
                return Err(format!("our software has a bug: submit {c} of a timed pair ({} once, {} twice) is bounded by {bound} node executions in one command buffer, over the cap {cap}; nothing was submitted", candidate.name, reference.name));
            }
        }
        if warmup > 0 {
            reference.passes(warmup, 0)?;
            candidate.passes(warmup, 0)?;
        }
        let device = &candidate.dev.device;
        let queue = &candidate.dev.queue;
        let fence = Fence::new(candidate.dev)?;
        let queries = super::PAIR_QUERIES_PER_SUBMIT * n as u32;
        let bytes = 8 * u64::from(queries);
        let query_set = &device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("pair timestamps"), ty: wgpu::QueryType::Timestamp, count: queries });
        let resolve = &device.create_buffer(&wgpu::BufferDescriptor { label: Some("pair stamps"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stamps = &device.create_buffer(&wgpu::BufferDescriptor { label: Some("pair stamps read"), size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let period = f64::from(queue.get_timestamp_period());
        let (mut times, mut firsts, mut buffers) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs), Vec::with_capacity(pairs));
        for p in 0..pairs {
            let schedule = super::pair_schedule(n, p % 2 == 0);
            let mut written = vec![usize::MAX; queries as usize];
            for (b, steps) in schedule.iter().enumerate() {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("timed pair") });
                for step in steps {
                    let (k, submit) = match *step {
                        super::PairStep::WarmReference { submit, .. } => (reference, submit),
                        super::PairStep::Timed { candidate: is_c, submit, .. } => (if is_c { candidate } else { reference }, submit),
                    };
                    let query = step.query();
                    written[query as usize] = b;
                    written[query as usize + 1] = b;
                    k.encode_submit(&mut enc, submit, Some((query_set, query)), Some(&fence));
                }
                queue.submit(Some(enc.finish()));
                crate::gpu_io::wait(device)?;
            }
            // resolved by a command buffer of their own after the pair's last submit has completed (see `passes`)
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pair timestamps") });
            enc.resolve_query_set(query_set, 0..queries, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, stamps, 0, bytes);
            queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            stamps.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).expect("the receiver waits for the mapping");
            });
            crate::gpu_io::wait(device)?;
            rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
            let s: Vec<u64> = {
                let mapped = crate::gpu_io::mapped(&stamps.slice(..))?;
                bytemuck::cast_slice(&mapped).to_vec()
            };
            stamps.unmap();
            let (mut c_ns, mut r_ns, mut pair_buffers) = (0f64, 0f64, Vec::with_capacity(n));
            for (c, steps) in schedule.iter().enumerate() {
                for (i, w) in steps.windows(2).enumerate() {
                    let (first, second) = (w[0].query() as usize, w[1].query() as usize);
                    if s[second] < s[first + 1] {
                        return Err(format!("our software has a bug: in pair {p} submit {c} pass {} ({:?}) began at tick {} before pass {i} ({:?}) ended at tick {}: the two overlapped", i + 1, w[1], s[second], w[0], s[first + 1]));
                    }
                }
                let timed: Vec<(bool, u32)> = steps.iter().filter_map(|st| if let super::PairStep::Timed { candidate, query, .. } = *st { Some((candidate, query)) } else { None }).collect();
                let mut idx = [0usize; 4];
                for (is_c, q) in timed {
                    let ns = s[q as usize + 1].wrapping_sub(s[q as usize]) as f64 * period;
                    let at = if is_c { 0 } else { 2 };
                    idx[at] = written[q as usize];
                    idx[at + 1] = written[q as usize + 1];
                    if is_c {
                        c_ns += ns;
                    } else {
                        r_ns += ns;
                    }
                }
                pair_buffers.push(idx);
            }
            times.push((c_ns / 1e6, r_ns / 1e6));
            firsts.push(p % 2 == 0);
            buffers.push(pair_buffers);
        }
        Ok(PairTimes { times, candidate_first: firsts, stamp_buffers: buffers, submits: n, aligned })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planted picture: a linear 0.03 is shown, and saved, as about 48/255 (the sRGB transfer), not 8/255; the
    /// encoding is clamped, monotone, exact at 0 and 1, and leaves a NaN to the comparison.
    #[test]
    fn a_dark_linear_value_is_saved_and_compared_as_the_screen_shows_it() {
        assert_eq!((srgb_display(0.03) * 255.0 + 0.5) as u8, 48);
        assert_eq!((srgb_display(0.0), srgb_display(1.0), srgb_display(-0.5), srgb_display(7.0)), (0.0, 1.0, 0.0, 1.0));
        assert!(srgb_display(f32::NAN).is_nan());
        let mut last = 0.0;
        for i in 1..=1000 {
            let v = srgb_display(i as f32 / 1000.0);
            assert!(v > last, "not increasing at {i}");
            last = v;
        }
        let p = Picture { width: 1, height: 1, rgba: vec![0.03, 0.03, 0.03, 1.0] };
        let z = zlib_rgb(&p);
        // zlib header (2), stored-block header (5), the filter byte, then the three channels.
        assert_eq!(&z[8..11], &[48, 48, 48]);
        let black = Picture { width: 1, height: 1, rgba: vec![0.0, 0.0, 0.0, 1.0] };
        let d = compare(&black, &p).expect("same size");
        assert!((d.max_abs - f64::from(srgb_display(0.03))).abs() < 1e-9 && d.max_abs > 0.18, "{d:?}");
    }

    #[test]
    fn the_ocr_helpers_measure_edit_distance_and_crop_the_ink() {
        let c = |s: &str| s.chars().collect::<Vec<_>>();
        assert_eq!(levenshtein(&c("CUNELS"), &c("CUNEUS")), 1);
        assert_eq!(levenshtein(&c(""), &c("CUNEUS")), 6);
        assert_eq!(levenshtein(&c("CUNEUS"), &c("CUNEUS")), 0);
        let mut p = Picture { width: 40, height: 20, rgba: vec![0.1; 40 * 20 * 4] };
        for x in 10..20 {
            for k in 0..3 {
                p.rgba[(10 * 40 + x) * 4 + k] = 1.0;
            }
        }
        let q = ocr_prepared(&p, 200.0 / 255.0, 100).expect("ink");
        // A 10 x 1 stroke with a margin of 2: 14 x 5, enlarged by 8 to 112 x 40.
        assert_eq!((q.width, q.height), (112, 40));
        assert!(ocr_prepared(&Picture { width: 4, height: 4, rgba: vec![0.0; 64] }, 0.5, 10).is_none());
    }

    #[test]
    fn a_ppm_target_reads_back_as_its_own_display_colours() {
        let dir = std::env::temp_dir().join(format!("ppm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.ppm");
        let mut f = b"P6\n# a comment\n2 1\n255\n".to_vec();
        f.extend_from_slice(&[0, 48, 255, 128, 128, 128]);
        std::fs::write(&path, f).unwrap();
        let p = read_ppm(&path).unwrap();
        assert_eq!((p.width, p.height), (2, 1));
        let shown: Vec<u8> = p.rgba.iter().enumerate().filter(|(i, _)| i % 4 != 3).map(|(_, v)| (srgb_display(*v) * 255.0 + 0.5) as u8).collect();
        assert_eq!(shown, vec![0, 48, 255, 128, 128, 128]);
        assert_eq!(largest_colour_share(&p, 4), (0.5, 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

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
        // In display space: the difference of the two encoded values.
        let shown = f64::from(srgb_display(0.75) - srgb_display(0.5));
        assert!((shown - 0.1455).abs() < 1e-3, "{shown}");
        assert!((d.max_abs - shown).abs() < 1e-9 && (d.mean_abs - shown / 6.0).abs() < 1e-9);
        assert!(compare(&a, &Picture { width: 1, height: 2, rgba: vec![0.0; 8] }).is_err());
    }

    /// A picture whose colour channels are NaN and whose alpha is right (what the volumepassage run45 champions drew:
    /// a store of 0/0 and of pow of a negative) is as wrong as can be on every error statistic, not a near match; the
    /// same NaN in both pictures is no difference.
    #[test]
    fn a_nan_picture_is_the_largest_error_on_every_statistic() {
        let want = Picture { width: 2, height: 1, rgba: vec![0.2, 0.4, 0.6, 1.0, 0.1, 0.3, 0.5, 1.0] };
        let got = Picture { width: 2, height: 1, rgba: vec![f32::NAN, f32::NAN, f32::NAN, 1.0, f32::NAN, f32::NAN, f32::NAN, 1.0] };
        let d = compare(&got, &want).expect("same size");
        assert_eq!((d.differing_channels, d.non_finite_mismatch, d.pixels_over_one_step), (6, 6, 2));
        assert!(d.rmse.is_infinite() && d.mean_abs.is_infinite() && d.max_abs.is_infinite(), "{d:?}");
        let same = compare(&got, &got).expect("same size");
        assert_eq!((same.differing_channels, same.non_finite_mismatch, same.pixels_over_one_step), (0, 0, 0));
        assert_eq!((same.rmse, same.mean_abs, same.max_abs), (0.0, 0.0, 0.0));
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

    /// A TIMED PAIR GETS ONE SUBMIT PLAN. The reference dispatched by the candidate's plan launches the same submits over
    /// the same pixels: with the candidate's workgroup shape the reference's workgroup rectangles ARE the candidate's (the
    /// plan's tiles give the split back exactly), and with any other shape the submit count is the same and each reference
    /// submit holds the candidate's pixels; a one-submit plan is the whole picture for every shape.
    #[test]
    fn a_timed_pair_uses_identical_submit_plans() {
        let shapes = [(8u32, 8u32), (2, 16), (16, 2), (4, 4), (1, 64), (7, 3)];
        for (w, h) in [(64u32, 64u32), (256, 256), (200, 136), (77, 33)] {
            for &cwg in &shapes {
                let (gx, gy) = (w.div_ceil(cwg.0), h.div_ceil(cwg.1));
                for per in [None, Some(1u64), Some(3), Some(u64::from(gx)), Some(u64::from(gx) * 2 + 1), Some(1_000_000)] {
                    let t = tiles(gx, gy, per);
                    let plan = SubmitPlan::of_tiles(&t, cwg, w, h);
                    assert_eq!(plan.submits(), t.len());
                    assert_eq!(plan.tiles(cwg), t, "{w}x{h} wg {cwg:?} per {per:?}: the plan's tiles are the split");
                    assert_eq!(SubmitPlan::of_tiles(&plan.tiles(cwg), cwg, w, h), plan, "the same shape gives the same plan");
                    assert!(plan.aligned(cwg, w, h));
                    let px: u64 = plan.rects.iter().map(|r| u64::from(r.2) * u64::from(r.3)).sum();
                    assert_eq!(px, u64::from(w) * u64::from(h), "{w}x{h} wg {cwg:?} per {per:?}: the plan covers every pixel once");
                    for &rwg in &shapes {
                        let reference = SubmitPlan::of_tiles(&plan.tiles(rwg), rwg, w, h);
                        assert!(reference.launches(&plan), "{w}x{h} candidate {cwg:?} reference {rwg:?} per {per:?}");
                        assert_eq!(reference.submits(), plan.submits());
                        if plan.aligned(rwg, w, h) {
                            assert_eq!(reference, plan, "on the grid of both shapes the plans are identical");
                        }
                    }
                }
                let one = SubmitPlan::of_tiles(&tiles(gx, gy, None), cwg, w, h);
                assert_eq!(one, SubmitPlan::single(w, h), "one submit is the whole picture for every workgroup shape");
            }
        }
        let a = SubmitPlan { rects: vec![(0, 0, 8, 8), (8, 0, 8, 8)] };
        assert!(!a.launches(&SubmitPlan::single(16, 8)), "two submits do not launch one");
        assert!(!SubmitPlan { rects: vec![(0, 0, 8, 8), (8, 0, 4, 8)] }.launches(&a), "a smaller rectangle does not hold the candidate's");
    }

    /// THE TIMING SIZE IS ONE SUBMIT FOR THE ENCODED ORIGINAL. volumepassage read into the true kingdom (its own work bound,
    /// `Fuel::Tree`, the encoded original's workgroup): at a picture of 256 x 256 the timing size is 64 x 64, the largest
    /// power-of-two square the guard allows in one submit (128 x 128 needs more than one); at 64 the pass is one submit;
    /// a proven kernel is one submit at the size itself.
    #[test]
    fn the_timing_size_is_single_submit_for_the_encoded_original() {
        use crate::wgsl::cuneus_originals::{image_dir, volumepassage};
        use crate::wgsl::true_kingdom::TrueKingdom;
        use crate::wgsl::work_guard::{Fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT};
        let kingdom = TrueKingdom::load(&image_dir("volumepassage")).expect("the volumepassage table");
        let tree = volumepassage(&kingdom).expect("the original reads exactly");
        let src = kingdom.render(&tree).expect("render");
        let module = crate::wgsl::naga_shape::parse_unscanned(&src).expect("naga");
        let ep = module.entry_points.iter().find(|e| e.name == "main").expect("main");
        let wg = (ep.workgroup_size[0], ep.workgroup_size[1]);
        let fuel = kingdom.tree_fuel(&tree).expect("fuel");
        let cap = CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;
        let s = single_submit_size(&fuel, wg, 256, cap).expect("size").expect("a single-submit size");
        assert_eq!(s, 64, "volumepassage: 64 x 64 (bound {:?} an invocation, workgroup {wg:?})", fuel.per_invocation());
        let per = fuel.workgroups_per_submit(u64::from(wg.0 * wg.1), cap).expect("split");
        assert_eq!(tiles(s.div_ceil(wg.0), s.div_ceil(wg.1), per).len(), 1, "one submit at the timing size");
        assert!(tiles(128u32.div_ceil(wg.0), 128u32.div_ceil(wg.1), per).len() > 1, "128 x 128 is more than one submit");
        assert_eq!(single_submit_size(&fuel, wg, 64, cap).expect("size"), Some(64));
        assert_eq!(single_submit_size(&fuel, wg, 100, cap).expect("size"), Some(64), "a power of two at most the size");
        assert_eq!(single_submit_size(&Fuel::Proven("a test"), (8, 8), 256, cap).expect("size"), Some(256));
        assert_eq!(single_submit_size(&Fuel::Tree { per_invocation: cap + 1 }, (1, 1), 256, cap).expect("size"), None, "not even one invocation fits");
    }

    /// A PAIR'S TIMESTAMPS COME FROM ONE COMMAND BUFFER: every submit's command buffer holds the reference's warm-ups
    /// first, then exactly one timed candidate dispatch and one timed reference dispatch of that same submit, in the
    /// order asked; every pass (warm-ups included) has its own begin and end query, the queries of all submits are
    /// distinct and cover 0..PAIR_QUERIES_PER_SUBMIT * n.
    #[test]
    fn a_pairs_timestamps_come_from_one_command_buffer() {
        for n in [1usize, 2, 5] {
            for first in [true, false] {
                let s = pair_schedule(n, first);
                assert_eq!(s.len(), n, "one command buffer a submit");
                let mut queries = Vec::new();
                for (c, buf) in s.iter().enumerate() {
                    assert_eq!(buf.len(), PAIR_WARM_DISPATCHES + 2);
                    assert!(buf[..PAIR_WARM_DISPATCHES].iter().all(|st| matches!(st, PairStep::WarmReference { submit, .. } if *submit == c)), "the warm-ups lead their command buffer");
                    let timed: Vec<(bool, usize, u32)> = buf.iter().filter_map(|st| if let PairStep::Timed { candidate, submit, query } = *st { Some((candidate, submit, query)) } else { None }).collect();
                    assert_eq!(timed.len(), 2);
                    assert_eq!((timed[0].0, timed[1].0), (first, !first), "the order alternates by pair");
                    assert!(timed.iter().all(|t| t.1 == c), "both timed dispatches are this submit's");
                    queries.extend(buf.iter().flat_map(|st| [st.query(), st.query() + 1]));
                }
                queries.sort_unstable();
                assert_eq!(queries, (0..PAIR_QUERIES_PER_SUBMIT * n as u32).collect::<Vec<_>>());
            }
        }
    }

    /// THE FENCE KERNEL VALIDATES: it is the first dispatch of every pass of a timed pair, so it must be a legal module
    /// with one read-write storage binding and its entry point.
    #[test]
    fn the_pair_fence_kernel_is_valid_wgsl() {
        let module = crate::wgsl::naga_shape::parse_unscanned(FENCE_WGSL).expect("the fence parses");
        assert!(module.entry_points.iter().any(|e| e.name == "main" && e.workgroup_size == [1, 1, 1]));
        crate::wgsl::target::Target::current().validate(&module, FENCE_WGSL).expect("the fence validates");
    }

    #[test]
    fn the_median_of_an_even_and_an_odd_list() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
