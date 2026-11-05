//! The pixel family of the Futhark kingdom: a picture as a map over the pixels of a scalar program (plan
//! `docs/PLAN_futhark_kingdom.md` 7.7, task K4; one instance a shader).
//!
//! A cuneus image shader runs one invocation a pixel: it reads the pixel's coordinates and its uniforms, runs bounded
//! loops over scalar and vector arithmetic, and stores a colour. Here that is a Futhark program
//!
//! ```text
//! map (\yi -> map (\xi -> let px = .. in let py = .. in <pixel block>) (iota cols)) (iota rows)
//! ```
//!
//! whose pixel block is a chain of `let`s and bounded `loop`s over `f32`, ending in `[r, g, b]`. The kingdom is ROWS of
//! the master tables, one variant of `futhark` an instance ([`rows`]: shape types only, scope by `var.f32`, the loop
//! bound by `env.trips`), written by `kingdoms/futhark/pixel_rows.py` through `gen_tables.py`.
//!
//! What this module holds besides [`rows`]: the instance's artefacts (its interface, read from the shader's uniform
//! blocks, and its e-graph over the rows, `kingdoms/futhark/pixel/<shader>/`), the input vector of a case, the band
//! plan that keeps every submit under the GPU cap, the half-float rounding of the output texture, Futhark's C backend
//! as a reference runner, and the references written by hand ([`VOLUMEPASSAGE_REFERENCE`], [`CUNEUS_REFERENCE`]),
//! which [`rows::PixelRows::read`] reads.

/// One instance of the family: the generated artefacts of one shader.
#[derive(Debug, Clone, Copy)]
pub struct Instance {
    pub name: &'static str,
    interface: &'static str,
    pub datatype: &'static str,
    pub rules: &'static str,
    pub egraph_rows: &'static str,
}

/// cuneus `volumepassage.wgsl`: a 50-step ray march through a 7-octave sine noise.
pub const VOLUMEPASSAGE: Instance = Instance {
    name: "volumepassage",
    interface: include_str!("../kingdoms/futhark/pixel/volumepassage/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark/pixel/volumepassage/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark/pixel/volumepassage/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark/pixel/volumepassage/egraph.rows"),
};

/// cuneus `cuneus.wgsl`: the glowing "CUNEUS" letters (a 5-layer loop over a text SDF with lighting).
pub const CUNEUS: Instance = Instance {
    name: "cuneus",
    interface: include_str!("../kingdoms/futhark/pixel/cuneus/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark/pixel/cuneus/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark/pixel/cuneus/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark/pixel/cuneus/egraph.rows"),
};

/// The IMAGE TARGET instance: no shader and no uniforms; a program is a function of the pixel's position (`px`,
/// `py`, and `nx`, `ny` as fractions of the picture) and the picture's size. Any picture file is a target of it.
pub const IMAGE: Instance = Instance {
    name: "image",
    interface: include_str!("../kingdoms/futhark/pixel/image/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark/pixel/image/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark/pixel/image/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark/pixel/image/egraph.rows"),
};

/// An instance and its hand-written reference (None for the image target, which has none), by name.
pub fn instance(shader: &str) -> Result<(Instance, Option<&'static str>), String> {
    match shader {
        "volumepassage" => Ok((VOLUMEPASSAGE, Some(VOLUMEPASSAGE_REFERENCE))),
        "cuneus" => Ok((CUNEUS, Some(CUNEUS_REFERENCE))),
        "image" => Ok((IMAGE, None)),
        other => Err(format!("no pixel instance for {other}")),
    }
}

/// The pixel family on the rows of the master tables (plan 7.7 K4).
pub mod rows;

/// The ruleset name the generated rules are in.
pub const RULESET: &str = "futhark_pixel";

/// One input of the interface: a uniform field of the shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub group: u32,
    pub binding: u32,
    /// The field's word offset in its block.
    pub word: usize,
    /// The Futhark variable the program reads it as (`u_speed`).
    pub spelling: String,
}

/// The interface of an instance, in the order of the input vector.
pub fn interface(inst: &Instance) -> Result<Vec<Input>, String> {
    let mut out = Vec::new();
    for line in inst.interface.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 5 {
            return Err(format!("{}: the interface line {line:?} has {} fields, wants 5", inst.name, f.len()));
        }
        let num = |s: &str| s.parse::<usize>().map_err(|e| format!("{}: {line:?}: {e}", inst.name));
        out.push(Input { name: f[0].to_string(), group: num(f[1])? as u32, binding: num(f[2])? as u32, word: num(f[3])?, spelling: f[4].to_string() });
    }
    Ok(out)
}

/// The number of words of the input vector before the interface's inputs:
/// width, height, and the tile: its first column and row, its columns and rows.
pub const HEADER: usize = 6;

/// A rectangle of the picture one run draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub col0: u32,
    pub row0: u32,
    pub cols: u32,
    pub rows: u32,
}

/// The input vector of one tile of one case: `[width, height, col0, row0,
/// cols, rows]` and then every interface input, read from the case's uniform
/// blocks (`(group, binding, bytes)`) at its word.
pub fn inputs(iface: &[Input], blocks: &[(u32, u32, &[u8])], width: u32, height: u32, tile: Tile) -> Result<Vec<f32>, String> {
    let mut ps = vec![width as f32, height as f32, tile.col0 as f32, tile.row0 as f32, tile.cols as f32, tile.rows as f32];
    for i in iface {
        let (_, _, bytes) = blocks
            .iter()
            .find(|(g, b, _)| *g == i.group && *b == i.binding)
            .ok_or_else(|| format!("the case has no uniform block at group {} binding {} for {}", i.group, i.binding, i.name))?;
        let w = bytes.get(i.word * 4..i.word * 4 + 4).ok_or_else(|| format!("the block of {} has no word {}", i.name, i.word))?;
        ps.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    }
    Ok(ps)
}



/// The tiles a picture is drawn in, so that no submit's bound (`work` a pixel
/// times the tile's pixels) exceeds `cap`: bands of whole rows when a row
/// fits, runs of columns within a row when it does not. One pixel over the cap
/// is a misconfigured experiment (named, never run).
pub fn tiles(work: u128, width: u32, height: u32, cap: u128) -> Result<Vec<Tile>, String> {
    if work > cap {
        return Err(format!("one pixel of this tree is bounded by {work} node executions, over the per-submit cap {cap}"));
    }
    let per = cap / work.max(1);
    let mut out = Vec::new();
    if per >= u128::from(width) {
        let rows = u32::try_from((per / u128::from(width)).min(u128::from(height))).unwrap_or(height).max(1);
        let mut r = 0;
        while r < height {
            let n = rows.min(height - r);
            out.push(Tile { col0: 0, row0: r, cols: width, rows: n });
            r += n;
        }
    } else {
        let cols = u32::try_from(per).unwrap_or(1).max(1);
        for r in 0..height {
            let mut c = 0;
            while c < width {
                let n = cols.min(width - c);
                out.push(Tile { col0: c, row0: r, cols: n, rows: 1 });
                c += n;
            }
        }
    }
    Ok(out)
}

/// Put a tile's `[rows][cols][3]` output into a picture's RGBA (each channel
/// rounded to a half, as the texture store rounds it; alpha 1).
pub fn place(rgba: &mut [f32], width: u32, tile: Tile, values: &[f32]) -> Result<(), String> {
    if values.len() != (tile.cols * tile.rows * 3) as usize {
        return Err(format!("a tile of {}x{} gave {} values", tile.cols, tile.rows, values.len()));
    }
    for y in 0..tile.rows {
        for x in 0..tile.cols {
            let v = &values[((y * tile.cols + x) * 3) as usize..][..3];
            let at = (((tile.row0 + y) * width + tile.col0 + x) * 4) as usize;
            rgba[at..at + 4].copy_from_slice(&[f16_round(v[0]), f16_round(v[1]), f16_round(v[2]), 1.0]);
        }
    }
    Ok(())
}

/// `x` rounded to the nearest half-float (ties to even), as a store into an
/// `rgba16float` texture rounds it, returned as the `f32` of that half.
pub fn f16_round(x: f32) -> f32 {
    crate_half::to_f32(crate_half::from_f32(x))
}

/// Half-float conversion, ties to even.
mod crate_half {
    pub fn from_f32(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = man | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = m >> shift;
            let rem = m & ((1 << shift) - 1);
            let mid = 1 << (shift - 1);
            let rounded = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
            return sign | rounded as u16;
        }
        let half = ((e as u32) << 10) | (man >> 13);
        let rem = man & 0x1fff;
        let rounded = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
        sign | rounded as u16
    }

    pub fn to_f32(h: u16) -> f32 {
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
}



/// Compile a program with Futhark's sequential C backend (`futhark c`, the
/// released compiler) into `dir`; returns the executable. The reference
/// path beside the GPU's (`futhark_wgpu_host`).
pub fn compile_c(source: &str, dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, source).map_err(|e| format!("write {}: {e}", fut.display()))?;
    let exe = dir.join("prog_c");
    let out = crate::futhark_exec::output_with_timeout(
        std::process::Command::new("futhark").arg("c").arg(&fut).arg("-o").arg(&exe),
        std::time::Duration::from_secs(120),
        None,
    )?;
    if !out.status.success() {
        return Err(format!("futhark c failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(exe)
}

/// Run a compiled program (either backend's executable) on an input vector
/// and read its `[rows][cols][3]f32` back from Futhark's binary format.
pub fn run_exe(exe: &std::path::Path, ps: &[f32]) -> Result<Vec<f32>, String> {
    let out = crate::futhark_exec::output_with_timeout(
        std::process::Command::new(exe).arg("-b"),
        std::time::Duration::from_secs(120),
        Some(&crate::futhark_exec::input_text(ps)?),
    )?;
    if !out.status.success() {
        return Err(format!("{} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
    }
    crate::futhark_exec::read_binary_f32(&out.stdout)
}






// ---- a program written by hand, as a tree of rows ---------------------------

/// The reference: volumepassage written by hand in the family's rows, the
/// original's arithmetic operation for operation and in its order (vector
/// maths per lane; `normalize(v)` as `v / sqrt(dot(v, v))`; the matrices
/// column by column). It is the TARGET a search is measured against, never a
/// seed of a petri dish.
///
/// The form is an s-expression the reader ([`rows::PixelRows::read`]) reads with the scope
/// tracked for it: `let VALUE REST` binds the next variable, `loopS (INITS)
/// COUNT BODY REST` binds S state variables and (in the body) the counter,
/// `blockS BODY REST` binds the S values its body returns, so
/// the comment after each binding names the variable it is.
pub const VOLUMEPASSAGE_REFERENCE: &str = r"
(image
 (let (div (sub px (mul 0.5 u_width)) u_zoom)                                   ; v0 (C - 0.5 r).x / zoom
 (let (div (sub py (mul 0.5 u_height)) u_zoom)                                  ; v1 (C - 0.5 r).y / zoom
 (let (sqrt (add (add (mul v0 v0) (mul v1 v1)) (mul u_height u_height)))        ; v2 |(v0, v1, r.y)|
 (let (div v0 v2)                                                               ; v3 the ray, x
 (let (div v1 v2)                                                               ; v4 the ray, y
 (let (div u_height v2)                                                         ; v5 the ray, z
 (let (mul u_time u_speed)                                                      ; v6 t
 (loop5 (0.0 0.0 0.0 0.0 0.0) 50                                                ; v7 z, v8 d, v9..v11 o; v12 j
  (let (add v7 (mul v8 0.6))                                                    ; v13 z += d * 0.6
  (let (mul v13 v3)                                                             ; v14 p.x
  (let (mul v13 v4)                                                             ; v15 p.y
  (let (add (mul v13 v5) v6)                                                    ; v16 p.z (+ t)
  (let (cos (mul v16 1.1))                                                      ; v17 c
  (let (sin (mul v16 1.1))                                                      ; v18 s
  (let (add (mul v17 v14) (mul v18 v15))                                        ; v19 rotated x
  (let (add (mul (neg v18) v14) (mul v17 v15))                                  ; v20 rotated y
  (let (mul v6 0.1)                                                             ; v21 t * 0.1
  (loop5 (0.0 1.0 (add v19 v21) (add v20 v21) (add v16 v21)) 7                  ; v22 v, v23 a, v24..v26 pos; v27 i
   (let (add v24 (mul v26 0.5))                                                 ; v28 (pos.xy + pos.z * 0.5).x
   (let (add v25 (mul v26 0.5))                                                 ; v29 (pos.xy + pos.z * 0.5).y
   (tuple (add v22 (mul (mul (sin (add (mul v28 3.0) (sin (mul v29 2.7))))
                             (cos (add (mul v29 1.1) (cos (mul v28 2.3))))) v23))
          (mul v23 0.5) (mul v24 2.0) (mul v25 2.0) (mul v26 2.0))))
   (let (add (abs (add (sub (sqrt (add (mul v19 v19) (mul v20 v20))) 1.0) (mul v22 0.3))) 0.01)  ; v27 d
   (let (mod v16 4.0)                                                           ; v28 p.z % 4
   (let (sub (sel v28 v28 (add v28 4.0)) 2.0)                                   ; v29 pmap(p.z, 4)
   (let (add v19 v22)                                                           ; v30 (p.xy + N).x
   (let (add v20 v22)                                                           ; v31 (p.xy + N).y
   (let (add v29 v22)                                                           ; v32 (p.xz + N).y
   (let (max (mul (sqrt (add (mul v30 v30) (mul v31 v31))) 0.8) 0.005)         ; v33 den1
   (let (max (mul (sqrt (add (mul v30 v30) (mul v32 v32))) 0.8) 0.005)         ; v34 den2
   (let (add 0.5 (mul 0.5 (sin (add (mul v22 1.0) v6))))                        ; v35 0.5 + 0.5 sin(N + t)
   (tuple v13 v27
          (add v9 (mul (div (add (add (div u_color1_r v33) (div u_color2_r v34)) (mul u_color3_r v35)) v27) 0.12))
          (add v10 (mul (div (add (add (div u_color1_g v33) (div u_color2_g v34)) (mul u_color3_g v35)) v27) 0.12))
          (add v11 (mul (div (add (add (div u_color1_b v33) (div u_color2_b v34)) (mul u_color3_b v35)) v27) 0.12)))
   ))))))))))
  )))))))))
  (let (tanh (mul v9 u_intensity))                                              ; v12
  (let (tanh (mul v10 u_intensity))                                             ; v13
  (let (tanh (mul v11 u_intensity))                                             ; v14
  (let (add (add (mul 0.59719 v12) (mul 0.35458 v13)) (mul 0.04823 v14))       ; v15 (m1 v).x
  (let (add (add (mul 0.07600 v12) (mul 0.90834 v13)) (mul 0.01566 v14))       ; v16 (m1 v).y
  (let (add (add (mul 0.02840 v12) (mul 0.13383 v13)) (mul 0.83777 v14))       ; v17 (m1 v).z
  (let (div (sub (mul v15 (add v15 0.0245786)) 0.000090537) (add (mul v15 (add (mul 0.983729 v15) 0.4329510)) 0.238081))  ; v18
  (let (div (sub (mul v16 (add v16 0.0245786)) 0.000090537) (add (mul v16 (add (mul 0.983729 v16) 0.4329510)) 0.238081))  ; v19
  (let (div (sub (mul v17 (add v17 0.0245786)) 0.000090537) (add (mul v17 (add (mul 0.983729 v17) 0.4329510)) 0.238081))  ; v20
  (let (div 1.0 u_gamma)                                                        ; v21
  (rgb (pow (max (add (add (mul 1.60475 v18) (mul -0.53108 v19)) (mul -0.07367 v20)) 0.0) v21)
       (pow (max (add (add (mul -0.10208 v18) (mul 1.10813 v19)) (mul -0.00605 v20)) 0.0) v21)
       (pow (max (add (add (mul -0.00327 v18) (mul -0.07276 v19)) (mul 1.07602 v20)) 0.0) v21))
  ))))))))))
 )))))))))
";

/// The reference: cuneus.wgsl written by hand in the family's rows, the original's arithmetic in its order, the
/// TARGET only. How the WGSL maps onto the rows:
///
/// * the layer loop `for (var i = 0.1; i < 1.0; i += 0.2)` is a 5-trip `loop5` whose state is the colour's four
///   lanes and `i` itself (so `i` accumulates in f32 exactly as the original's does);
/// * `continue` (skip the layer when `alpha > alpha_threshold && |d| > 0.1`) is a per-lane select:
///   `sel(max(thr - alpha, 0.1 - |d|), updated, old)` keeps the old lane exactly when both strict tests hold;
/// * the text SDF at the five points the shader evaluates (the centre and `+-eps` in x and y, the normal's four
///   neighbours shared by the light and the environment terms) is a 5-trip loop, the point chosen by a select
///   chain on the counter and each result stored into its lane by a select; each letter, and each `box_sdf` and
///   `hs` inside one, is a `block` so its temporaries end with it (the scope holds at most 40 variables);
/// * the 8 lights are an 8-trip loop over the counter, `positions[i]` a select chain;
/// * `hs` with `c = (0, 1)`: the matrix `(-c.x, c.y; c.y, c.x)` maps `p` to `(p.y, |p.x|)`, and
///   `l * sign(-c.x)` is 0; `mix(a, b, t)` is `a * (1 - t) + b * t`, `smoothstep` `t * t * (3 - 2t)` of the
///   clamped ramp, `normalize(v)` `v / |v|`, constants such as `PI / 4.0` computed from the pool's literals.
///
/// Generated by a scratch script from these rules (so the 30 box and letter forms are spelled once); the text
/// is the source of truth.
pub const CUNEUS_REFERENCE: &str = r"
(image
 (let (div (mul 6.0 (sub px (mul 0.5 u_width))) u_height)
 (let (div (mul 6.0 (neg (sub py (mul 0.5 u_height)))) u_height)
 (let (mul u_time 0.5)
 (loop5 (u_background_color u_background_color u_background_color 1.0 0.1) 5
 (let (mul (sin (add (mul v2 0.3) (mul v7 0.5))) 0.05)
 (let (cos v9)
 (let (sin v9)
 (let (add (mul v0 v10) (mul v1 v11))
 (let (add (mul v0 (neg v11)) (mul v1 v10))
 (let (add 0.1 (mul 0.02 (sin (add (mul v2 0.2) (mul v7 0.7)))))
 (let (mul v14 v7)
 (loop5 (0.0 0.0 0.0 0.0 0.0) 5
 (let (sel (sub v21 4.0) (sub v12 0.0) (sel (sub v21 3.0) (add v12 0.0) (sel (sub v21 2.0) (sub v12 0.001) (sel (sub v21 1.0) (add v12 0.001) v12))))
 (let (sel (sub v21 4.0) (sub v13 0.001) (sel (sub v21 3.0) (add v13 0.001) (sel (sub v21 2.0) (sub v13 0.0) (sel (sub v21 1.0) (add v13 0.0) v13))))
 (block1 (let (add v22 (add 3.0 (mul 3.0 v15)))
 (let (add v23 0.5)
 (block1 (let (sub v24 0.5)
 (let (sub v25 0.5)
 (let (sqrt (add (mul v26 v26) (mul v27 v27)))
 (let (sub (abs (sub v26 0.3)) 0.5)
 (let (sub (abs (sub v27 0.0)) 0.2)
 (tuple (min 1.0e6 (max (max (sub v28 0.5) (neg (sub v28 0.3))) (neg (add (sqrt (add (mul (max v29 0.0) (max v29 0.0)) (mul (max v30 0.0) (max v30 0.0)))) (min (max v29 v30) 0.0)))))))))))
 (block1 (let (sqrt (add (mul (abs (sub (sub (sub v24 (add 1.05 v15)) 0.5) 0.0)) (abs (sub (sub (sub v24 (add 1.05 v15)) 0.5) 0.0))) (mul (sub (sub v25 0.475) 0.125) (sub (sub v25 0.475) 0.125))))
 (let (sel (neg (abs (sub (sub (sub v24 (add 1.05 v15)) 0.5) 0.0))) 0.0 (sub (sub v25 0.475) 0.125))
 (let (sub v28 0.4)
 (let (sub (abs (sub (sel (neg v28) v27 (abs (sub (sub (sub v24 (add 1.05 v15)) 0.5) 0.0))) 0.5)) 0.1)
 (tuple (min v26 (add (sqrt (add (mul (max v29 0.0) (max v29 0.0)) (mul (max v30 0.0) (max v30 0.0)))) (min 0.0 (max v29 v30)))))))))
 (block1 (block1
 (let (sub (abs (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) 0.2)) 0.1)
 (let (sub (abs (sub v25 0.5)) 0.5)
 (tuple (add (sqrt (add (mul (max v28 0.0) (max v28 0.0)) (mul (max v29 0.0) (max v29 0.0)))) (min (max v28 v29) 0.0)))))
 (block1 (let (sub (abs (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) 0.8)) 0.1)
 (let (sub (abs (sub v25 0.5)) 0.5)
 (tuple (add (sqrt (add (mul (max v29 0.0) (max v29 0.0)) (mul (max v30 0.0) (max v30 0.0)))) (min (max v29 v30) 0.0)))))
 (let (add (mul (cos (div 3.141592654 4.0)) (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) 0.5)) (mul (neg (sin (div 3.141592654 4.0))) (sub v25 0.5)))
 (let (add (mul (sin (div 3.141592654 4.0)) (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) 0.5)) (mul (cos (div 3.141592654 4.0)) (sub v25 0.5)))
 (block1 (let (sub (abs v30) 0.7)
 (let (sub (abs v31) 0.1)
 (tuple (add (sqrt (add (mul (max v32 0.0) (max v32 0.0)) (mul (max v33 0.0) (max v33 0.0)))) (min (max v32 v33) 0.0)))))
 (tuple (min v27 (min (min v28 v29) v32))))))))
 (block1 (let (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) 0.5)
 (let (sub v25 0.5)
 (block1 (let (sub (abs (sub v29 -0.3)) 0.1)
 (let (sub (abs (sub v30 0.0)) 0.5)
 (tuple (add (sqrt (add (mul (max v31 0.0) (max v31 0.0)) (mul (max v32 0.0) (max v32 0.0)))) (min (max v31 v32) 0.0)))))
 (block1 (let (sub (abs (sub v29 0.0)) 0.3)
 (let (sub (abs (sub v30 0.4)) 0.1)
 (tuple (add (sqrt (add (mul (max v32 0.0) (max v32 0.0)) (mul (max v33 0.0) (max v33 0.0)))) (min (max v32 v33) 0.0)))))
 (block1 (let (sub (abs (sub v29 0.0)) 0.3)
 (let (sub (abs (sub v30 0.0)) 0.1)
 (tuple (add (sqrt (add (mul (max v33 0.0) (max v33 0.0)) (mul (max v34 0.0) (max v34 0.0)))) (min (max v33 v34) 0.0)))))
 (block1 (let (sub (abs (sub v29 0.0)) 0.3)
 (let (sub (abs (sub v30 -0.4)) 0.1)
 (tuple (add (sqrt (add (mul (max v34 0.0) (max v34 0.0)) (mul (max v35 0.0) (max v35 0.0)))) (min (max v34 v35) 0.0)))))
 (tuple (min v28 (min (min (min v31 v32) v33) v34)))))))))
 (block1 (let (sqrt (add (mul (abs (sub (sub (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) (add 1.0 v15)) 0.5) 0.0)) (abs (sub (sub (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) (add 1.0 v15)) 0.5) 0.0))) (mul (sub (sub v25 0.475) 0.125) (sub (sub v25 0.475) 0.125))))
 (let (sel (neg (abs (sub (sub (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) (add 1.0 v15)) 0.5) 0.0))) 0.0 (sub (sub v25 0.475) 0.125))
 (let (sub v31 0.4)
 (let (sub (abs (sub (sel (neg v31) v30 (abs (sub (sub (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) (add 1.0 v15)) 0.5) 0.0))) 0.5)) 0.1)
 (tuple (min v29 (add (sqrt (add (mul (max v32 0.0) (max v32 0.0)) (mul (max v33 0.0) (max v33 0.0)))) (min 0.0 (max v32 v33)))))))))
 (block1 (let (add (mul (cos (sub (div (neg 3.141592654) 6.0) (div 3.141592654 2.0))) (sub (sub (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) (add 1.0 v15)) (add 1.0 v15)) 0.435)) (mul (neg (sin (sub (div (neg 3.141592654) 6.0) (div 3.141592654 2.0)))) (sub v25 0.5)))
 (let (add (mul (sin (sub (div (neg 3.141592654) 6.0) (div 3.141592654 2.0))) (sub (sub (sub (sub (sub (sub v24 (add 1.05 v15)) (add 1.0 v15)) (add 1.1 v15)) (add 1.0 v15)) (add 1.0 v15)) 0.435)) (mul (cos (sub (div (neg 3.141592654) 6.0) (div 3.141592654 2.0))) (sub v25 0.5)))
 (block1 (let (sqrt (add (mul (abs (sub v31 (div (mul -0.25 3.0) 4.0))) (abs (sub v31 (div (mul -0.25 3.0) 4.0)))) (mul (sub v32 (div -0.125 2.0)) (sub v32 (div -0.125 2.0)))))
 (let (sel (neg (abs (sub v31 (div (mul -0.25 3.0) 4.0)))) 0.0 (sub v32 (div -0.125 2.0)))
 (let (sub v34 0.2)
 (let (sub (abs (sub (sel (neg v34) v33 (abs (sub v31 (div (mul -0.25 3.0) 4.0)))) 0.375)) 0.1)
 (tuple (add (sqrt (add (mul (max v35 0.0) (max v35 0.0)) (mul (max v36 0.0) (max v36 0.0)))) (min 0.0 (max v35 v36))))))))
 (let (add (mul (cos 3.141592654) v31) (mul (neg (sin 3.141592654)) v32))
 (let (add (mul (sin 3.141592654) v31) (mul (cos 3.141592654) v32))
 (block1 (let (sqrt (add (mul (abs (sub v34 (div (mul -0.25 3.0) 4.0))) (abs (sub v34 (div (mul -0.25 3.0) 4.0)))) (mul (sub v35 (div -0.125 2.0)) (sub v35 (div -0.125 2.0)))))
 (let (sel (neg (abs (sub v34 (div (mul -0.25 3.0) 4.0)))) 0.0 (sub v35 (div -0.125 2.0)))
 (let (sub v37 0.2)
 (let (sub (abs (sub (sel (neg v37) v36 (abs (sub v34 (div (mul -0.25 3.0) 4.0)))) 0.375)) 0.1)
 (tuple (add (sqrt (add (mul (max v38 0.0) (max v38 0.0)) (mul (max v39 0.0) (max v39 0.0)))) (min 0.0 (max v38 v39))))))))
 (tuple (min v30 (min v33 v36)))))))))
 (tuple v31)))))))))
 (tuple (sel (neg (abs (sub v21 0.0))) v24 v16) (sel (neg (abs (sub v21 1.0))) v24 v17) (sel (neg (abs (sub v21 2.0))) v24 v18) (sel (neg (abs (sub v21 3.0))) v24 v19) (sel (neg (abs (sub v21 4.0))) v24 v20)))))
 (block3 (let (sub v17 v18)
 (let (sub v19 v20)
 (let (sqrt (add (mul v21 v21) (mul v22 v22)))
 (tuple (div v21 v23) (div v22 v23) 0.0))))
 (block1 (let (min (max (div (sub v16 -0.01) (sub 0.01 -0.01)) 0.0) 1.0)
 (tuple (mul (mul v24 v24) (sub 3.0 (mul 2.0 v24)))))
 (block1 (let (sin (add (mul v7 13.37) (mul v2 0.3)))
 (let (cos (sub (mul v7 7.54) (mul v2 0.4)))
 (let (sin (add (mul v7 9.21) (mul v2 0.5)))
 (loop3 (0.0 0.0 0.0) 8
 (let (sub v12 (add (sel (sub v31 7.0) (add (neg (add 3.0 (mul 3.0 v7))) (add 6.0 (mul (mul (sub 0.0 6.0) 0.5) (add (sin (div (mul (mul 2.0 3.141592654) v2) 7.0)) 1.0)))) (sel (sub v31 6.0) (add (neg (add 3.0 (mul 3.0 v7))) (add 0.0 (mul (mul (sub 6.0 0.0) 0.5) (add (sin (div (mul (mul 2.0 3.141592654) v2) 8.0)) 1.0)))) (sel (sub v31 5.0) (add (neg (add 3.0 (mul 3.0 v7))) 5.525) (sel (sub v31 4.0) (add (neg (add 3.0 (mul 3.0 v7))) 4.65) (sel (sub v31 3.0) (add (neg (add 3.0 (mul 3.0 v7))) 3.65) (sel (sub v31 2.0) (add (neg (add 3.0 (mul 3.0 v7))) 2.65) (sel (sub v31 1.0) (add (neg (add 3.0 (mul 3.0 v7))) 1.55) (add (neg (add 3.0 (mul 3.0 v7))) 0.5)))))))) (mul (cos (add (add (mul v2 0.5) (mul v31 0.7)) v25)) 0.2)))
 (let (sub v13 (add (sel (sub v31 7.0) (add 1.0 (mul (mul (sub 0.0 1.0) 0.5) (add (sin (div (mul (mul 2.0 3.141592654) v2) 6.0)) 1.0))) (sel (sub v31 6.0) (add 0.0 (mul (mul (sub 1.0 0.0) 0.5) (add (sin (div (mul (mul 2.0 3.141592654) v2) 5.0)) 1.0))) (sel (sub v31 5.0) 0.5 (sel (sub v31 4.0) 0.5 (sel (sub v31 3.0) 0.5 (sel (sub v31 2.0) 0.5 (sel (sub v31 1.0) 0.5 0.5))))))) (mul (sin (add (add (mul v2 0.7) (mul v31 0.5)) v26)) 0.2)))
 (let (add 0.2 (mul 0.1 (sin (add (mul (mul v7 v31) 1.3) (mul v2 0.3)))))
 (tuple (add v28 (mul (div 1.0 (add 1.0 (mul (mul (sqrt (add (mul v32 v32) (mul v33 v33))) u_falloff_distance) (add 1.0 (mul 0.3 (sin (add (mul v27 v31) v2))))))) v34)) (add v29 v34) 0.0))))
 (let (mul (div v28 v29) 1.5)
 (let (pow (sub 1.0 (abs (add (mul (div v12 (sqrt (add (mul v12 v12) (mul v13 v13)))) v21) (mul (div v13 (sqrt (add (mul v12 v12) (mul v13 v13)))) v22)))) u_rim_power)
 (let (min (max (div (sub (abs v16) 0.0) (sub 0.05 0.0)) 0.0) 1.0)
 (let (mul v32 (sub 1.0 (mul (mul v33 v33) (sub 3.0 (mul 2.0 v33)))))
 (tuple (mul (add (mul (mul v31 (sub 1.1 (mul (mul v7 u_ao_strength) (add 1.0 (mul 0.2 (sin (add (mul v7 20.0) v2))))))) u_light_intensity) (mul v34 0.8)) (add 0.9 (mul 0.2 (add (mul (mul (sin (add (mul v7 10.0) v2)) (cos (sub (mul v7 7.0) (mul v2 0.5)))) 0.5) 0.5)))))))))))))
 (block1 (let (cos (mul v2 0.5))
 (let (sin (mul v2 0.5))
 (let (sqrt (add (mul v26 v26) (mul v27 v27)))
 (let (add (mul (add (mul v21 (div v26 v28)) (mul v22 (div v27 v28))) 0.5) 0.5)
 (tuple (mul (mul (add (mul v29 (sub 1.0 0.5)) (mul (add (mul (sin (add (mul v7 4.0) v2)) 0.5) 0.5) 0.5)) (sub 1.0 (div v7 1.0))) u_env_light_strength))))))
 (let (add 0.8 (mul (mul (mul 0.5 (sin (add (mul v7 13.37) v2))) (cos (sub (mul v7 7.54) (mul v2 0.4)))) (sin (add (mul v9 3.0) (mul v2 0.7)))))
 (let (add 0.2 (mul (mul 0.1 (cos (add (mul v7 9.21) (mul v2 0.5)))) (sin (sub (mul v9 5.0) (mul v2 0.3)))))
 (let (add (mul (sin (add (add (add (mul v7 0.8) (mul v9 2.0)) u_hue_color_x) (mul v2 0.3))) v28) v27)
 (let (add (mul (sin (add (add (add (mul v7 0.8) (mul v9 2.0)) u_hue_color_y) (mul v2 0.3))) v28) v27)
 (let (add (mul (sin (add (add (add (mul v7 0.8) (mul v9 2.0)) u_hue_color_z) (mul v2 0.3))) v28) v27)
 (let (add (mul (sin (add (add (add (mul v7 0.8) (mul v9 2.0)) u_hue_color_w) (mul v2 0.3))) v28) v27)
 (let (sub 0.0 (div (sub v7 0.1) (sub 3.0 0.1)))
 (let (add (mul (mul (sin (add (mul (add (mul v12 v12) (mul v13 v13)) 3.0) v2)) u_iridescence_power) v33) 0.9)
 (let (mul (div 0.1 (add (abs v16) 0.05)) (sub 1.0 (mul v33 u_mix_factor_scale)))
 (let (mul (mul 0.3 (sub 0.2 v24)) (add (div v13 (add (abs v16) 0.1)) v25))
 (tuple (sel (max (sub u_alpha_threshold v24) (sub 0.1 (abs v16))) (mul (add (mul (mul (mul v29 (add (mul v25 u_global_light) v26)) v34) (sub 1.0 v24)) (mul v3 v24)) (add (mul 1.0 (sub 1.0 v35)) (mul (add v29 v36) v35))) v3) (sel (max (sub u_alpha_threshold v24) (sub 0.1 (abs v16))) (mul (add (mul (mul (mul v30 (add (mul v25 u_global_light) v26)) (mul v34 0.95)) (sub 1.0 v24)) (mul v4 v24)) (add (mul 1.0 (sub 1.0 v35)) (mul (add v30 v36) v35))) v4) (sel (max (sub u_alpha_threshold v24) (sub 0.1 (abs v16))) (mul (add (mul (mul (mul v31 (add (mul v25 u_global_light) v26)) (mul v34 1.05)) (sub 1.0 v24)) (mul v5 v24)) (add (mul 1.0 (sub 1.0 v35)) (mul (add v31 v36) v35))) v5) (sel (max (sub u_alpha_threshold v24) (sub 0.1 (abs v16))) (mul (add (mul (mul (mul v32 (add (mul v25 u_global_light) v26)) 1.0) (sub 1.0 v24)) (mul v6 v24)) (add (mul 1.0 (sub 1.0 v35)) (mul (add v32 v36) v35))) v6) (add v7 0.2)))))))))))))))))))))))) (rgb (pow (mul v3 u_global_light) (div 1.0 0.45)) (pow (mul v4 u_global_light) (div 1.0 0.45)) (pow (mul v5 u_global_light) (div 1.0 0.45))))))))
";

#[derive(Debug, Clone, PartialEq)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn parse_sexp(text: &str) -> Result<Sexp, String> {
    let mut tokens = Vec::new();
    for line in text.lines() {
        let line = line.split(';').next().unwrap_or("");
        let spaced = line.replace('(', " ( ").replace(')', " ) ");
        tokens.extend(spaced.split_whitespace().map(str::to_string));
    }
    let mut pos = 0;
    let out = parse_at(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!("text after the expression: {:?}", &tokens[pos..tokens.len().min(pos + 5)]));
    }
    Ok(out)
}

fn parse_at(tokens: &[String], pos: &mut usize) -> Result<Sexp, String> {
    let t = tokens.get(*pos).ok_or("the expression ends early")?;
    *pos += 1;
    match t.as_str() {
        "(" => {
            let mut items = Vec::new();
            loop {
                match tokens.get(*pos).map(String::as_str) {
                    None => return Err("an unclosed (".into()),
                    Some(")") => {
                        *pos += 1;
                        return Ok(Sexp::List(items));
                    }
                    Some(_) => items.push(parse_at(tokens, pos)?),
                }
            }
        }
        ")" => Err("an unopened )".into()),
        _ => Ok(Sexp::Atom(t.clone())),
    }
}






#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_rounding_is_ties_to_even() {
        for (x, want) in [(1.0f32, 1.0f32), (0.1, 0.099975586), (65504.0, 65504.0), (1e-8, 0.0), (-2.5, -2.5), (1.0 + 1.0 / 2048.0, 1.0), (1.0 + 3.0 / 2048.0, 1.0 + 2.0 / 1024.0), (6.1e-5, 6.097555e-5), (65520.0, f32::INFINITY), (3e-8, 5.9604645e-8), (2.98e-8, 0.0)] {
            assert_eq!(f16_round(x).to_bits(), want.to_bits(), "{x}");
        }
        assert!(f16_round(f32::NAN).is_nan());
        assert_eq!(f16_round(1e6), f32::INFINITY);
    }

    #[test]
    fn tiles_keep_every_submit_under_the_cap() {
        assert_eq!(tiles(10, 64, 64, 10 * 64 * 64).unwrap(), vec![Tile { col0: 0, row0: 0, cols: 64, rows: 64 }]);
        let b = tiles(10, 64, 64, 10 * 64 * 5).unwrap();
        assert_eq!(b.len(), 13);
        assert!(b.iter().all(|t| t.rows <= 5 && t.cols == 64) && b.iter().map(|t| t.rows).sum::<u32>() == 64);
        let c = tiles(10, 64, 2, 10 * 20).unwrap();
        assert_eq!(c.len(), 8, "four runs of 20, 20, 20, 4 columns in each of two rows");
        assert!(c.iter().all(|t| t.rows == 1 && t.cols <= 20) && c.iter().map(|t| t.cols).sum::<u32>() == 128);
        assert!(tiles(11, 64, 64, 10).is_err());
    }
}
