//! ST9: cuneus visual shaders written as trees of the rows of the true kingdom's image instance
//! (`kingdoms/wgsl_true/img/<shader>`, `python3 gen_tables.py img_<shader>`).
//!
//! The originals are `~/Dev/gamakon/cuneus/examples/shaders/*.wgsl`, copied unchanged under `tests/corpus/cuneus_visual`.
//! Each construct is the row that spells it. A construct with no row is a table gap: the builder records every
//! missing row and returns them all, never a substitute. User functions are inlined (the kingdom has no calls); a
//! value used twice lives in a register (a `reg.set_*` statement then `reg.get_*`), so no subtree is copied.
//!
//! Arithmetic is the original's, operation for operation and in the original's order, so the picture is the original's
//! picture; what differs from the source text is spelling only: a vector plus a scalar is the vector plus the scalar
//! splatted (every lane the same operation), `if (out of range) { return; }` is the store's own bounds guard.

use std::cell::RefCell;
use std::path::PathBuf;

use crate::typed_gep::Tree;
use crate::wgsl::image::{words_f32, BindingData};
use crate::wgsl::true_kingdom::TrueKingdom;

/// The directory of the generated image instance of `shader` (`volumepassage`, ...).
pub fn image_dir(shader: &str) -> PathBuf {
    crate::wgsl::true_kingdom::default_dir().join("img").join(shader)
}

/// The fixed uniforms of a run: the host's defaults (`cuneus/examples/<shader>.rs`) and a fixed time.
pub fn case_bindings(shader: &str, time: f32, width: u32, height: u32) -> Result<Vec<BindingData>, String> {
    let time_block = {
        let mut b = Vec::new();
        b.extend_from_slice(&time.to_le_bytes());
        b.extend_from_slice(&0.016f32.to_le_bytes());
        b.extend_from_slice(&300u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b
    };
    match shader {
        // VolumeParams { speed, intensity, color1, color2, color3, gamma, zoom, 3 pads }
        "volumepassage" => Ok(vec![
            BindingData::Uniform { group: 0, binding: 0, bytes: time_block },
            BindingData::Output { group: 1, binding: 0 },
            BindingData::Uniform { group: 1, binding: 1, bytes: words_f32(&[1.0, 0.001, 0.1, 0.3, 0.7, 0.8, 0.4, 0.2, 1.0, 1.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0]) },
        ]),
        // NebulaParams: iterations 17, formuparam 0.52, volsteps 6, stepsize 0.31, zoom 5, tile 0.35, speed 0.02, brightness 0.00062,
        // dust_intensity 1, distfading 0.95, color_variation 0.51, n_boxes 10, rotation 1, depth 5, color_mode 1, pad, rotation_x/y 0,
        // click_state 0, scale 1, exposure 1.6, gamma 0.4, 7 pads, time_scale 1, visual_mode 0, 2 pads (+ the host's 3 trailing pads)
        "nebula" => {
            let mut params = Vec::new();
            let i = |v: i32| v.to_le_bytes();
            let f = |v: f32| v.to_le_bytes();
            for w in [
                i(17), f(0.52), i(6), f(0.31), f(5.0), f(0.35), f(0.020), f(0.00062), f(1.0), f(0.95), f(0.51), f(10.0), i(1), f(5.0), i(1), f(0.0),
                f(0.0), f(0.0), i(0), f(1.0), f(1.6), f(0.4), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(1.0), i(0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0),
            ] {
                params.extend_from_slice(&w);
            }
            Ok(vec![
                BindingData::Uniform { group: 0, binding: 0, bytes: time_block },
                BindingData::Output { group: 1, binding: 0 },
                BindingData::Uniform { group: 1, binding: 1, bytes: params },
                BindingData::ZeroStorage { group: 2, binding: 0, bytes: 3 * u64::from(width) * u64::from(height) * 4 },
            ])
        }
        other => Err(format!("no case for {other}")),
    }
}

/// Rows by name, with every table gap collected.
pub struct Rows<'a> {
    pub k: &'a TrueKingdom,
    gaps: RefCell<Vec<String>>,
}

impl<'a> Rows<'a> {
    pub fn new(k: &'a TrueKingdom) -> Rows<'a> {
        Rows { k, gaps: RefCell::new(Vec::new()) }
    }

    fn gap(&self, what: String) -> Tree {
        let mut g = self.gaps.borrow_mut();
        if !g.contains(&what) {
            g.push(what);
        }
        Tree { symbol: 0, const_index: None, kids: Vec::new() }
    }

    /// A node of row `name` with `kids`.
    pub fn n(&self, name: &str, kids: Vec<Tree>) -> Tree {
        match self.k.rows.iter().position(|r| r.name == name) {
            None => self.gap(format!("no row {name}")),
            Some(i) => {
                let want = self.k.rows[i].args.len();
                if want != kids.len() {
                    return self.gap(format!("{name} takes {want} arguments, given {}", kids.len()));
                }
                Tree { symbol: i as u32, const_index: None, kids }
            }
        }
    }

    /// A constant placeholder `name` whose `C_T` entry is `text`.
    pub fn c(&self, name: &str, text: &str) -> Tree {
        let Some(i) = self.k.rows.iter().position(|r| r.name == name) else { return self.gap(format!("no row {name}")) };
        match self.k.constants.get(&self.k.rows[i].out).and_then(|c| c.iter().position(|t| t == text)) {
            Some(j) => Tree { symbol: i as u32, const_index: Some(j as u32), kids: Vec::new() },
            None => self.gap(format!("{name} has no constant {text:?} in C_T of {}", self.k.type_name(self.k.rows[i].out))),
        }
    }

    /// The gaps found so far, or the empty list.
    pub fn gaps(&self) -> Vec<String> {
        self.gaps.borrow().clone()
    }

    // ---- scalars
    pub fn real(&self, text: &str) -> Tree {
        self.c("literal.real", text)
    }
    pub fn lit64(&self, text: &str) -> Tree {
        self.c("literal.lit64", text)
    }
    pub fn add(&self, a: Tree, b: Tree) -> Tree {
        self.n("arith.add", vec![a, b])
    }
    pub fn sub(&self, a: Tree, b: Tree) -> Tree {
        self.n("arith.sub", vec![a, b])
    }
    pub fn mul(&self, a: Tree, b: Tree) -> Tree {
        self.n("arith.mul", vec![a, b])
    }
    pub fn div(&self, a: Tree, b: Tree) -> Tree {
        self.n("arith.div", vec![a, b])
    }
    pub fn f1(&self, op: &str, a: Tree) -> Tree {
        self.n(&format!("arith.{op}"), vec![a])
    }
    pub fn max(&self, a: Tree, b: Tree) -> Tree {
        self.n("arith.max", vec![a, b])
    }

    // ---- vectors (`l` lanes 2, 3, 4)
    pub fn vop(&self, op: &str, l: u8, a: Tree, b: Tree) -> Tree {
        self.n(&format!("arith.{op}_v{l}"), vec![a, b])
    }
    pub fn vun(&self, op: &str, l: u8, a: Tree) -> Tree {
        self.n(&format!("arith.{op}_v{l}"), vec![a])
    }
    /// `scalar * vector`.
    pub fn smul(&self, l: u8, s: Tree, v: Tree) -> Tree {
        self.n(&format!("arith.mul_scalar_vec_v{l}"), vec![s, v])
    }
    pub fn splat(&self, l: u8, s: Tree) -> Tree {
        self.n(&format!("shape.splat{l}"), vec![s])
    }
    pub fn lane(&self, l: u8, i: u8, v: Tree) -> Tree {
        self.n(&format!("shape.access_{i}_v{l}"), vec![v])
    }
    pub fn swz(&self, l: u8, pat: &str, v: Tree) -> Tree {
        self.n(&format!("shape.swizzle_{pat}_v{l}"), vec![v])
    }
    pub fn v2(&self, x: Tree, y: Tree) -> Tree {
        self.n("shape.compose2", vec![x, y])
    }
    /// `vec3(xy, z)`.
    pub fn v3(&self, xy: Tree, z: Tree) -> Tree {
        self.n("shape.compose3", vec![xy, z])
    }
    /// `vec3(x, y, z)`.
    pub fn v3s(&self, x: Tree, y: Tree, z: Tree) -> Tree {
        let xy = self.v2(x, y);
        self.v3(xy, z)
    }
    pub fn v4(&self, xyz: Tree, w: Tree) -> Tree {
        self.n("shape.compose4", vec![xyz, w])
    }
    pub fn length(&self, l: u8, v: Tree) -> Tree {
        self.n(&format!("geom.length_v{l}"), vec![v])
    }

    // ---- registers
    pub fn rf(&self, i: usize) -> Tree {
        self.n(&format!("reg.rf{i}"), vec![])
    }
    pub fn gf(&self, i: usize) -> Tree {
        let r = self.rf(i);
        self.n("reg.get_f", vec![r])
    }
    /// A vec3 register (`C<i>`), a vec2 register (`w<i>`).
    pub fn r3(&self, i: usize) -> Tree {
        self.n(&format!("reg.C{i}"), vec![])
    }
    pub fn g3(&self, i: usize) -> Tree {
        let r = self.r3(i);
        self.n("reg.get_v3", vec![r])
    }
    pub fn r2(&self, i: usize) -> Tree {
        self.n(&format!("reg.w{i}"), vec![])
    }
    pub fn g2(&self, i: usize) -> Tree {
        let r = self.r2(i);
        self.n("reg.get_v2", vec![r])
    }

    /// A u32 register (`ru<i>`).
    pub fn ri(&self, i: usize) -> Tree {
        self.n(&format!("reg.ru{i}"), vec![])
    }
    pub fn gi(&self, i: usize) -> Tree {
        let r = self.ri(i);
        self.n("reg.get_i", vec![r])
    }
    pub fn set_i(&self, at: Rung, i: usize, v: Tree) -> Tree {
        let r = self.ri(i);
        self.n(&format!("reg.set_i_{}", at.suffix()), vec![r, v])
    }

    // ---- statements and blocks at rung `(x, d, k)`
    pub fn set_f(&self, at: Rung, i: usize, v: Tree) -> Tree {
        let r = self.rf(i);
        self.n(&format!("reg.set_f_{}", at.suffix()), vec![r, v])
    }
    pub fn set_3(&self, at: Rung, i: usize, v: Tree) -> Tree {
        let r = self.r3(i);
        self.n(&format!("reg.set_v3_{}", at.suffix()), vec![r, v])
    }
    pub fn set_2(&self, at: Rung, i: usize, v: Tree) -> Tree {
        let r = self.r2(i);
        self.n(&format!("reg.set_v2_{}", at.suffix()), vec![r, v])
    }

    /// A block of 1..=4 statements of rung `at`; more are nested through `stmt.block`.
    pub fn block(&self, at: Rung, mut stmts: Vec<Tree>) -> Tree {
        if stmts.len() > 4 {
            let rest = stmts.split_off(3);
            let inner = self.block(at, rest);
            stmts.push(self.n(&format!("stmt.block_{}", at.suffix()), vec![inner]));
        }
        let name = match stmts.len() {
            1 => "one",
            2 => "seq2",
            3 => "seq3",
            _ => "seq4",
        };
        self.n(&format!("block.{name}_{}", at.suffix()), stmts)
    }

    /// The epilogue: `leads` are C-rung statements (in `at`'s depth and stratum), `last` the store (an O-rung
    /// statement). At most three leads sit directly; the earlier ones are wrapped in one C block statement.
    pub fn epilogue(&self, d: u8, k: u8, mut leads: Vec<Tree>, last: Tree) -> Tree {
        let c = Rung { x: 'C', d, k };
        while leads.len() > 3 {
            let rest = leads.split_off(leads.len() - 2);
            let head = self.block(c, std::mem::take(&mut leads));
            let wrapped = self.n(&format!("stmt.block_{}", c.suffix()), vec![head]);
            leads = vec![wrapped];
            leads.extend(rest);
        }
        let name = match leads.len() {
            0 => "one",
            1 => "seq2",
            2 => "seq3",
            _ => "seq4",
        };
        leads.push(last);
        self.n(&format!("block.{name}_O{d}_f{k}"), leads)
    }
}

/// A statement rung: kind `x` (L, C, O or W), depth `d`, fuel stratum `k`.
#[derive(Debug, Clone, Copy)]
pub struct Rung {
    pub x: char,
    pub d: u8,
    pub k: u8,
}

impl Rung {
    pub fn suffix(&self) -> String {
        format!("{}{}_f{}", self.x, self.d, self.k)
    }
}

/// `volumepassage.wgsl` (cuneus `examples/shaders`, entry `main`): a 50-step ray march through a 7-octave sine noise,
/// tone mapped, one pixel per invocation, 8 x 8 workgroups. Returns the tree, or every table gap.
pub fn volumepassage(k: &TrueKingdom) -> Result<Tree, Vec<String>> {
    let b = Rows::new(k);
    let l = |d: u8, kk: u8| Rung { x: 'L', d, k: kk };
    let p = |name: &str| b.n(&format!("load.params.{name}"), vec![]);
    let real = |t: &str| b.real(t);
    // registers: rf0 z, rf1 d, rf2 N, rf3 i, rf4 t, rf5 den1, rf6 v, rf7 a, rf8 cos, rf9 sin, rf10 angle, rf11 n(q), rf12 r (pmap), rf13 den2, rf14 s
    //            C0 o, C1 p, C2 pos, C3 c   (vec3);  w0 C (pixel), w1 R (dims), w2 rotated xy, w3 q   (vec2)
    let (z, d, nn, i, t, den1, v, a, co, si, ang, nq, rm, den2, sc) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    let (o3, p3, pos3, c3) = (0usize, 1usize, 2usize, 3usize);
    let (cpix, rdim, rot, q2) = (0usize, 1usize, 2usize, 3usize);

    // ---- inner loop body (depth 2, stratum 0): v += n(pos.xy + pos.z * 0.5) * a; pos *= 2.0; a *= 0.5
    let r20 = l(2, 0);
    let q = b.vop("add", 2, b.swz(3, "xy", b.g3(pos3)), b.splat(2, b.mul(b.lane(3, 2, b.g3(pos3)), real("0.5"))));
    let qx = || b.lane(2, 0, b.g2(q2));
    let qy = || b.lane(2, 1, b.g2(q2));
    // n(q) = sin(q.x * 3.0 + sin(q.y * 2.7)) * cos(q.y * 1.1 + cos(q.x * 2.3))
    let n_of_q = b.mul(
        b.f1("sin", b.add(b.mul(qx(), real("3.0")), b.f1("sin", b.mul(qy(), real("2.7"))))),
        b.f1("cos", b.add(b.mul(qy(), real("1.1")), b.f1("cos", b.mul(qx(), real("2.3"))))),
    );
    let inner_body = b.block(
        r20,
        vec![
            b.set_2(r20, q2, q),
            b.set_f(r20, nq, n_of_q),
            b.set_f(r20, v, b.add(b.gf(v), b.mul(b.gf(nq), b.gf(a)))),
            b.set_3(r20, pos3, b.smul(3, real("2.0"), b.g3(pos3))),
            b.set_f(r20, a, b.mul(b.gf(a), real("0.5"))),
        ],
    );
    // the 7 octaves: a counted loop of a literal bound, depth 1 stratum 0 -> statement of stratum 1
    let inner = b.n("loop.tile_L1_f0", vec![b.lit64("7"), inner_body]);

    // ---- outer loop body (depth 1, stratum 1)
    let r11 = l(1, 1);
    let xy_p = || b.swz(3, "xy", b.g3(p3));
    let pz = || b.lane(3, 2, b.g3(p3));
    let color = |n: u8| b.v3s(p(&format!("color{n}_r")), p(&format!("color{n}_g")), p(&format!("color{n}_b")));
    let body = b.block(
        r11,
        vec![
            b.set_f(r11, i, b.add(b.gf(i), real("1.0"))),
            b.set_f(r11, z, b.add(b.gf(z), b.mul(b.gf(d), real("0.6")))),
            // p = z * normalize(vec3((C - 0.5 * r.xy) / zoom, r.y))
            b.set_3(
                r11,
                p3,
                b.smul(
                    3,
                    b.gf(z),
                    b.n(
                        "geom.normalize_v3",
                        vec![b.v3(b.vop("div", 2, b.vop("sub", 2, b.g2(cpix), b.smul(2, real("0.5"), b.g2(rdim))), b.splat(2, p("zoom"))), b.lane(2, 1, b.g2(rdim)))],
                    ),
                ),
            ),
            // p.z += t
            b.set_3(r11, p3, b.v3(xy_p(), b.add(pz(), b.gf(t)))),
            // let R_mat = mm2(p.z * 1.1)
            b.set_f(r11, ang, b.mul(pz(), real("1.1"))),
            b.set_f(r11, co, b.f1("cos", b.gf(ang))),
            b.set_f(r11, si, b.f1("sin", b.gf(ang))),
            // rotated_xy = R_mat * p.xy, with m2(c, -s, s, c)
            b.set_2(
                r11,
                rot,
                b.n(
                    "arith.mul_mat_vec_2x2",
                    vec![b.n("shape.compose_mat2x2", vec![b.v2(b.gf(co), b.f1("neg", b.gf(si))), b.v2(b.gf(si), b.gf(co))]), xy_p()],
                ),
            ),
            // p.x = rotated_xy.x; p.y = rotated_xy.y
            b.set_3(r11, p3, b.v3(b.g2(rot), pz())),
            // N = f(p + t * 0.1)
            b.set_3(r11, pos3, b.vop("add", 3, b.g3(p3), b.splat(3, b.mul(b.gf(t), real("0.1"))))),
            b.set_f(r11, v, real("0.0")),
            b.set_f(r11, a, real("1.0")),
            inner,
            b.set_f(r11, nn, b.gf(v)),
            // d = length(p.xy) - 1.0 + N * 0.3
            b.set_f(r11, d, b.add(b.sub(b.length(2, xy_p()), real("1.0")), b.mul(b.gf(nn), real("0.3")))),
            // p.z = pmap(p.z, 4.0): r = value % domain; select(r + domain, r, r >= 0.0) - (domain * 0.5)
            b.set_f(r11, rm, b.n("arith.rem", vec![pz(), real("4.0")])),
            b.set_3(
                r11,
                p3,
                b.v3(
                    xy_p(),
                    b.sub(
                        b.n("select.real", vec![b.add(b.gf(rm), real("4.0")), b.gf(rm), b.n("compare.ge_real", vec![b.gf(rm), real("0.0")])]),
                        b.mul(real("4.0"), real("0.5")),
                    ),
                ),
            ),
            // d = abs(d) + 0.01
            b.set_f(r11, d, b.add(b.f1("abs", b.gf(d)), real("0.01"))),
            // den1, den2
            b.set_f(r11, den1, b.max(b.mul(b.length(2, b.vop("add", 2, xy_p(), b.splat(2, b.gf(nn)))), real("0.8")), real("0.005"))),
            b.set_f(r11, den2, b.max(b.mul(b.length(2, b.vop("add", 2, b.swz(3, "xz", b.g3(p3)), b.splat(2, b.gf(nn)))), real("0.8")), real("0.005"))),
            // c = color1 / den1; c += color2 / den2; c += color3 * (0.5 + 0.5 * sin(N * 1.0 + t))
            b.set_3(r11, c3, b.vop("div", 3, color(1), b.splat(3, b.gf(den1)))),
            b.set_3(r11, c3, b.vop("add", 3, b.g3(c3), b.vop("div", 3, color(2), b.splat(3, b.gf(den2))))),
            b.set_f(r11, sc, b.add(real("0.5"), b.mul(real("0.5"), b.f1("sin", b.add(b.mul(b.gf(nn), real("1.0")), b.gf(t)))))),
            b.set_3(r11, c3, b.vop("add", 3, b.g3(c3), b.smul(3, b.gf(sc), color(3)))),
            // o += c / d * 0.12
            b.set_3(r11, o3, b.vop("add", 3, b.g3(o3), b.smul(3, real("0.12"), b.vop("div", 3, b.g3(c3), b.splat(3, b.gf(d)))))),
        ],
    );
    let outer = b.n("loop.tile_L0_f1", vec![b.lit64("50"), body]);

    // ---- preamble (depth 0, stratum 2)
    let r02 = l(0, 2);
    let gid = || b.n("builtin.global_invocation_id", vec![]);
    let px = b.n("convert.index_to_f32", vec![b.n("index.of_col", vec![b.n("axis.x_gid", vec![gid()])])]);
    let py = b.n("convert.index_to_f32", vec![b.n("index.of_row", vec![b.n("axis.y_gid", vec![gid()])])]);
    let w = b.n("convert.count_to_f32", vec![b.n("load.image_w_output", vec![])]);
    let h = b.n("convert.count_to_f32", vec![b.n("load.image_h_output", vec![])]);
    let pre = b.block(
        r02,
        vec![
            b.set_2(r02, rdim, b.v2(w, h)),
            b.set_2(r02, cpix, b.v2(px, py)),
            b.set_3(r02, o3, b.n("own.vec3", vec![])),
            b.set_f(r02, i, real("0.0")),
            b.set_f(r02, d, real("0.0")),
            b.set_f(r02, z, real("0.0")),
            b.set_f(r02, t, b.mul(b.n("load.time_data.time", vec![]), p("speed"))),
            b.set_f(r02, nn, real("0.0")),
            outer,
        ],
    );
    let entry_body = pre;

    // ---- epilogue (depth 0, stratum 2): tanh, ACES, max 0, gamma, store
    let c02 = Rung { x: 'C', d: 0, k: 2 };
    let mat3 = |cols: [[&str; 3]; 3]| b.n("shape.compose_mat3x3", cols.iter().map(|c| b.v3s(real(c[0]), real(c[1]), real(c[2]))).collect());
    let m1 = || mat3([["0.59719", "0.07600", "0.02840"], ["0.35458", "0.90834", "0.13383"], ["0.04823", "0.01566", "0.83777"]]);
    let m2 = || mat3([["1.60475", "-0.10208", "-0.00327"], ["-0.53108", "1.10813", "-0.07276"], ["-0.07367", "-0.00605", "1.07602"]]);
    let (res, av, aa, ab) = (4usize, 5usize, 6usize, 7usize);
    let s3 = |x: &str| b.splat(3, real(x));
    let epi = b.epilogue(
        0,
        2,
        vec![
            b.set_3(c02, res, b.vun("tanh", 3, b.smul(3, p("intensity"), b.g3(o3)))),
            // aces_tonemap: v = m1 * color; a = v * (v + 0.0245786) - 0.000090537; b = v * (0.983729 * v + 0.4329510) + 0.238081; m2 * (a / b)
            b.set_3(c02, av, b.n("arith.mul_mat_vec_3x3", vec![m1(), b.g3(res)])),
            b.set_3(c02, aa, b.vop("sub", 3, b.vop("mul", 3, b.g3(av), b.vop("add", 3, b.g3(av), s3("0.0245786"))), s3("0.000090537"))),
            b.set_3(
                c02,
                ab,
                b.vop("add", 3, b.vop("mul", 3, b.g3(av), b.vop("add", 3, b.smul(3, real("0.983729"), b.g3(av)), s3("0.4329510"))), s3("0.238081")),
            ),
            b.set_3(c02, res, b.n("arith.mul_mat_vec_3x3", vec![m2(), b.vop("div", 3, b.g3(aa), b.g3(ab))])),
            // result = max(result, vec3(0.0)); result = pow(result, vec3(1.0 / gamma))
            b.set_3(c02, res, b.vop("max", 3, b.g3(res), b.n("own.vec3", vec![]))),
            b.set_3(c02, res, b.vop("pow", 3, b.g3(res), b.splat(3, b.div(real("1.0"), p("gamma"))))),
        ],
        b.n(
            "store.output_O0_f2",
            vec![b.n("own.u32_threadrow_output", vec![]), b.n("own.u32_threadcol_output", vec![]), b.v4(b.g3(res), real("1.0"))],
        ),
    );

    let grid = b.n("grid.of", vec![b.c("literal.shape", "8 8"), b.c("literal.blk_col", "1"), b.c("literal.blk_row", "1"), b.n("own.u32_ktile", vec![])]);
    let root = b.n("kernel.entry_safe", vec![grid, entry_body, epi]);
    let gaps = b.gaps();
    if gaps.is_empty() {
        Ok(root)
    } else {
        Err(gaps)
    }
}

/// `nebula.wgsl` (entry `main_image`; the host's first stage `volumetric_render` is a bare `return`): Kali's volumetric
/// nebula, `volsteps` ray steps of `iterations` folds each, graded, tone mapped, gamma. `mainVR`, `grade`, `aces` and
/// `gamma` are inlined. The two atomic reads `re` and `be`, `dir` and `ro2` of `main_image` and `dn` of `mainVR` have no
/// effect on the picture and are kept: the tree is the original's, not a simplification of it.
pub fn nebula(k: &TrueKingdom) -> Result<Tree, Vec<String>> {
    let b = Rows::new(k);
    let l = |d: u8, kk: u8| Rung { x: 'L', d, k: kk };
    let p = |name: &str| b.n(&format!("load.params.{name}"), vec![]);
    let real = |t: &str| b.real(t);
    let gid = || b.n("builtin.global_invocation_id", vec![]);
    let s3 = |x: &str| b.splat(3, real(x));
    let mix = |x: Tree, y: Tree, t: Tree| b.n("arith.mix", vec![x, y, t]);
    let mix3 = |x: Tree, y: Tree, t: Tree| b.n("arith.mix_v3", vec![x, y, t]);
    let smooth = |e0: Tree, e1: Tree, x: Tree| b.n("arith.smoothstep", vec![e0, e1, x]);
    let dot3 = |x: Tree, y: Tree| b.n("geom.dot_v3", vec![x, y]);
    // registers: f: 0 bi, 1 re, 2 be, 3 time, 4 t, 5 a1, 6 a2, 7 c1, 8 s1, 9 c2, 10 s2, 11 dc, 12 fd, 13 dd, 14 dof, 15 s, 16 fade, 17 pa, 18 a,
    //            19 pm, 20 df, 21 dn, 22 df2, 23 cp, 24 hb, 25 vb, 26 lum;  u32: 0 pi, 1 co
    //            v3: 0 color, 1 dir(main), 2 ro2, 3 dir, 4 ray, 5 v, 6 p, 7 dc2, 8 col, 9 bf, 10 dl, 11 ml, 12 lb, 13 graded/fc2, 14 tmp;  v2: 0 res, 1 fc, 2 uv, 3 uv(VR)
    let (bi, re, be, time, t, a1, a2, c1, s1, c2, s2, dc, fd, dd, dof, s, fade, pa, a) = (0usize, 1usize, 2usize, 3usize, 4usize, 5usize, 6usize, 7usize, 8usize, 9usize, 10usize, 11usize, 12usize, 13usize, 14usize, 15usize, 16usize, 17usize, 18usize);
    let (pm, df, dn, df2, cp, hb, vb, lum) = (19usize, 20usize, 21usize, 22usize, 23usize, 24usize, 25usize, 26usize);
    let (color, dir_main, ro2, dir, ray, v, pp, dc2, col, bf, dl, ml, lb, fc2) = (0usize, 1usize, 2usize, 3usize, 4usize, 5usize, 6usize, 7usize, 8usize, 9usize, 10usize, 11usize, 12usize, 13usize);
    let (res, fc, uvm, uv) = (0usize, 1usize, 2usize, 3usize);
    let (pi, co) = (0usize, 1usize);
    let xy3 = |i: usize| b.swz(3, "xy", b.g3(i));
    let z3 = |i: usize| b.lane(3, 2, b.g3(i));
    let rot_m = |c: usize, sn: usize| b.n("shape.compose_mat2x2", vec![b.v2(b.gf(c), b.f1("neg", b.gf(sn))), b.v2(b.gf(sn), b.gf(c))]);
    let rotate = |m: Tree, i: usize| b.v3(b.n("arith.mul_mat_vec_2x2", vec![m, xy3(i)]), z3(i));
    // uv = fragCoord / res - 0.5; uv.y *= res.y / res.x
    let uv_of = || b.vop("sub", 2, b.vop("div", 2, b.g2(fc), b.g2(res)), b.splat(2, real("0.5")));
    let fix_y = |i: usize| b.v2(b.lane(2, 0, b.g2(i)), b.mul(b.lane(2, 1, b.g2(i)), b.div(b.lane(2, 1, b.g2(res)), b.lane(2, 0, b.g2(res)))));

    // ---- the fold loop (depth 2, stratum 0)
    let r20 = l(2, 0);
    let inner_body = b.block(
        r20,
        vec![
            b.set_f(r20, pm, b.add(real("2.0"), b.mul(b.f1("sin", b.mul(b.length(3, b.g3(pp)), real("0.5"))), real("0.3")))),
            b.set_3(
                r20,
                pp,
                b.vop(
                    "sub",
                    3,
                    b.vop("div", 3, b.vun("abs", 3, b.g3(pp)), b.splat(3, b.n("arith.pow", vec![dot3(b.g3(pp), b.g3(pp)), b.mul(b.gf(pm), real("0.5"))]))),
                    b.splat(3, p("formuparam")),
                ),
            ),
            b.set_f(r20, a, b.add(b.gf(a), b.f1("abs", b.sub(b.length(3, b.g3(pp)), b.gf(pa))))),
            b.set_f(r20, pa, b.length(3, b.g3(pp))),
        ],
    );
    let inner = b.n("loop.tileu_6_L1_f0", vec![b.n("bound.of_uniform_6", vec![b.n("uniform.params.iterations", vec![])]), inner_body]);

    // ---- the ray-step loop (depth 1, stratum 1)
    let r11 = l(1, 1);
    let body = b.block(
        r11,
        vec![
            // p = ray + s * dir * 0.5
            b.set_3(r11, pp, b.vop("add", 3, b.g3(ray), b.smul(3, real("0.5"), b.smul(3, b.gf(s), b.g3(dir))))),
            // p = abs(vec3(tile) - (p * (vec3(tile) * 1.0)))
            b.set_3(r11, pp, b.vun("abs", 3, b.vop("sub", 3, b.splat(3, p("tile")), b.vop("mul", 3, b.g3(pp), b.smul(3, real("1.0"), b.splat(3, p("tile"))))))),
            b.set_f(r11, pa, real("0.0")),
            b.set_f(r11, a, real("0.0")),
            inner,
            // a *= a * a
            b.set_f(r11, a, b.mul(b.gf(a), b.mul(b.gf(a), b.gf(a)))),
            b.set_f(r11, df, smooth(real("0.3"), real("1.0"), b.gf(dof))),
            // a *= 2.0 * mix(0.2, 1.0, df); fade *= mix(0.7, 1.0, df)
            b.set_f(r11, a, b.mul(b.gf(a), b.mul(real("2.0"), mix(real("0.2"), real("1.0"), b.gf(df))))),
            b.set_f(r11, fade, b.mul(b.gf(fade), mix(real("0.7"), real("1.0"), b.gf(df)))),
            // dn = sin(p.x * 0.5 + time * 0.1) * cos(p.y * 0.3) * sin(p.z * 0.4)   (unused)
            b.set_f(
                r11,
                dn,
                b.mul(
                    b.mul(b.f1("sin", b.add(b.mul(b.lane(3, 0, b.g3(pp)), real("0.5")), b.mul(b.gf(time), real("0.1")))), b.f1("cos", b.mul(b.lane(3, 1, b.g3(pp)), real("0.3")))),
                    b.f1("sin", b.mul(z3(pp), real("0.4"))),
                ),
            ),
            b.set_f(r11, df2, b.max(real("0.0"), p("dust_intensity"))),
            b.set_3(r11, dc2, b.v3s(real("0.3"), real("0.7"), real("0.7"))),
            // cp = a * color_variation * 2.0 + time * 0.5
            b.set_f(r11, cp, b.add(b.mul(b.mul(b.gf(a), p("color_variation")), real("2.0")), b.mul(b.gf(time), real("0.5")))),
            b.set_3(
                r11,
                col,
                b.v3s(
                    b.add(real("0.8"), b.mul(b.f1("sin", b.gf(cp)), real("0.4"))),
                    b.add(real("0.6"), b.mul(b.f1("sin", b.add(b.gf(cp), real("2.0"))), real("0.1"))),
                    b.add(real("0.9"), b.mul(b.f1("sin", b.add(b.gf(cp), real("4.0"))), real("0.3"))),
                ),
            ),
            b.set_f(r11, hb, b.add(b.mul(b.f1("sin", b.add(b.mul(b.lane(3, 0, b.g3(pp)), real("0.5")), real("0.3"))), real("0.2")), real("1.0"))),
            b.set_f(r11, vb, b.add(b.mul(b.f1("cos", b.add(b.mul(b.lane(3, 1, b.g3(pp)), real("0.4")), real("0.25"))), real("0.2")), real("1.0"))),
            b.set_3(r11, bf, b.v3s(b.gf(hb), b.gf(vb), mix(b.gf(hb), b.gf(vb), real("0.5")))),
            // dl = vec3(fade) * dc2 * df2
            b.set_3(r11, dl, b.smul(3, b.gf(df2), b.vop("mul", 3, b.splat(3, b.gf(fade)), b.g3(dc2)))),
            // ml = vec3(s, s * s, s * s * s) * a * brightness * fade * col
            b.set_3(
                r11,
                ml,
                b.vop(
                    "mul",
                    3,
                    b.smul(3, b.gf(fade), b.smul(3, p("brightness"), b.smul(3, b.gf(a), b.v3s(b.gf(s), b.mul(b.gf(s), b.gf(s)), b.mul(b.mul(b.gf(s), b.gf(s)), b.gf(s)))))),
                    b.g3(col),
                ),
            ),
            b.set_3(r11, lb, b.vop("mul", 3, b.g3(bf), b.v3s(real("0.6"), real("0.6"), real("0.4")))),
            b.set_3(r11, v, b.vop("add", 3, b.g3(v), b.smul(3, b.lane(3, 0, b.g3(lb)), b.g3(dl)))),
            b.set_3(r11, v, b.vop("add", 3, b.g3(v), b.vop("mul", 3, b.g3(ml), b.g3(lb)))),
            // fade *= distfading * mix(0.95, 1.05, bf.z)
            b.set_f(r11, fade, b.mul(b.gf(fade), b.mul(p("distfading"), mix(real("0.95"), real("1.05"), b.lane(3, 2, b.g3(bf)))))),
            b.set_f(r11, s, b.add(b.gf(s), p("stepsize"))),
        ],
    );
    let outer = b.n("loop.tileu_6_L0_f1", vec![b.n("bound.of_uniform_6", vec![b.n("uniform.params.volsteps", vec![])]), body]);

    // ---- main_image, then mainVR, in order (depth 0, stratum 2)
    let r02 = l(0, 2);
    let w = b.n("convert.count_to_f32", vec![b.n("load.image_w_output", vec![])]);
    let h = b.n("convert.count_to_f32", vec![b.n("load.image_h_output", vec![])]);
    let id_x = || b.n("index.of_col", vec![b.n("axis.x_gid", vec![gid()])]);
    let id_y = || b.n("index.of_row", vec![b.n("axis.y_gid", vec![gid()])]);
    let atomic_f = |at: Tree| b.n("convert.count_to_f32", vec![b.n("load.atomic_buffer", vec![at])]);
    let stmts = vec![
        b.set_2(r02, res, b.v2(w, h)),
        // pi = id.x + u32(res.x) * id.y; co = u32(res.x * res.y)
        b.set_i(r02, pi, b.n("index.add", vec![id_x(), b.n("index.mul", vec![b.n("convert.f32_to_index", vec![b.lane(2, 0, b.g2(res))]), id_y()])])),
        b.set_i(r02, co, b.n("convert.f32_to_index", vec![b.mul(b.lane(2, 0, b.g2(res)), b.lane(2, 1, b.g2(res)))])),
        b.set_f(r02, bi, b.mul(atomic_f(b.gi(pi)), real("0.001"))),
        b.set_f(r02, re, b.mul(atomic_f(b.n("index.add", vec![b.gi(pi), b.gi(co)])), real("0.01"))),
        b.set_f(r02, be, b.mul(atomic_f(b.n("index.add", vec![b.gi(pi), b.n("index.mul", vec![b.gi(co), b.c("literal.index", "2")])])), real("0.01"))),
        b.set_3(r02, color, b.v3s(b.gf(bi), b.mul(b.gf(bi), real("0.7")), b.gf(bi))),
        b.set_2(r02, fc, b.v2(b.n("convert.index_to_f32", vec![id_x()]), b.n("convert.index_to_f32", vec![id_y()]))),
        b.set_2(r02, uvm, uv_of()),
        b.set_2(r02, uvm, fix_y(uvm)),
        b.set_3(r02, dir_main, b.v3(b.smul(2, p("zoom"), b.g2(uvm)), real("1.0"))),
        b.set_3(r02, ro2, b.v3s(real("1.0"), real("0.5"), real("0.5"))),
        b.set_f(r02, time, b.mul(b.n("load.u_time.time", vec![]), p("time_scale"))),
        // ---- mainVR
        b.set_2(r02, uv, uv_of()),
        b.set_2(r02, uv, fix_y(uv)),
        b.set_3(r02, dir, b.v3(b.smul(2, p("zoom"), b.g2(uv)), real("1.0"))),
        b.set_f(r02, t, b.add(b.mul(b.gf(time), p("speed")), real("0.25"))),
        b.set_f(r02, a1, b.add(real("0.5"), b.mul(b.div(p("rotation_x"), real("800.0")), real("2.0")))),
        b.set_f(r02, a2, b.add(real("0.8"), b.mul(b.div(p("rotation_y"), real("600.0")), real("2.0")))),
        b.set_f(r02, s1, b.f1("sin", b.gf(a1))),
        b.set_f(r02, c1, b.f1("cos", b.gf(a1))),
        b.set_f(r02, s2, b.f1("sin", b.gf(a2))),
        b.set_f(r02, c2, b.f1("cos", b.gf(a2))),
        b.set_3(r02, dir, rotate(rot_m(c1, s1), dir)),
        b.set_3(r02, dir, rotate(rot_m(c2, s2), dir)),
        b.set_3(r02, ray, b.v3s(real("1.0"), real("0.5"), real("0.5"))),
        b.set_3(r02, ray, b.vop("add", 3, b.g3(ray), b.v3s(b.mul(b.gf(t), real("2.0")), b.gf(t), real("-2.0")))),
        b.set_3(r02, ray, rotate(rot_m(c1, s1), ray)),
        b.set_3(r02, ray, rotate(rot_m(c2, s2), ray)),
        b.set_f(r02, dc, b.length(2, xy3(dir))),
        b.set_f(r02, fd, real("0.0")),
        b.set_f(r02, dd, b.f1("abs", b.sub(b.gf(dc), b.gf(fd)))),
        b.set_f(r02, dof, b.sub(real("1.0"), smooth(real("0.0"), real("0.0"), b.mul(b.gf(dd), b.gf(dd))))),
        b.set_f(r02, s, real("0.1")),
        b.set_f(r02, fade, real("1.0")),
        b.set_3(r02, v, b.n("own.vec3", vec![])),
        outer,
        // v = mix(vec3(length(v)), v, 1.0); fc2 = v * 0.03 + color * 0.05; fc2 *= exposure
        b.set_3(r02, v, mix3(b.splat(3, b.length(3, b.g3(v))), b.g3(v), s3("1.0"))),
        b.set_3(r02, fc2, b.vop("add", 3, b.smul(3, real("0.03"), b.g3(v)), b.smul(3, real("0.05"), b.g3(color)))),
        b.set_3(r02, fc2, b.smul(3, p("exposure"), b.g3(fc2))),
        // grade
        b.set_3(r02, fc2, b.vop("pow", 3, b.g3(fc2), s3("0.9"))),
        b.set_3(r02, fc2, b.vop("mul", 3, b.g3(fc2), b.v3s(real("1.1"), real("1.05"), real("0.95")))),
        b.set_3(r02, fc2, mix3(b.g3(fc2), b.vop("mul", 3, b.g3(fc2), b.g3(fc2)), s3("0.3"))),
        b.set_f(r02, lum, dot3(b.g3(fc2), b.v3s(real("0.299"), real("0.587"), real("0.114")))),
        b.set_3(r02, fc2, mix3(b.splat(3, b.gf(lum)), b.g3(fc2), s3("1.2"))),
    ];
    let entry_body = b.block(r02, stmts);

    // ---- epilogue: aces = clamp((c * (a c + b)) / (c * (cc c + d) + e), 0, 1); gamma; store
    let c02 = Rung { x: 'C', d: 0, k: 2 };
    let num = b.vop("mul", 3, b.g3(fc2), b.vop("add", 3, b.smul(3, real("2.51"), b.g3(fc2)), s3("0.03")));
    let den = b.vop("add", 3, b.vop("mul", 3, b.g3(fc2), b.vop("add", 3, b.smul(3, real("2.43"), b.g3(fc2)), s3("0.59"))), s3("0.14"));
    let zero3 = || b.n("own.vec3", vec![]);
    let epi = b.epilogue(
        0,
        2,
        vec![
            b.set_3(c02, fc2, b.vop("min", 3, b.vop("max", 3, b.vop("div", 3, num, den), zero3()), s3("1.0"))),
            b.set_3(c02, fc2, b.vop("pow", 3, b.vop("max", 3, b.g3(fc2), zero3()), b.splat(3, b.div(real("1.0"), p("gamma"))))),
        ],
        b.n("store.output_O0_f2", vec![b.n("own.u32_threadrow_output", vec![]), b.n("own.u32_threadcol_output", vec![]), b.v4(b.g3(fc2), real("1.0"))]),
    );
    let grid = b.n("grid.of", vec![b.c("literal.shape", "16 16"), b.c("literal.blk_col", "1"), b.c("literal.blk_row", "1"), b.n("own.u32_ktile", vec![])]);
    let root = b.n("kernel.entry_safe", vec![grid, entry_body, epi]);
    let gaps = b.gaps();
    if gaps.is_empty() {
        Ok(root)
    } else {
        Err(gaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::{express, write_back, GeneLayout};
    use crate::wgsl::true_kingdom::{random_gene, Rng, Share};

    fn functions(t: &Tree, k: &TrueKingdom) -> usize {
        usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
    }

    /// Each original is a tree of rows with no table gap; the write-back puts it in a gene at the smallest program size
    /// that holds it and the typed walk reads the same tree back; it renders and validates in naga for the installed target.
    fn round_trips(shader: &str, build: fn(&TrueKingdom) -> Result<Tree, Vec<String>>, nodes: usize) {
        let k = TrueKingdom::load(&image_dir(shader)).expect("table");
        let c = k.relations().check().expect("kingdom check");
        let tree = build(&k).unwrap_or_else(|g| panic!("table gaps: {g:?}"));
        assert_eq!(tree.level_order().len(), nodes);
        let n = functions(&tree, &k);
        let layout = GeneLayout::from_program_size(&c, k.root, n).expect("layout");
        let const_len = k.const_lens();
        let previous = random_gene(&c, &layout, &const_len, &mut Rng(3), Share::Half, true);
        let gene = write_back(&c, &layout, &previous, &tree).expect("write-back");
        assert_eq!(express(&c, &layout, &gene, &const_len).expect("express").tree(), tree);
        let src = k.render(&tree).expect("render");
        let module = crate::wgsl::naga_shape::parse_unscanned(&src).expect("naga parse");
        crate::wgsl::target::Target::current().validate(&module, &src).expect("naga validate");
        assert!(src.contains("textureStore(output"));
    }

    #[test]
    fn volumepassage_is_a_tree_of_rows_that_round_trips_through_a_gene() {
        round_trips("volumepassage", volumepassage, 492);
    }

    #[test]
    fn nebula_is_a_tree_of_rows_that_round_trips_through_a_gene() {
        round_trips("nebula", nebula, 878);
    }

    /// The first node of row `name` in the tree, depth first.
    fn find<'t>(t: &'t Tree, k: &TrueKingdom, name: &str) -> Option<&'t Tree> {
        if k.rows[t.symbol as usize].name == name {
            return Some(t);
        }
        t.kids.iter().find_map(|c| find(c, k, name))
    }

    /// The work bound of the encoded volumepassage: the 50-trip march (`loop.tile_L0_f1`) holds the 7-octave noise
    /// (`loop.tile_L1_f0`); every node outside the march counts once, the march's body 50 times, the noise's body 350
    /// times; each loop's literal bound is outside its own loop. The bound is far under the kingdom's `H * 2^(6K)`.
    #[test]
    fn the_work_bound_of_the_encoded_volumepassage_is_its_loops_trips_times_their_bodies() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let tree = volumepassage(&k).unwrap_or_else(|g| panic!("table gaps: {g:?}"));
        let size = |t: &Tree| t.level_order().len() as u128;
        let outer = find(&tree, &k, "loop.tile_L0_f1").expect("the march");
        let inner = find(outer, &k, "loop.tile_L1_f0").expect("the noise");
        let (outer_body, inner_body) = (size(&outer.kids[1]), size(&inner.kids[1]));
        let want = (size(&tree) - outer_body) + 50 * (outer_body - inner_body) + 350 * inner_body;
        let got = k.tree_work_bound(&tree).expect("bound");
        assert_eq!(got, want, "outer body {outer_body} nodes, inner body {inner_body}");
        let c = k.relations().check().expect("kingdom check");
        let h = k.head_total(&c, functions(&tree, &k)).expect("H");
        eprintln!("volumepassage: {} nodes, bound {got} an invocation (H * 2^12 = {}); 256 x 256: {} in one dispatch", size(&tree), u128::from(h) << 12, got * 65_536);
        assert!(got < u128::from(h) << 12);
    }

    /// The encoded volumepassage drawn in one dispatch (proven: its loops are the original's) and drawn tiled, in submits
    /// of three whole workgroups under a cap lowered to force the split (the tree's own bound), give the same picture,
    /// and both are the original shader's, for a picture the 8 x 8 grid divides and for one it does not.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_tiled_dispatch_draws_the_same_picture_as_one_dispatch_on_the_device() {
        use crate::wgsl::image::{compare, ImageKernel};
        use crate::wgsl::oracle::device::{Device, Fuel};
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let tree = volumepassage(&k).unwrap_or_else(|g| panic!("table gaps: {g:?}"));
        let src = k.render(&tree).expect("render");
        let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).expect("the original");
        let per = k.tree_work_bound(&tree).expect("bound");
        let dev = Device::new().expect("device");
        for (w, h) in [(64u32, 64u32), (60, 44)] {
            let bindings = case_bindings("volumepassage", 5.0, w, h).expect("bindings");
            let reference = ImageKernel::compile(&dev, "volumepassage", &original, "main", &bindings, w, h, Fuel::Proven("cuneus original volumepassage")).expect("the original").run(0, 1).expect("run");
            let one = ImageKernel::compile(&dev, "encoded", &src, "main", &bindings, w, h, Fuel::Proven("the encoded original: its loops are the original's 50 x 7")).expect("one dispatch");
            assert_eq!(one.submits(), 1);
            let tiled = ImageKernel::compile_under(&dev, "encoded", &src, "main", &bindings, w, h, Fuel::Tree { per_invocation: per }, per * 64 * 3).expect("tiled");
            let (gx, gy, _) = tiled.dispatch();
            assert_eq!(tiled.submits(), (gy * gx.div_ceil(3)) as usize, "{w}x{h}: three workgroups a submit, row by row");
            let a = one.run(0, 1).expect("run");
            let b = tiled.run(0, 1).expect("run");
            assert_eq!(b.submits, tiled.submits());
            let d = compare(&b.picture, &a.picture).expect("same size");
            assert_eq!(d.differing_channels, 0, "{w}x{h}: tiled against one dispatch: {d:?}");
            let o = compare(&a.picture, &reference.picture).expect("same size");
            assert_eq!(o.differing_channels, 0, "{w}x{h}: encoded against the original: {o:?}");
        }
    }

    #[test]
    fn a_missing_row_is_a_named_gap_never_a_substitute() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let b = Rows::new(&k);
        let _ = b.n("arith.no_such_row", vec![]);
        let _ = b.c("literal.real", "0.123456789");
        let gaps = b.gaps();
        assert_eq!(gaps.len(), 2, "{gaps:?}");
        assert!(gaps[0].starts_with("no row") && gaps[1].contains("no constant"));
    }
}
