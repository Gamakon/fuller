//! SNAP FOR THE WGSL KINGDOM: a float literal within `rel_tol` of a library constant ([`snap_library`]) but not equal to
//! it takes the `C_T` entry that IS that constant, appended to `C_T` when `C_T` lacks it (append-only for the run's
//! epoch; the caller rebuilds what reads `C_T`). It reads the float literal rows of a [`TrueKingdom`] (the data type
//! `f32` of the rows, `f32.*` of an older table) and nothing else, so it works on genes from the rows.

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::Ty;
use crate::typed_gep::Tree;
use crate::wgsl::true_kingdom::{RowKind, TrueKingdom};

/// A float literal row and the f32 value of each entry of its `C_T`.
#[derive(Debug, Clone)]
struct FloatConst {
    row: u32,
    values: Vec<Option<f32>>,
}

/// What a snap did: literals snapped, and of them those whose exact value was appended to `C_T`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SnapCounts {
    pub snapped: u32,
    pub appended: u32,
}

/// The float literal rows of a kingdom, read once.
#[derive(Debug, Clone)]
pub struct Snap {
    consts: BTreeMap<Ty, FloatConst>,
}

impl Snap {
    /// The float literal rows of `k`: every constant row of a float type whose `C_T` entries all parse as floats.
    pub fn new(k: &TrueKingdom) -> Snap {
        let mut consts = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            let name = k.type_name(r.out);
            if r.kind != RowKind::Constant || !(name == "f32" || name.starts_with("f32.")) {
                continue;
            }
            let Some(texts) = k.constants.get(&r.out) else { continue };
            let values: Vec<Option<f32>> = texts.iter().map(|t| t.trim().parse::<f64>().ok().map(|v| v as f32)).collect();
            if values.iter().all(Option::is_some) {
                consts.entry(r.out).or_insert(FloatConst { row: i as u32, values });
            }
        }
        Snap { consts }
    }

    fn value_of(&self, t: &Tree) -> Option<(Ty, f32)> {
        let i = t.const_index? as usize;
        self.consts.iter().find(|(_, c)| c.row == t.symbol).and_then(|(ty, c)| c.values.get(i).copied().flatten().map(|v| (*ty, v)))
    }

    /// Float literals near a library constant snapped to that constant. A literal that already is a library value is
    /// left alone; of the values within `rel_tol` the nearest is taken. Its `C_T` entry is used when `C_T` holds it bit
    /// for bit, and otherwise appended to `C_T`. `None` when nothing snaps.
    pub fn snap(&self, k: &mut TrueKingdom, tree: &Tree, rel_tol: f64) -> (Option<Tree>, SnapCounts) {
        let library = snap_library();
        let mut counts = SnapCounts::default();
        fn index_of(k: &TrueKingdom, ty: Ty, v: f32) -> Option<u32> {
            k.constants.get(&ty)?.iter().position(|t| t.trim().parse::<f64>().ok().is_some_and(|x| (x as f32).to_bits() == v.to_bits())).map(|i| i as u32)
        }
        fn go(o: &Snap, k: &mut TrueKingdom, t: &Tree, lib: &[(String, f64)], tol: f64, counts: &mut SnapCounts) -> Tree {
            if let Some((ty, v)) = o.value_of(t) {
                let is_library = lib.iter().any(|(_, s)| (*s as f32).to_bits() == v.to_bits());
                if v.is_finite() && v != 0.0 && !is_library {
                    let v64 = f64::from(v);
                    let near = lib.iter().filter(|(_, s)| (v64 - s).abs() <= tol * s.abs()).min_by(|a, b| (v64 - a.1).abs().total_cmp(&(v64 - b.1).abs()));
                    if let Some((_, s)) = near {
                        let exact = *s as f32;
                        let i = match index_of(k, ty, exact) {
                            Some(i) => i,
                            None => {
                                let v = k.constants.entry(ty).or_default();
                                v.push(format!("{exact:?}"));
                                counts.appended += 1;
                                v.len() as u32 - 1
                            }
                        };
                        counts.snapped += 1;
                        return Tree { symbol: t.symbol, const_index: Some(i), kids: Vec::new() };
                    }
                }
            }
            Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| go(o, k, c, lib, tol, counts)).collect() }
        }
        let out = go(self, k, tree, &library, rel_tol, &mut counts);
        (if out == *tree { None } else { Some(out) }, counts)
    }
}

/// The snap library: pi, e, sqrt 2, tau, 1/sqrt 2, ln 2, 2 pi, pi/2, and every rational p/q with q <= 12 and
/// |p/q| <= 64. The literals of a WGSL kernel are entries of a finite `C_T`, so a constant is a near miss only against
/// a short exact value it could have been (0.59719 against 3/5, 1.60475 against 8/5).
pub fn snap_library() -> Vec<(String, f64)> {
    use std::f64::consts::{E, FRAC_1_SQRT_2, FRAC_PI_2, LN_2, PI, SQRT_2, TAU};
    let mut out: Vec<(String, f64)> = [("pi", PI), ("e", E), ("sqrt2", SQRT_2), ("tau", TAU), ("1/sqrt2", FRAC_1_SQRT_2), ("ln2", LN_2), ("pi/2", FRAC_PI_2)].iter().map(|(n, v)| (n.to_string(), *v)).collect();
    let mut seen: BTreeSet<u64> = out.iter().map(|(_, v)| v.to_bits()).collect();
    for q in 1..=12i64 {
        for p in -64 * q..=64 * q {
            let v = p as f64 / q as f64;
            if p != 0 && seen.insert(v.to_bits()) {
                out.push((format!("{p}/{q}"), v));
            }
        }
    }
    out
}
