//! Snap on the device, stage 1: the constant lattice as a sorted table, and
//! the match — a numeric literal to the lattice entry it is within tolerance
//! of, or none.
//!
//! Snap searches the NUMBERS evolution and the least-squares wrap produce
//! against the lattice of known constant forms (`snap_karva::lattice`); named
//! constants are never planted as drawable terminals. This file is the table
//! and the search. The graft, the R² guard and the write-back into the gene
//! are later stages and build on the layout fixed here:
//!
//!   values[n]       f32   |value| of each entry, strictly ascending
//!   info[n × 4]     u32   (template offset in nodes, node count, flags, family)
//!   templates[..]   u32   the entries' `math` forms, 4 words per node in the
//!                         linter's node layout: (op, arg0, arg1, f32 bits)
//!
//! An entry id is the entry's index in `values`, and indexes `info`.
//!
//! WHICH ENTRY. A literal `c` is within tolerance of a constant `t` iff
//! `|c - t| / |t| <= rel_tol` (against `t == 0`, iff `|c| <= rel_tol`) —
//! `snap.rs::best_match`'s band. The band around a literal can hold several
//! forms (up to 6 at 1e-3, 9 at 5e-3), and the nearest is often an
//! obscure coincidence: 3.14 is nearer `(2*e)/(3*gamma)` than `pi`. The choice
//! among them is multi-objective — nearness, shortness, fit to the literal's
//! context — and this project's mechanism for that is HFF with the TrueNorth
//! pole. Per candidate, three objectives in [0, 1], smaller is better:
//!
//!   x_near = relative error / rel_tol
//!   x_size = template nodes / the table's largest template
//!   x_fit  = `SnapAffinity` [the literal's CONTEXT] [the entry's FAMILY]
//!
//! and the smallest TrueNorth angle wins, ties to the lower entry. The angle is
//! `acos(1 - min(sum(x^2) / 3, 1))`, monotone in the energy `sum(x^2)`: the F64
//! search asks `score::truenorth_angle` (hff's own function), the device and
//! its F32 twin take the argmin of the energy, and a test holds the two orders
//! together. The affinity table is DATA (`SnapTable::affinity`), uploaded, never
//! a constant in the WGSL.
//!
//! The band is scanned with a FIXED bound: `BAND` entries each side of the
//! insertion point. A band that reaches past it is DETECTED (`Found::
//! BandOverflow`, counted by the callers), never truncated: the literal gets no
//! entry. At the supported tolerances (<= 5e-3) it cannot happen — a test
//! measures the widest band the table can produce.
//!
//! Every constant is tried as itself and negated (`snap_karva::snap_variants`
//! tries an entry only as itself; this is a superset). A constant of the
//! opposite sign has a relative error of at least 1, so under the
//! `rel_tol <= 0.5` this module accepts it never qualifies: the table holds
//! magnitudes and the sign travels as one bit.
//!
//! The device proposes in f32. Nothing it proposes leaves the device as a
//! constant until `confirm_f64` has re-derived the decision in f64.

use std::collections::BTreeSet;

use super::engine::LitMode;
use super::flat::{Flat, LNode};
use super::node::Tree;
use crate::gpu_eval::Op;
use crate::snap_karva::ConstEntry;

pub const SNAP_WGSL: &str = include_str!("snap.wgsl");

/// "No entry" on the device.
pub const NONE: u32 = u32::MAX;
/// Largest template the table carries. The linter's slot is 64 nodes and its
/// work area 80, which leaves room for 15 added nodes.
pub const TEMPLATE_MAX: usize = 15;
/// Words per entry in `info`.
pub const INFO_STRIDE: usize = 4;
/// `info` flag: the entry's template evaluates to MINUS its table value (the
/// lattice entry was negative and had no cheaper positive twin).
pub const FLAG_NEGATED: u32 = 1;

/// Entries scanned each side of the insertion point. Measured on the standard
/// table (`the_band_never_reaches_past_its_bound`): the widest band any literal
/// can have is 6 entries at 1e-3 and 9 at 5e-3, both sides together.
pub const BAND: usize = 16;
/// Second word of a device hit whose band reached past `BAND`: no entry.
pub const BAND_OVERFLOW: u32 = 2;
pub const FAMILIES: usize = 6;
pub const CONTEXTS: usize = 3;

/// What an entry IS, from the constant names its template uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Uses pi (with or without e, a root, phi, gamma).
    Pi = 0,
    /// Uses e and not pi.
    E = 1,
    /// sqrt2 or sqrt3, and neither pi nor e.
    Root = 2,
    /// No named constant at all.
    Rational = 3,
    /// Anything dimensional: h, hbar, c, G, kB, NA, qe, me, eps0, mu0, g_earth,
    /// and any name this file does not know.
    Physical = 4,
    /// phi, gamma.
    Other = 5,
}

impl Family {
    pub const ALL: [Family; FAMILIES] =
        [Family::Pi, Family::E, Family::Root, Family::Rational, Family::Physical, Family::Other];
}

/// Where a literal sits in its expression (`snap_graft::contexts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Context {
    /// Its nearest enclosing Sin / Cos / Tan or Exp / Log is a Sin, Cos or Tan.
    Cyclic = 0,
    /// Neither encloses it: polynomial and rational forms, a bare scale or offset.
    Algebraic = 1,
    /// Its nearest enclosing one is an Exp, ProtectedExp, Log or ProtectedLog.
    Exponential = 2,
}

impl Context {
    pub const ALL: [Context; CONTEXTS] = [Context::Cyclic, Context::Algebraic, Context::Exponential];
}

/// x_fit, the third objective: how badly an entry's family fits a literal's
/// context, in [0, 1] (0 = at home). Data, to be tuned: `fit[context][family]`,
/// families in `Family`'s order (pi, e, root, rational, physical, other).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapAffinity {
    pub fit: [[f32; FAMILIES]; CONTEXTS],
}

impl Default for SnapAffinity {
    /// A cyclic function showing 3.14 means pi; a polynomial maybe not; under
    /// an exponential the shortest form wins. A physical constant fits nowhere:
    /// SRBench supplies those as input columns, so a bare numeric match to one
    /// is almost certainly a coincidence.
    fn default() -> SnapAffinity {
        SnapAffinity {
            fit: [
                //  pi   e    root rational physical other
                [0.0, 0.5, 0.5, 0.3, 1.0, 0.7],
                [0.3, 0.5, 0.3, 0.0, 1.0, 0.7],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
        }
    }
}

/// A match: which entry, and whether the form to graft is the entry's
/// template negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub entry: u32,
    pub negative: bool,
}

/// What a search found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Found {
    Hit(Hit),
    Nothing,
    /// More entries within tolerance than the fixed scan covers: no entry is
    /// proposed, and the caller counts it.
    BandOverflow,
}

/// What the build left out, counted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dropped {
    /// Value is NaN or infinite.
    pub non_finite: usize,
    /// Labels whose |value| is not a normal f32 (overflows to inf, or underflows to a
    /// subnormal or to zero): the device cannot hold it. A genuine zero is
    /// kept.
    pub out_of_f32_range: Vec<String>,
    /// (label, node count) of templates larger than `TEMPLATE_MAX`.
    pub oversize: Vec<(String, usize)>,
    /// Labels of the losers with the same |value| in f64 as the entry kept (this is where a negative entry
    /// goes when its positive twin exists).
    pub exact_duplicates: Vec<String>,
    /// Labels of the losers different in f64, the same f32: the device could not tell them apart,
    /// and the f32 and f64 searches would split on them.
    pub f32_collisions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SnapTable {
    /// |value| per entry, strictly ascending (in f64 and in f32).
    pub values: Vec<f64>,
    pub values_f32: Vec<f32>,
    pub labels: Vec<String>,
    pub maths: Vec<String>,
    /// `INFO_STRIDE` words per entry.
    pub info: Vec<u32>,
    /// Every template, 4 words per node, `Var` nodes indexing `names`.
    pub templates: Vec<u32>,
    /// The constant names the templates' `Var` nodes refer to, sorted.
    pub names: Vec<String>,
    /// Template sizes: `sizes[k]` entries have a `k`-node template.
    pub sizes: Vec<usize>,
    pub entries_in: usize,
    pub dropped: Dropped,
    /// x_fit's table. `SnapTable::with_affinity` replaces it.
    pub affinity: SnapAffinity,
    /// Entries per family, in `Family`'s order.
    pub census: [usize; FAMILIES],
    /// Labels whose only dimensional names are h and hbar, in a ratio: h / hbar
    /// is 2 pi, so the entry is classed by what it IS (`hbar/h` is 1/(2 pi):
    /// family pi) and listed here.
    pub through_h_over_hbar: Vec<String>,
}

struct Candidate<'a> {
    magnitude: f64,
    entry: &'a ConstEntry,
    tree: Tree,
    nodes: usize,
}

fn var_names(tree: &Tree, out: &mut BTreeSet<String>) {
    match tree {
        Tree::Num(_) => {}
        Tree::Var(name) => {
            out.insert(name.clone());
        }
        Tree::App(_, kids) => kids.iter().for_each(|k| var_names(k, out)),
    }
}

/// The names that carry no dimension. Every other name is `Family::Physical`.
const PURE_NAMES: [&str; 6] = ["pi", "e", "sqrt2", "sqrt3", "phi", "gamma"];

/// The power each name is raised to, when the tree is a monomial (products,
/// quotients, roots and literal powers only). `None`: it has a sum in it.
fn powers(tree: &Tree, power: f64, out: &mut Vec<(String, f64)>) -> Option<()> {
    match tree {
        Tree::Num(_) => Some(()),
        Tree::Var(name) => {
            match out.iter_mut().find(|(n, _)| n == name) {
                Some((_, p)) => *p += power,
                None => out.push((name.clone(), power)),
            }
            Some(())
        }
        Tree::App(op, kids) => match (op, kids.as_slice()) {
            (Op::Mul, [a, b]) => powers(a, power, out).and(powers(b, power, out)),
            (Op::Div, [a, b]) => powers(a, power, out).and(powers(b, -power, out)),
            (Op::Neg, [a]) => powers(a, power, out),
            (Op::Inv, [a]) => powers(a, -power, out),
            (Op::Sqrt, [a]) => powers(a, power / 2.0, out),
            (Op::Pow2, [a]) => powers(a, power * 2.0, out),
            (Op::Pow3, [a]) => powers(a, power * 3.0, out),
            (Op::Pow, [a, Tree::Num(k)]) => powers(a, power * k, out),
            _ => None,
        },
    }
}

/// An entry's family, and whether it got there through h / hbar = 2 pi.
fn family_of(tree: &Tree) -> (Family, bool) {
    let mut names = BTreeSet::new();
    var_names(tree, &mut names);
    let mut used: Vec<String> = names.into_iter().collect();
    let mut through = false;
    if used.iter().any(|n| !PURE_NAMES.contains(&n.as_str())) {
        // h = 2 pi hbar: move h's power onto pi and hbar, and see what is left.
        let mut found = Vec::new();
        if powers(tree, 1.0, &mut found).is_none() {
            return (Family::Physical, false);
        }
        let power_of = |name: &str| found.iter().find(|(n, _)| n == name).map(|(_, p)| *p).unwrap_or(0.0);
        let h = power_of("h");
        let left: Vec<(String, f64)> = found
            .iter()
            .filter(|(n, _)| n != "h" && n != "hbar" && n != "pi")
            .cloned()
            .chain([("hbar".to_string(), power_of("hbar") + h), ("pi".to_string(), power_of("pi") + h)])
            .filter(|(_, p)| *p != 0.0)
            .collect();
        if left.iter().any(|(n, _)| !PURE_NAMES.contains(&n.as_str())) {
            return (Family::Physical, false);
        }
        used = left.into_iter().map(|(n, _)| n).collect();
        through = true;
    }
    let has = |name: &str| used.iter().any(|n| n == name);
    let family = if has("pi") {
        Family::Pi
    } else if has("e") {
        Family::E
    } else if has("sqrt2") || has("sqrt3") {
        Family::Root
    } else if used.is_empty() {
        Family::Rational
    } else {
        Family::Other
    };
    (family, through)
}

/// The choice inside a band, or the band reaching past the scan.
struct Picked {
    entry: Option<usize>,
    overflow: bool,
}

/// The f64 choice: every entry of the band, the smallest TrueNorth angle (by
/// hff's own function); an angle is monotone in the energy, which settles two
/// angles that acos cannot tell apart; then the lower entry.
fn pick_f64(t: &SnapTable, x: f64, tol: f64, context: Context) -> Picked {
    let table = &t.values;
    let rel_err = |v: f64| if v == 0.0 { x } else { (x - v).abs() / v };
    let at = table.partition_point(|v| *v < x);
    let (first, last) = (at.saturating_sub(BAND), (at + BAND).min(table.len()));
    if (first > 0 && rel_err(table[first - 1]) <= tol) || (last < table.len() && rel_err(table[last]) <= tol) {
        return Picked { entry: None, overflow: true };
    }
    let mut best: Option<(usize, f64, f64)> = None;
    for (i, value) in table.iter().enumerate().take(last).skip(first) {
        if rel_err(*value) > tol {
            continue;
        }
        let x3 = t.objectives(x, i as u32, context, tol);
        let energy: f64 = x3.iter().map(|v| v * v).sum();
        let angle = crate::score::truenorth_angle(&x3);
        if best.is_none_or(|(_, a, e)| angle < a || (angle == a && energy < e)) {
            best = Some((i, angle, energy));
        }
    }
    Picked { entry: best.map(|b| b.0), overflow: false }
}

const F32_MANTISSA: u32 = 0x007f_ffff;
/// The exponent the literal is moved to before any arithmetic: [4, 8).
const FRAME_EXPONENT: u32 = 129;

/// One candidate as the pair (distance, scale): its relative error is
/// `distance / scale`, and nobody divides. `x` and `t` are f32 bit patterns of
/// magnitudes. `None`: the two are more than a binade apart, so the relative
/// error is over 1/2 and the entry cannot qualify.
///
/// Both are first moved, by one exact power of two, to where the literal lies
/// in [4, 8). A device's f32 is not the host's at the ends of the range — this
/// kernel's first form, `abs(x - t) / t`, matched `f32::MAX` and
/// `f32::MIN_POSITIVE` to entries the host refused — and in this frame no
/// difference, product or tolerance is subnormal or overflows.
fn frame_f32(x: u32, t: u32) -> Option<(f32, f32)> {
    if t == 0 {
        // Against a zero constant the error is the literal's own magnitude.
        return Some((f32::from_bits(x), 1.0));
    }
    let (x_exp, t_exp) = (x >> 23, t >> 23);
    if x_exp == 0 || t_exp + 1 < x_exp || t_exp > x_exp + 1 {
        return None;
    }
    let xf = f32::from_bits((x & F32_MANTISSA) | (FRAME_EXPONENT << 23));
    let tf = f32::from_bits((t & F32_MANTISSA) | ((t_exp + FRAME_EXPONENT - x_exp) << 23));
    Some(((xf - tf).abs(), tf))
}

/// The candidate's (distance, scale) when it is within the tolerance.
fn in_band_f32(x: u32, t: u32, tol: f32) -> Option<(f32, f32)> {
    frame_f32(x, t).filter(|(d, scale)| *d <= tol * *scale)
}

/// The f32 choice, the code `snap.wgsl` transcribes: an integer search (the
/// bit patterns of non-negative floats order as the floats do), then the band
/// in ascending order.
///
/// The energy is x_near^2 + (x_size^2 + x_fit^2). The second term is an ADD of
/// two numbers the host squared and uploaded (`affinity_words`), and two
/// energies are compared as `(x - bx) * (x + bx) < bc - c`: no product feeds a
/// sum anywhere, so a device that fuses multiply-adds has nothing to fuse, and
/// every operation is one correctly rounded f32 operation on both sides.
/// Strict `<` in ascending order: a tie keeps the lower entry.
fn pick_f32(t: &SnapTable, words: &[f32], x: u32, tol: f32, inv_tol: f32, context: Context) -> Picked {
    let table = &t.values_f32;
    let at = table.partition_point(|v| v.to_bits() < x);
    let (first, last) = (at.saturating_sub(BAND), (at + BAND).min(table.len()));
    if (first > 0 && in_band_f32(x, table[first - 1].to_bits(), tol).is_some())
        || (last < table.len() && in_band_f32(x, table[last].to_bits(), tol).is_some())
    {
        return Picked { entry: None, overflow: true };
    }
    let mut best: Option<(usize, f32, f32)> = None;
    for (i, value) in table.iter().enumerate().take(last).skip(first) {
        let Some((d, scale)) = in_band_f32(x, value.to_bits(), tol) else {
            continue;
        };
        let near = (d / scale) * inv_tol;
        let family = t.info[i * INFO_STRIDE + 3] as usize;
        let nodes = t.info[i * INFO_STRIDE + 1] as usize;
        let cost = words[FIT_WORDS + nodes] + words[context as usize * FAMILIES + family];
        if best.is_none_or(|(_, bx, bc)| (near - bx) * (near + bx) < bc - cost) {
            best = Some((i, near, cost));
        }
    }
    Picked { entry: best.map(|b| b.0), overflow: false }
}

/// 1 / rel_tol, and 0 for a tolerance of 0 (x_near is then 0 * 0).
fn inverse(tol: f64) -> f64 {
    if tol == 0.0 { 0.0 } else { 1.0 / tol }
}

/// Words of x_fit^2 at the head of the device's affinity block; x_size^2 per
/// node count (0 ..= `TEMPLATE_MAX`) follows.
pub const FIT_WORDS: usize = CONTEXTS * FAMILIES;
pub const AFFINITY_WORDS: usize = FIT_WORDS + TEMPLATE_MAX + 1;

impl SnapTable {
    /// The table of the crate's own lattice.
    pub fn standard() -> Result<SnapTable, String> {
        SnapTable::build(&crate::snap_karva::lattice())
    }

    /// Order: ascending |value|, ties by label, then by math. Of entries that
    /// share an f32 value the one with the fewest template nodes is kept; on a
    /// tie a positive entry before a negative one, then by label, then math.
    pub fn build(entries: &[ConstEntry]) -> Result<SnapTable, String> {
        let mut dropped = Dropped::default();
        let mut cands: Vec<Candidate> = Vec::new();
        for e in entries {
            if !e.value.is_finite() {
                dropped.non_finite += 1;
                continue;
            }
            let magnitude = e.value.abs();
            if magnitude != 0.0 && !(magnitude as f32).is_normal() {
                dropped.out_of_f32_range.push(e.label.clone());
                continue;
            }
            let tree = Tree::parse(&e.math).map_err(|err| format!("lattice entry {:?}: {err}", e.label))?;
            let nodes = tree.node_count();
            if nodes > TEMPLATE_MAX {
                dropped.oversize.push((e.label.clone(), nodes));
                continue;
            }
            cands.push(Candidate { magnitude, entry: e, tree, nodes });
        }
        cands.sort_by(|a, b| {
            a.magnitude
                .total_cmp(&b.magnitude)
                .then_with(|| a.entry.label.cmp(&b.entry.label))
                .then_with(|| a.entry.math.cmp(&b.entry.math))
        });

        // The f32 cast is monotone, so entries that collide in f32 are
        // neighbours in the f64 order.
        let mut kept: Vec<Candidate> = Vec::new();
        for c in cands {
            let Some(last) = kept.last_mut() else {
                kept.push(c);
                continue;
            };
            if (last.magnitude as f32).to_bits() != (c.magnitude as f32).to_bits() {
                kept.push(c);
                continue;
            }
            // A negative entry and its positive twin meet here too.
            let exact = last.magnitude == c.magnitude;
            // Fewest nodes; then a positive entry before a negative one; then
            // the sort order (label, math), which `last` already leads.
            let loser = if (c.nodes, c.entry.value < 0.0) < (last.nodes, last.entry.value < 0.0) {
                std::mem::replace(last, c)
            } else {
                c
            };
            if exact {
                dropped.exact_duplicates.push(loser.entry.label.clone());
            } else {
                dropped.f32_collisions.push(loser.entry.label.clone());
            }
        }

        let mut name_set = BTreeSet::new();
        kept.iter().for_each(|c| var_names(&c.tree, &mut name_set));
        let names: Vec<String> = name_set.into_iter().collect();

        let mut table = SnapTable {
            values: Vec::with_capacity(kept.len()),
            values_f32: Vec::with_capacity(kept.len()),
            labels: Vec::with_capacity(kept.len()),
            maths: Vec::with_capacity(kept.len()),
            info: Vec::with_capacity(kept.len() * INFO_STRIDE),
            templates: Vec::new(),
            names,
            sizes: vec![0; TEMPLATE_MAX + 1],
            entries_in: entries.len(),
            dropped,
            affinity: SnapAffinity::default(),
            census: [0; FAMILIES],
            through_h_over_hbar: Vec::new(),
        };
        for c in &kept {
            let flat = Flat::from_tree_in(&c.tree, &table.names)?;
            let offset = (table.templates.len() / 4) as u32;
            for n in &flat.nodes {
                // A template's literals ride as f32 bits and nothing else, so
                // they have to be exact in f32.
                if n.op == Op::Num as u32 && f64::from(n.lit as f32) != n.lit {
                    return Err(format!("lattice entry {:?}: literal {} is not exact in f32", c.entry.label, n.lit));
                }
                table.templates.extend_from_slice(&[n.op, n.arg0, n.arg1, (n.lit as f32).to_bits()]);
            }
            let flags = if c.entry.value < 0.0 { FLAG_NEGATED } else { 0 };
            let (family, through) = family_of(&c.tree);
            table.census[family as usize] += 1;
            if through {
                table.through_h_over_hbar.push(c.entry.label.clone());
            }
            table.info.extend_from_slice(&[offset, flat.nodes.len() as u32, flags, family as u32]);
            table.sizes[flat.nodes.len()] += 1;
            table.values.push(c.magnitude);
            table.values_f32.push(c.magnitude as f32);
            table.labels.push(c.entry.label.clone());
            table.maths.push(c.entry.math.clone());
        }
        Ok(table)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Bytes resident on the device: values, info, templates.
    pub fn device_bytes(&self) -> (usize, usize, usize) {
        (self.values_f32.len() * 4, self.info.len() * 4, self.templates.len() * 4)
    }

    /// Steps the kernel's fixed binary search runs: `ceil(log2(n)) + 1`.
    pub fn search_steps(&self) -> u32 {
        self.len().next_power_of_two().trailing_zeros() + 1
    }

    /// An entry's template, read back out of the device block.
    pub fn template(&self, entry: u32) -> Flat {
        let at = entry as usize * INFO_STRIDE;
        let (offset, count) = (self.info[at] as usize, self.info[at + 1] as usize);
        let nodes = self.templates[offset * 4..(offset + count) * 4]
            .chunks_exact(4)
            .map(|w| LNode {
                op: w[0],
                arg0: w[1],
                arg1: w[2],
                lit: if w[0] == Op::Num as u32 { f64::from(f32::from_bits(w[3])) } else { 0.0 },
            })
            .collect();
        Flat { nodes, vars: self.names.clone() }
    }

    /// True when the entry's template evaluates to minus its table value.
    pub fn negated(&self, entry: u32) -> bool {
        self.info[entry as usize * INFO_STRIDE + 2] & FLAG_NEGATED != 0
    }

    /// The same table with another x_fit.
    pub fn with_affinity(mut self, affinity: SnapAffinity) -> SnapTable {
        self.affinity = affinity;
        self
    }

    pub fn family(&self, entry: u32) -> Family {
        Family::ALL[self.info[entry as usize * INFO_STRIDE + 3] as usize]
    }

    /// x_size's denominator: the largest template the table holds (7 in the
    /// standard table), not the capacity `TEMPLATE_MAX`.
    pub fn max_template_nodes(&self) -> usize {
        self.sizes.iter().rposition(|n| *n > 0).unwrap_or(0).max(1)
    }

    /// The device's affinity block: x_fit^2 per (context, family), then x_size^2
    /// per node count — squared here, in f32, so the kernel only adds them.
    pub fn affinity_words(&self) -> [f32; AFFINITY_WORDS] {
        let largest = self.max_template_nodes() as f32;
        let mut words = [0.0f32; AFFINITY_WORDS];
        for (word, fit) in words.iter_mut().zip(self.affinity.fit.iter().flatten()) {
            *word = fit * fit;
        }
        for (n, word) in words[FIT_WORDS..].iter_mut().enumerate() {
            let x = n as f32 / largest;
            *word = x * x;
        }
        words
    }

    /// (x_near, x_size, x_fit) of `entry` for the literal magnitude `x`, in f64.
    pub fn objectives(&self, x: f64, entry: u32, context: Context, rel_tol: f64) -> [f64; 3] {
        let t = self.values[entry as usize];
        let rel_err = if t == 0.0 { x.abs() } else { (x.abs() - t).abs() / t };
        let nodes = self.info[entry as usize * INFO_STRIDE + 1];
        [
            rel_err * inverse(rel_tol),
            f64::from(nodes) / self.max_template_nodes() as f64,
            f64::from(self.affinity.fit[context as usize][self.family(entry) as usize]),
        ]
    }

    /// The search, with the overfull band told apart from no entry. `F32` is
    /// the device's arithmetic: the literal, the table, the tolerance and the
    /// affinities are all rounded to f32 first.
    pub fn search(&self, v: f64, context: Context, rel_tol: f64, mode: LitMode) -> Found {
        assert!((0.0..=0.5).contains(&rel_tol), "rel_tol {rel_tol}: outside 0 ..= 0.5");
        let (picked, literal_negative) = match mode {
            LitMode::F64 => {
                if !v.is_finite() {
                    return Found::Nothing;
                }
                (pick_f64(self, v.abs(), rel_tol, context), v.is_sign_negative())
            }
            LitMode::F32 => {
                let x = (v as f32).abs();
                // The kernel refuses these on the bit pattern.
                if !(x.is_normal() || x == 0.0) {
                    return Found::Nothing;
                }
                let tol = rel_tol as f32;
                let inv_tol = inverse(rel_tol) as f32;
                (pick_f32(self, &self.affinity_words(), x.to_bits(), tol, inv_tol, context), (v as f32).is_sign_negative())
            }
        };
        if picked.overflow {
            return Found::BandOverflow;
        }
        let Some(entry) = picked.entry else {
            return Found::Nothing;
        };
        let entry = entry as u32;
        // Against a zero constant there is no sign: snap.rs tries +0 first.
        let negative = self.values[entry as usize] != 0.0 && (literal_negative != self.negated(entry));
        Found::Hit(Hit { entry, negative })
    }

    /// The entry `v` snaps to in `context`, or `None` (also for an overfull
    /// band: `search` tells the two apart).
    pub fn nearest_in(&self, v: f64, context: Context, rel_tol: f64, mode: LitMode) -> Option<Hit> {
        match self.search(v, context, rel_tol, mode) {
            Found::Hit(hit) => Some(hit),
            Found::Nothing | Found::BandOverflow => None,
        }
    }

    /// `nearest_in` for a literal on its own: a bare scale or offset is
    /// `Context::Algebraic`.
    pub fn nearest(&self, v: f64, rel_tol: f64, mode: LitMode) -> Option<Hit> {
        self.nearest_in(v, Context::Algebraic, rel_tol, mode)
    }

    /// The host's word on a device proposal: true iff the f64 search makes
    /// exactly this decision for the f64 literal in this context. Nothing the
    /// device matched may be reported or written back as a constant without it.
    pub fn confirm_f64_in(&self, value: f64, context: Context, hit: Hit, rel_tol: f64) -> bool {
        self.nearest_in(value, context, rel_tol, LitMode::F64) == Some(hit)
    }

    /// `confirm_f64_in` for a literal on its own.
    pub fn confirm_f64(&self, value: f64, hit: Hit, rel_tol: f64) -> bool {
        self.confirm_f64_in(value, Context::Algebraic, hit, rel_tol)
    }

    /// The signed constant a hit stands for.
    pub fn signed_value(&self, hit: Hit) -> f64 {
        let v = self.values[hit.entry as usize];
        if hit.negative != self.negated(hit.entry) { -v } else { v }
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const MAX_GROUPS_PER_DIM: u32 = crate::gpu_eval::MAX_GROUPS_PER_DIM;

    /// wgpu buffers do not cross devices and a `Device` is not `Clone`: the
    /// kernel either owns its device or borrows the one its blocks live on.
    enum Gpu<'d> {
        Own(wgpu::Device, wgpu::Queue),
        Shared(&'d wgpu::Device, &'d wgpu::Queue),
    }

    pub struct SnapKernel<'d> {
        gpu: Gpu<'d>,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        values_buf: wgpu::Buffer,
        info_buf: wgpu::Buffer,
        templates_buf: wgpu::Buffer,
        affinity_buf: wgpu::Buffer,
        table: SnapTable,
    }

    struct Resident {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        values_buf: wgpu::Buffer,
        info_buf: wgpu::Buffer,
        templates_buf: wgpu::Buffer,
        affinity_buf: wgpu::Buffer,
    }

    /// The pipeline and the resident blocks, on `device`.
    fn resident(device: &wgpu::Device, table: &SnapTable) -> Resident {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fuller-snap"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SNAP_WGSL)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fuller-snap-layout"),
            entries: &(0..7)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if i == 4 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: i != 3 }
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
            label: Some("fuller-snap-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "snap_main",
            compilation_options: Default::default(),
        });
        let block = |label: &str, words: &[u32]| {
            // wgpu rejects a zero-sized binding.
            let padded: Vec<u32> = if words.is_empty() { vec![0] } else { words.to_vec() };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&padded),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let value_bits: Vec<u32> = table.values_f32.iter().map(|v| v.to_bits()).collect();
        let affinity_bits: Vec<u32> = table.affinity_words().iter().map(|v| v.to_bits()).collect();
        Resident {
            values_buf: block("snap-values", &value_bits),
            info_buf: block("snap-info", &table.info),
            templates_buf: block("snap-templates", &table.templates),
            affinity_buf: block("snap-affinity", &affinity_bits),
            pipeline,
            layout,
        }
    }

    impl SnapKernel<'static> {
        pub fn new(table: SnapTable) -> Result<Self, String> {
            pollster::block_on(Self::new_async(table))
        }

        async fn new_async(table: SnapTable) -> Result<Self, String> {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok_or("no GPU adapter")?;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("fuller-snap-device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("request_device: {e}"))?;
            let r = resident(&device, &table);
            Ok(Self::assemble(Gpu::Own(device, queue), r, table))
        }
    }

    impl<'d> SnapKernel<'d> {
        /// The kernel on a device somebody else owns — the evaluator's, so that
        /// match, graft, evaluation and the guard share their buffers.
        pub fn on_device(device: &'d wgpu::Device, queue: &'d wgpu::Queue, table: SnapTable) -> Result<Self, String> {
            let r = resident(device, &table);
            Ok(Self::assemble(Gpu::Shared(device, queue), r, table))
        }

        fn assemble(gpu: Gpu<'d>, r: Resident, table: SnapTable) -> Self {
            Self {
                gpu,
                pipeline: r.pipeline,
                layout: r.layout,
                values_buf: r.values_buf,
                info_buf: r.info_buf,
                templates_buf: r.templates_buf,
                affinity_buf: r.affinity_buf,
                table,
            }
        }

        pub fn table(&self) -> &SnapTable {
            &self.table
        }

        /// The resident template block, for the graft kernel to bind (the
        /// match kernel does not read it).
        pub fn templates_buffer(&self) -> &wgpu::Buffer {
            &self.templates_buf
        }

        /// The resident per-entry (offset, count, flags, family) rows.
        pub fn info_buffer(&self) -> &wgpu::Buffer {
            &self.info_buf
        }

        /// The device the resident blocks live on: a kernel that binds them
        /// (the graft) has to be built on it.
        pub fn device(&self) -> &wgpu::Device {
            match &self.gpu {
                Gpu::Own(device, _) => device,
                Gpu::Shared(device, _) => device,
            }
        }

        pub fn queue(&self) -> &wgpu::Queue {
            match &self.gpu {
                Gpu::Own(_, queue) => queue,
                Gpu::Shared(_, queue) => queue,
            }
        }

        /// Record the match of `n_lit` literals — `buffers` is (f32 bits, a
        /// context code each, the hits: 2 words per literal) — on `enc`, so a later pass of the same command buffer reads
        /// the hits with no host round trip. `max_groups` is the most
        /// workgroups one dispatch row may hold. The uniform buffer returned is
        /// the caller's to destroy after the submit.
        pub fn match_pass(
            &self,
            enc: &mut wgpu::CommandEncoder,
            buffers: [&wgpu::Buffer; 3],
            n_lit: u32,
            rel_tol: f64,
            max_groups: u32,
        ) -> wgpu::Buffer {
            assert!((0.0..=0.5).contains(&rel_tol), "rel_tol {rel_tol}: outside 0 ..= 0.5");
            assert!(n_lit > 0 && (1..=MAX_GROUPS_PER_DIM).contains(&max_groups));
            let [lits_buf, contexts_buf, hits_buf] = buffers;
            let groups = n_lit.div_ceil(64);
            let groups_x = groups.min(max_groups);
            let groups_y = groups.div_ceil(groups_x);
            let cfg = [
                n_lit,
                self.table.len() as u32,
                groups_x * 64,
                self.table.search_steps(),
                (rel_tol as f32).to_bits(),
                (inverse(rel_tol) as f32).to_bits(),
                0,
                0,
            ];
            let cfg_buf = self.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("snap-cfg"),
                contents: bytemuck::cast_slice(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let buffers =
                [lits_buf, &self.values_buf, &self.info_buf, hits_buf, &cfg_buf, contexts_buf, &self.affinity_buf];
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let bind = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &entries,
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            cfg_buf
        }

        /// Match every literal in one dispatch, each on its own (`Context::
        /// Algebraic`).
        pub fn run(&self, literals: &[f32], rel_tol: f64) -> Result<Vec<Option<Hit>>, String> {
            let found = self.run_in(literals, &vec![Context::Algebraic; literals.len()], rel_tol)?;
            Ok(found.into_iter().map(|f| if let Found::Hit(hit) = f { Some(hit) } else { None }).collect())
        }

        /// Match every literal, in its context, in one dispatch.
        pub fn run_in(&self, literals: &[f32], contexts: &[Context], rel_tol: f64) -> Result<Vec<Found>, String> {
            assert!((0.0..=0.5).contains(&rel_tol), "rel_tol {rel_tol}: outside 0 ..= 0.5");
            assert_eq!(literals.len(), contexts.len(), "a context per literal");
            if literals.is_empty() {
                return Ok(Vec::new());
            }
            let device = self.device();
            let n_lit = u32::try_from(literals.len()).map_err(|_| "more than u32::MAX literals".to_string())?;
            let bits: Vec<u32> = literals.iter().map(|v| v.to_bits()).collect();
            let codes: Vec<u32> = contexts.iter().map(|c| *c as u32).collect();
            let upload = |label: &str, words: &[u32]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(words),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let lits_buf = upload("snap-literals", &bits);
            let contexts_buf = upload("snap-contexts", &codes);
            let out_bytes = (literals.len() * 8) as u64;
            let hits_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snap-hits"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snap-hits-read"),
                size: out_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let cfg_buf =
                self.match_pass(&mut enc, [&lits_buf, &contexts_buf, &hits_buf], n_lit, rel_tol, MAX_GROUPS_PER_DIM);
            enc.copy_buffer_to_buffer(&hits_buf, 0, &read_buf, 0, out_bytes);
            self.queue().submit(Some(enc.finish()));

            let slice = read_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|e| format!("map_async channel: {e}"))?
                .map_err(|e| format!("map_async: {e}"))?;
            let words = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
            read_buf.unmap();

            // Release per-dispatch buffers explicitly and drain wgpu's deferred
            // queue — see gpu_eval::GpuEvaluator::eval for what happens otherwise.
            for b in [&lits_buf, &contexts_buf, &hits_buf, &read_buf, &cfg_buf] {
                b.destroy();
            }
            device.poll(wgpu::Maintain::Poll);
            Ok(words.chunks_exact(2).map(found_of).collect())
        }
    }
}

/// A device hit's two words.
pub fn found_of(w: &[u32]) -> Found {
    if w[0] != NONE {
        Found::Hit(Hit { entry: w[0], negative: w[1] != 0 })
    } else if w[1] == BAND_OVERFLOW {
        Found::BandOverflow
    } else {
        Found::Nothing
    }
}

#[cfg(feature = "gpu")]
pub use gpu::SnapKernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::draw;
    use crate::lint::pack::arity;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-3;

    /// The constants as SRBench's laws PRINT them (3 significant figures) —
    /// text, because a number typed as 3.14 is a rounded constant on purpose.
    const SRBENCH: [&str; 23] = [
        "0.013", "0.016", "0.04", "0.048", "0.05", "0.053", "0.075", "0.08", "0.1", "0.101", "0.119", "0.125",
        "0.159", "0.239", "0.318", "0.399", "0.5", "2.89", "3.14", "4.19", "6.28", "12.6", "25.1",
    ];

    /// Snapshots of the standard table's families.
    const CENSUS: [usize; FAMILIES] = [634, 214, 276, 25, 5200, 219];
    const THROUGH_H_OVER_HBAR: usize = 3;
    /// The widest band at 1e-3 and at 5e-3.
    const WIDEST: (usize, usize) = (6, 9);

    fn printed(text: &str) -> f64 {
        text.parse().expect("a number")
    }

    fn table() -> SnapTable {
        SnapTable::standard().expect("the lattice builds")
    }

    /// 10,000 literals log-uniform over 1e-6..1e6, a fair coin for the sign,
    /// then SRBench's 23. Rounded to f32 — what a gene's constant is — so the
    /// f32 and f64 searches start from the same number.
    fn literals() -> Vec<f32> {
        let mut out: Vec<f32> = (0..10_000u32)
            .map(|row| {
                let u = (draw(7, 0, row, 0, 0) >> 11) as f64 / (1u64 << 53) as f64;
                let magnitude = 10f64.powf(-6.0 + 12.0 * u);
                let v = if crate::evolve::coin(draw(7, 0, row, 1, 0)) { -magnitude } else { magnitude };
                v as f32
            })
            .collect();
        out.extend(SRBENCH.iter().map(|v| printed(v) as f32));
        out
    }

    /// The rule with no search and no band: EVERY entry, as itself and negated,
    /// inside the tolerance; the smallest TrueNorth angle (then the energy it is
    /// a function of, then the lower entry).
    fn brute_force(t: &SnapTable, v: f64, context: Context, rel_tol: f64) -> Option<f64> {
        if !v.is_finite() {
            return None;
        }
        let mut best: Option<(f64, f64, f64)> = None;
        for (i, &val) in t.values.iter().enumerate() {
            for signed in [val, -val] {
                let rel_err = if signed == 0.0 { v.abs() } else { (v - signed).abs() / signed.abs() };
                if rel_err > rel_tol {
                    continue;
                }
                let x3 = t.objectives(v, i as u32, context, rel_tol);
                let (angle, energy) = (crate::score::truenorth_angle(&x3), x3.iter().map(|x| x * x).sum::<f64>());
                if best.is_none_or(|(_, a, e)| angle < a || (angle == a && energy < e)) {
                    best = Some((signed, angle, energy));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// The context a test literal is given: they take turns.
    fn context_of(row: usize) -> Context {
        Context::ALL[row % CONTEXTS]
    }

    /// One candidate, legibly: label, family, the three objectives, the angle.
    fn described(t: &SnapTable, v: f64, entry: u32, context: Context, rel_tol: f64) -> String {
        let x3 = t.objectives(v, entry, context, rel_tol);
        format!(
            "{} [{:?}; near {:.3}, size {:.3}, fit {:.1}; angle {:.4}]",
            t.labels[entry as usize],
            t.family(entry),
            x3[0],
            x3[1],
            x3[2],
            crate::score::truenorth_angle(&x3)
        )
    }

    /// Every entry within tolerance of `v`, ascending.
    fn band(t: &SnapTable, v: f64, rel_tol: f64) -> Vec<u32> {
        (0..t.len() as u32).filter(|i| ((v.abs() - t.values[*i as usize]) / t.values[*i as usize]).abs() <= rel_tol).collect()
    }

    fn entry(value: f64, math: &str, label: &str) -> ConstEntry {
        ConstEntry { value, math: math.to_string(), label: label.to_string() }
    }

    #[test]
    fn the_table_is_strictly_ascending_and_its_counts_are_these() {
        let t = table();
        assert!(t.values.windows(2).all(|w| w[0] < w[1]), "f64 values");
        assert!(t.values_f32.windows(2).all(|w| w[0] < w[1]), "f32 values");
        assert!(t.values_f32.iter().all(|v| v.is_normal()));
        let (values, info, templates) = t.device_bytes();
        eprintln!(
            "snap table: {} lattice entries -> {} on the device; out of f32 range {} (first {:?}); exact duplicates {:?}; \
             f32 collisions {:?}; template sizes {:?}; \
             {} names; bytes: values {values}, info {info}, templates {templates}; search steps {}",
            t.entries_in,
            t.len(),
            t.dropped.out_of_f32_range.len(),
            &t.dropped.out_of_f32_range[..6],
            t.dropped.exact_duplicates,
            t.dropped.f32_collisions,
            t.sizes,
            t.names.len(),
            t.search_steps()
        );
        // A snapshot: a change to the lattice has to show up here.
        assert_eq!(t.entries_in, 7225);
        assert_eq!(t.dropped.non_finite, 0);
        assert_eq!(t.dropped.oversize, Vec::<(String, usize)>::new());
        assert_eq!(t.dropped.out_of_f32_range.len(), 605);
        // 37 = the 17 negative constants and the 20 negative small rationals,
        // each folded onto its positive twin.
        assert_eq!(t.dropped.exact_duplicates.len(), 37);
        assert!(t.dropped.exact_duplicates.iter().all(|l| l.starts_with('-')), "{:?}", t.dropped.exact_duplicates);
        assert_eq!(t.dropped.f32_collisions.len(), 15);
        assert_eq!(t.len(), 6568);
        assert_eq!(t.sizes[..8], [0, 29, 17, 990, 695, 1233, 538, 3066]);
        assert!(t.info.chunks_exact(INFO_STRIDE).all(|w| w[2] == 0), "no negated entry survives in the standard table");
        assert_eq!(t.max_template_nodes(), 7);
        // Family census, in `Family`'s order: pi, e, root, rational, physical, other.
        eprintln!(
            "snap families {:?}: {:?}; classed through h / hbar = 2 pi: {} ({:?} ...)",
            Family::ALL,
            t.census,
            t.through_h_over_hbar.len(),
            &t.through_h_over_hbar[..t.through_h_over_hbar.len().min(8)]
        );
        assert_eq!(t.census.iter().sum::<usize>(), t.len());
        assert_eq!(t.census, CENSUS);
        assert_eq!(t.through_h_over_hbar.len(), THROUGH_H_OVER_HBAR);
        assert_eq!(t.info.len(), t.len() * INFO_STRIDE);
        assert_eq!(t.sizes.iter().sum::<usize>(), t.len());
        assert_eq!(
            t.entries_in,
            t.len() + t.dropped.out_of_f32_range.len() + t.dropped.exact_duplicates.len() + t.dropped.f32_collisions.len()
        );
    }

    #[test]
    fn every_template_fits_and_evaluates_to_its_entry() {
        let t = table();
        let mut row: Vec<(String, f64)> =
            crate::snap_karva::constant_values().iter().map(|(k, v)| (k.clone(), *v)).collect();
        row.sort_by(|a, b| a.0.cmp(&b.0));
        for id in 0..t.len() as u32 {
            let flat = t.template(id);
            assert!(flat.nodes.len() <= TEMPLATE_MAX, "{}", t.labels[id as usize]);
            let tree = flat.to_tree();
            assert_eq!(tree.to_math(), Tree::parse(&t.maths[id as usize]).unwrap().to_math());
            let got = tree.eval(&row).unwrap();
            let want = if t.negated(id) { -t.values[id as usize] } else { t.values[id as usize] };
            assert!(
                ((got - want) / want).abs() <= 1e-12,
                "{}: template gives {got}, entry says {want}",
                t.labels[id as usize]
            );
            // Children after their parent, inside the template.
            for (k, n) in flat.nodes.iter().enumerate() {
                for a in [n.arg0, n.arg1].iter().take(arity(n.op)) {
                    assert!((*a as usize) > k && (*a as usize) < flat.nodes.len());
                }
            }
        }
    }

    #[test]
    fn known_constants_match() {
        let t = table();
        let label_of = |v: &str| t.nearest(printed(v), TOL, LitMode::F64).map(|h| t.labels[h.entry as usize].as_str());
        let value_of = |v: f64| t.nearest(v, TOL, LitMode::F64).map(|h| t.signed_value(h));
        let pi = t.nearest(PI, TOL, LitMode::F64).expect("pi");
        assert_eq!(t.labels[pi.entry as usize], "pi");
        assert!(!pi.negative);
        // The physicist's constant, not the nearest coincidence.
        assert_eq!(label_of("3.14159"), Some("pi"));
        assert_eq!(label_of("6.2832"), Some("2*pi"));
        assert_eq!(label_of("0.31831"), Some("1/pi"));
        assert_eq!(label_of("1.35914"), Some("e/2"));
        assert_eq!(value_of(0.0796), Some(1.0 / (4.0 * PI)));
        // 1/(2 pi): the table keeps `hbar/h` (3 nodes; the lattice's plain `1/(2*pi)`
        // is 5 and has the same f32 value), classed by what it is — family pi.
        let half_turn = t.nearest(0.159155, TOL, LitMode::F64).expect("1/(2 pi)");
        assert!((t.signed_value(half_turn) * 2.0 * PI - 1.0).abs() <= 1e-9);
        assert_eq!(t.family(half_turn.entry), Family::Pi);
        let shortest = band(&t, 0.159155, TOL)
            .into_iter()
            .filter(|i| (t.values[*i as usize] * 2.0 * PI - 1.0).abs() <= 1e-7)
            .map(|i| t.info[i as usize * INFO_STRIDE + 1])
            .min();
        assert_eq!(Some(t.info[half_turn.entry as usize * INFO_STRIDE + 1]), shortest);
        // Negative literals carry the sign and match the same entry.
        let minus_pi = t.nearest(-printed("3.1416"), TOL, LitMode::F64).expect("-pi");
        assert_eq!(minus_pi, Hit { entry: pi.entry, negative: true });
        assert_eq!(t.signed_value(minus_pi), -PI);
    }

    #[test]
    fn families_are_what_the_entry_is() {
        let t = table();
        let family_of_label = |label: &str| t.family(t.labels.iter().position(|l| l == label).expect(label) as u32);
        assert_eq!(family_of_label("pi"), Family::Pi);
        assert_eq!(family_of_label("2*pi"), Family::Pi);
        assert_eq!(family_of_label("e/2"), Family::E);
        assert_eq!(family_of_label("sqrt2"), Family::Root);
        assert_eq!(family_of_label("1/2"), Family::Rational);
        assert_eq!(family_of_label("3"), Family::Rational);
        assert_eq!(family_of_label("phi"), Family::Other);
        assert_eq!(family_of_label("gamma"), Family::Other);
        assert_eq!(family_of_label("c"), Family::Physical);
        assert_eq!(family_of_label("h"), Family::Physical);
        // h / hbar = 2 pi: a ratio of the two is a pi form, whatever its label.
        assert_eq!(family_of_label("hbar/h"), Family::Pi);
        let tree = |m: &str| Tree::parse(m).unwrap();
        // `(3*hbar)/(2*h)` is in the lattice and no longer in the table: the
        // plain `3/(4*pi)` has the same f32 value in 5 nodes to its 7.
        assert_eq!(family_of(&tree(r#"(Div (Mul (Num 3.0) (Var "hbar")) (Mul (Num 2.0) (Var "h")))"#)), (Family::Pi, true));
        assert_eq!(family_of_label("3/(4*pi)"), Family::Pi);
        assert_eq!(family_of(&tree(r#"(Div (Var "h") (Mul (Num 4.0) (Mul (Var "pi") (Var "hbar"))))"#)), (Family::Rational, true));
        assert_eq!(family_of(&tree(r#"(Div (Mul (Var "e") (Var "hbar")) (Var "h"))"#)), (Family::Pi, true));
        assert_eq!(family_of(&tree(r#"(Mul (Var "h") (Var "hbar"))"#)), (Family::Physical, false));
        assert_eq!(family_of(&tree(r#"(Sqrt (Div (Var "h") (Var "hbar")))"#)), (Family::Pi, true));
        assert_eq!(family_of(&tree(r#"(Add (Var "h") (Var "hbar"))"#)), (Family::Physical, false));
        assert_eq!(family_of(&tree(r#"(Var "five")"#)), (Family::Physical, false));
        // Did the lattice keep a plain twin of a value it writes through h and
        // hbar? Counted against the raw lattice, by value to 1e-8.
        let lattice = crate::snap_karva::lattice();
        let plain: Vec<f64> = lattice
            .iter()
            .filter(|e| family_of(&tree(&e.math)) != (Family::Physical, false) && !family_of(&tree(&e.math)).1)
            .map(|e| e.value.abs())
            .collect();
        let with_twin = t
            .through_h_over_hbar
            .iter()
            .filter(|label| {
                let v = t.values[t.labels.iter().position(|l| l == *label).unwrap()];
                plain.iter().any(|p| ((p - v) / v).abs() <= 1e-8)
            })
            .count();
        eprintln!(
            "{} entries are written through h / hbar; {with_twin} of them have a plain twin of the same value in the raw lattice",
            t.through_h_over_hbar.len()
        );
    }

    /// The widest band the table can produce: the most entries t with
    /// x / (1 + tol) <= t <= x / (1 - tol) for any x — a window of ratio
    /// (1 + tol) / (1 - tol) over the sorted values.
    fn widest_band(values: &[f64], tol: f64) -> usize {
        let ratio = (1.0 + tol) / (1.0 - tol);
        let mut widest = 0;
        let mut hi = 0;
        for lo in 0..values.len() {
            while hi < values.len() && values[hi] <= values[lo] * ratio {
                hi += 1;
            }
            widest = widest.max(hi - lo);
        }
        widest
    }

    #[test]
    fn the_band_never_reaches_past_its_bound() {
        let t = table();
        let f32_values: Vec<f64> = t.values_f32.iter().map(|v| f64::from(*v)).collect();
        let mut lits: Vec<f64> = literals().iter().map(|v| f64::from(*v)).collect();
        lits.extend(t.values.iter().copied());
        lits.extend(edge_literals(&t).iter().map(|v| f64::from(*v)));
        for rel_tol in [TOL, 5e-3] {
            let (wide, wide_f32) = (widest_band(&t.values, rel_tol), widest_band(&f32_values, rel_tol * (1.0 + 1e-5)));
            let mut overflows = 0usize;
            for (row, v) in lits.iter().enumerate() {
                for mode in [LitMode::F64, LitMode::F32] {
                    overflows += usize::from(t.search(*v, context_of(row), rel_tol, mode) == Found::BandOverflow);
                }
            }
            eprintln!(
                "snap band at {rel_tol}: widest {wide} entries (f32 table, tolerance stretched 1e-5: {wide_f32}); BAND = {BAND} each side; {overflows} overflows on {} literals",
                lits.len()
            );
            // The whole band fits on ONE side of the insertion point, with margin.
            assert!(wide_f32.max(wide) * 5 <= BAND * 4, "BAND {BAND} does not cover a band of {wide} with margin");
            assert_eq!(overflows, 0);
            assert_eq!(wide, if rel_tol == TOL { WIDEST.0 } else { WIDEST.1 });
        }
        // Past the supported tolerances the overfull band is reported, not cut.
        let dense = (0..t.len()).max_by_key(|i| band(&t, t.values[*i], 0.1).len()).unwrap();
        assert!(band(&t, t.values[dense], 0.1).len() > 2 * BAND);
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.search(t.values[dense], Context::Algebraic, 0.1, mode), Found::BandOverflow);
            assert_eq!(t.nearest(t.values[dense], 0.1, mode), None);
        }
    }

    #[test]
    fn far_from_everything_and_not_a_number_are_none() {
        let t = table();
        // Midway between two neighbours that are at least 4e-3 apart.
        let gap = t.values.windows(2).find(|w| w[0] > 1.0 && w[1] / w[0] > 1.01).expect("a gap");
        let lonely = (gap[0] * gap[1]).sqrt();
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(lonely, 2e-3, mode), None);
            assert_eq!(t.nearest(0.0, TOL, mode), None);
            assert_eq!(t.nearest(-0.0, TOL, mode), None);
            assert_eq!(t.nearest(f64::NAN, TOL, mode), None);
            assert_eq!(t.nearest(f64::INFINITY, TOL, mode), None);
            assert_eq!(t.nearest(f64::NEG_INFINITY, TOL, mode), None);
        }
        // Finite in f64, not in f32.
        assert_eq!(t.nearest(1e300, TOL, LitMode::F32), None);
        assert_eq!(t.nearest(1e-42, TOL, LitMode::F32), None);
    }

    #[test]
    fn the_tolerance_band_has_a_hard_edge() {
        let t = table();
        // An entry with 1% of clear air either side: no other band reaches
        // its edges. (pi is not one: `3307` sits 1e-3 below it.)
        let alone = (1..t.len() - 1)
            .find(|i| t.values[*i] > 1.0 && t.values[*i] / t.values[i - 1] > 1.01 && t.values[i + 1] / t.values[*i] > 1.01)
            .expect("an isolated entry");
        let (value, hit) = (t.values[alone], Hit { entry: alone as u32, negative: false });
        for side in [-1.0, 1.0] {
            let inside = value * (1.0 + side * TOL * (1.0 - 1e-9));
            let outside = value * (1.0 + side * TOL * (1.0 + 1e-9));
            assert_eq!(t.nearest(inside, TOL, LitMode::F64), Some(hit), "inside, side {side}");
            assert_eq!(t.nearest(outside, TOL, LitMode::F64), None, "outside, side {side}");
            assert!(t.confirm_f64(inside, hit, TOL));
            assert!(!t.confirm_f64(outside, hit, TOL));
            assert_eq!(t.nearest(-inside, TOL, LitMode::F64), Some(Hit { negative: true, ..hit }));
        }
        let pi = t.nearest(PI, TOL, LitMode::F64).unwrap();
        // rel_tol 0 is exact equality.
        assert_eq!(t.nearest(PI, 0.0, LitMode::F64), Some(pi));
        assert_eq!(t.nearest(PI * (1.0 + 1e-15), 0.0, LitMode::F64), None);
    }

    #[test]
    fn ties_keep_the_lower_entry_and_relative_error_decides() {
        let t = SnapTable::build(&[entry(1.0, "(Num 1.0)", "1"), entry(3.0, "(Num 3.0)", "3")]).unwrap();
        // 1.5: relative error exactly 0.5 against both. The lower entry keeps it.
        let lower = Some(Hit { entry: 0, negative: false });
        let upper = Some(Hit { entry: 1, negative: false });
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(1.5, 0.5, mode), lower);
            // 1.25 is nearer 1 on the number line and nearer 1 by ratio too;
            // 2.0 is the same distance from both, but 1/3 of 3 and 1/1 of 1.
            assert_eq!(t.nearest(1.25, 0.5, mode), lower);
            assert_eq!(t.nearest(2.0, 0.5, mode), upper);
        }
        assert_eq!(brute_force(&t, 1.5, Context::Algebraic, 0.5), Some(1.0));
        // Same distance, both inside the tolerance: the smaller RATIO wins.
        let t = SnapTable::build(&[entry(2.0, "(Num 2.0)", "2"), entry(4.0, "(Num 4.0)", "4")]).unwrap();
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(3.0, 0.5, mode), upper);
        }
        assert_eq!(brute_force(&t, 3.0, Context::Algebraic, 0.5), Some(4.0));
    }

    #[test]
    fn a_zero_entry_uses_the_absolute_rule_and_has_no_sign() {
        let t = SnapTable::build(&[entry(0.0, "(Num 0.0)", "0"), entry(1.0, "(Num 1.0)", "1")]).unwrap();
        let zero = Some(Hit { entry: 0, negative: false });
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(0.0, TOL, mode), zero);
            assert_eq!(t.nearest(-0.0, TOL, mode), zero);
            assert_eq!(t.nearest(9.9e-4, TOL, mode), zero);
            assert_eq!(t.nearest(-9.9e-4, TOL, mode), zero, "snap.rs tries +0 first: no sign");
            assert_eq!(t.nearest(1.1e-3, TOL, mode), None);
            assert_eq!(t.nearest(1.0005, TOL, mode), Some(Hit { entry: 1, negative: false }));
        }
        assert_eq!(brute_force(&t, -9.9e-4, Context::Algebraic, TOL), Some(0.0));
    }

    #[test]
    fn duplicates_negatives_and_the_f32_range_are_handled() {
        let t = SnapTable::build(&[
            entry(-2.0, r#"(Neg (Var "two"))"#, "-two"),
            entry(2.0, r#"(Mul (Var "one") (Var "two"))"#, "b"),
            entry(2.0, r#"(Var "two")"#, "two"),
            entry(2.0 + 4e-16, r#"(Add (Var "two") (Var "eps"))"#, "two+eps"),
            entry(-5.0, r#"(Neg (Var "five"))"#, "-five"),
            entry(1e-60, r#"(Var "tiny")"#, "tiny"),
            entry(1e60, r#"(Var "huge")"#, "huge"),
            entry(f64::NAN, r#"(Var "nan")"#, "nan"),
        ])
        .unwrap();
        assert_eq!(t.labels, ["two", "-five"]);
        assert_eq!(t.names, ["five", "two"]);
        let labels = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let want = Dropped {
            non_finite: 1,
            out_of_f32_range: labels(&["tiny", "huge"]),
            oversize: vec![],
            exact_duplicates: labels(&["b", "-two"]),
            f32_collisions: labels(&["two+eps"]),
        };
        assert_eq!(t.dropped, want);
        // A negative entry with no positive twin: its template is already the
        // negative form, so a NEGATIVE literal takes it as it is.
        assert_eq!(t.nearest(-5.0, TOL, LitMode::F64), Some(Hit { entry: 1, negative: false }));
        assert_eq!(t.nearest(5.0, TOL, LitMode::F64), Some(Hit { entry: 1, negative: true }));
        // Sixteen nodes is one too many.
        let mut big = r#"(Var "x")"#.to_string();
        for _ in 0..15 {
            big = format!("(Neg {big})");
        }
        let t = SnapTable::build(&[entry(1.0, &big, "big")]).unwrap();
        assert_eq!(t.dropped.oversize, [("big".to_string(), 16)]);
        assert!(t.is_empty());
        assert_eq!(t.nearest(1.0, TOL, LitMode::F64), None);
        // A template literal the device's f32 cannot hold exactly is refused.
        assert!(SnapTable::build(&[entry(0.1, "(Num 0.1)", "tenth")]).is_err());
    }

    #[test]
    fn the_banded_search_agrees_with_a_scan_of_the_whole_table() {
        let t = table();
        let lits = literals();
        let mut hits = 0usize;
        for rel_tol in [TOL, 5e-3] {
            for (row, v) in lits.iter().enumerate() {
                let v = f64::from(*v);
                let got = t.nearest_in(v, context_of(row), rel_tol, LitMode::F64).map(|h| t.signed_value(h));
                assert_eq!(got, brute_force(&t, v, context_of(row), rel_tol), "literal {v} at {rel_tol}");
                hits += usize::from(got.is_some());
            }
        }
        assert!(hits > 0);
        // SRBench's printed constants, as findings: the form chosen in each
        // context, with its three objectives and its angle.
        for rel_tol in [TOL, 5e-3] {
            let mut matched = 0;
            for v in SRBENCH {
                let chosen: Vec<String> = Context::ALL
                    .iter()
                    .map(|c| match t.nearest_in(printed(v), *c, rel_tol, LitMode::F64) {
                        Some(h) => format!("{c:?} -> {}", described(&t, printed(v), h.entry, *c, rel_tol)),
                        None => format!("{c:?} -> none"),
                    })
                    .collect();
                matched += usize::from(t.nearest(printed(v), rel_tol, LitMode::F64).is_some());
                eprintln!("SRBench {v} at {rel_tol} ({} in the band): {}", band(&t, printed(v), rel_tol).len(), chosen.join("; "));
            }
            // A snapshot of the finding, not a target: whether ANY form is
            // within tolerance of a constant printed to three figures.
            assert_eq!(matched, if rel_tol == TOL { 13 } else { 19 });
        }
    }

    /// The kernel takes the argmin of the ENERGY; HFF is the ANGLE. On random
    /// candidate sets the two orders are one order.
    #[test]
    fn the_energy_order_is_the_order_of_hffs_angles() {
        let unit = |row: u32, k: u32| (draw(31, 0, row, k, 0) >> 11) as f64 / (1u64 << 53) as f64;
        let mut strict = 0usize;
        for set in 0..4000u32 {
            let cands: Vec<[f64; 3]> = (0..8).map(|k| [unit(set, k * 3), unit(set, k * 3 + 1), unit(set, k * 3 + 2)]).collect();
            let energy = |x: &[f64; 3]| x.iter().map(|v| v * v).sum::<f64>();
            for a in &cands {
                for b in &cands {
                    let (ta, tb) = (crate::score::truenorth_angle(a), crate::score::truenorth_angle(b));
                    if energy(a) < energy(b) {
                        assert!(ta <= tb, "{a:?} has less energy than {b:?} and the larger angle");
                        strict += usize::from(ta < tb);
                    }
                }
            }
            let by_energy = cands.iter().map(energy).fold(f64::INFINITY, f64::min);
            let by_angle = cands.iter().map(|x| crate::score::truenorth_angle(x)).fold(f64::INFINITY, f64::min);
            let winner = cands.iter().find(|x| energy(x) == by_energy).unwrap();
            assert_eq!(crate::score::truenorth_angle(winner), by_angle, "set {set}");
            // And it is the construction the engine writes out.
            assert!((by_angle - (1.0 - (by_energy / 3.0).min(1.0)).acos()).abs() <= 1e-12);
        }
        assert!(strict > 100_000, "{strict}");
    }

    /// Real pairs from the table: a rational and a pi form that one literal can
    /// have within tolerance together.
    fn colliding_pairs(t: &SnapTable, rel_tol: f64) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for i in (0..t.len() as u32).filter(|i| t.family(*i) == Family::Rational) {
            for j in (0..t.len() as u32).filter(|j| t.family(*j) == Family::Pi) {
                let (a, b) = (t.values[i as usize], t.values[j as usize]);
                if a.min(b) * (1.0 + rel_tol) >= a.max(b) * (1.0 - rel_tol) {
                    out.push((i, j));
                }
            }
        }
        out
    }

    #[test]
    fn the_context_decides_between_a_rational_and_a_pi_form() {
        let t = table();
        let mut flips = [0usize; 2];
        for (k, rel_tol) in [TOL, 5e-3].into_iter().enumerate() {
            let pairs = colliding_pairs(&t, rel_tol);
            eprintln!(
                "rational / pi pairs one literal can hold within {rel_tol}: {} — {:?}",
                pairs.len(),
                pairs.iter().map(|(r, p)| format!("{} ~ {}", t.labels[*r as usize], t.labels[*p as usize])).collect::<Vec<_>>()
            );
            assert!(!pairs.is_empty());
            for (rational, pi_form) in &pairs {
                // Literals from one value to the other. Every real pi form here
                // is LONGER than its rational, so midway the rational wins in
                // every context; nearer the pi form, the context decides.
                let (a, b) = (t.values[*rational as usize], t.values[*pi_form as usize]);
                let decided: Vec<f64> = (0..=100)
                    .map(|step| a + (b - a) * f64::from(step) / 100.0)
                    .filter(|v| {
                        let winner = |c: Context| t.nearest_in(*v, c, rel_tol, LitMode::F64).map(|h| h.entry);
                        winner(Context::Cyclic) == Some(*pi_form) && winner(Context::Algebraic) == Some(*rational)
                    })
                    .collect();
                flips[k] += usize::from(!decided.is_empty());
                let Some(v) = decided.get(decided.len() / 2) else {
                    continue;
                };
                eprintln!("  {} ~ {}: the context decides for {} of 101 literals between them, e.g. {v}", t.labels[*rational as usize], t.labels[*pi_form as usize], decided.len());
                for c in Context::ALL {
                    let lines: Vec<String> = band(&t, *v, rel_tol).iter().map(|e| described(&t, *v, *e, c, rel_tol)).collect();
                    let winner = t.nearest_in(*v, c, rel_tol, LitMode::F64).expect("a band is not empty");
                    eprintln!("    in {c:?}: {} -> {}", lines.join(" | "), t.labels[winner.entry as usize]);
                }
            }
        }
        eprintln!("  pairs with a literal where cyclic -> the pi form and algebraic -> the rational: {flips:?} at [1e-3, 5e-3]");
        assert!(flips[0] > 0 && flips[1] > 0, "{flips:?}");
    }

    #[test]
    fn the_affinity_table_is_data_and_hff_trades_nearness_for_shortness() {
        let entries = [
            entry(3.0, "(Num 3.0)", "3"),
            entry(3.0 * (1.0 + 4e-4), r#"(Var "pi")"#, "pi-ish"),
            // Nearer than either, and seven nodes.
            entry(
                3.0 * (1.0 + 2e-4),
                r#"(Div (Mul (Num 3.0) (Var "e")) (Mul (Num 2.0) (Var "e")))"#,
                "long",
            ),
        ];
        let t = SnapTable::build(&entries).unwrap();
        assert_eq!(t.labels, ["3", "long", "pi-ish"]);
        assert_eq!([t.family(0), t.family(1), t.family(2)], [Family::Rational, Family::E, Family::Pi]);
        let v = 3.0 * (1.0 + 2.1e-4);
        let label = |t: &SnapTable, c: Context, mode: LitMode| t.labels[t.nearest_in(v, c, TOL, mode).unwrap().entry as usize].clone();
        for mode in [LitMode::F64, LitMode::F32] {
            // Under an exponential every family is at home: the nearest form is
            // `long` (near 0.01), and it loses to a one-node form at 0.19 or
            // 0.21 — a lexicographic nearness-first rule would have kept it.
            assert_eq!(label(&t, Context::Exponential, mode), "pi-ish");
            assert_eq!(label(&t, Context::Cyclic, mode), "pi-ish");
            assert_eq!(label(&t, Context::Algebraic, mode), "3");
        }
        for c in Context::ALL {
            eprintln!("  {v} in {c:?}: {}", (0..3).map(|e| described(&t, v, e, c, TOL)).collect::<Vec<_>>().join(" | "));
        }
        // Another table, another outcome: the choice is the data's.
        let mut affinity = SnapAffinity::default();
        affinity.fit[Context::Algebraic as usize][Family::Rational as usize] = 0.9;
        affinity.fit[Context::Algebraic as usize][Family::Pi as usize] = 0.0;
        let tuned = SnapTable::build(&entries).unwrap().with_affinity(affinity);
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(label(&tuned, Context::Algebraic, mode), "pi-ish");
        }
        // With every family at home everywhere, size and nearness are left.
        let flat = SnapTable::build(&entries).unwrap().with_affinity(SnapAffinity { fit: [[0.0; FAMILIES]; CONTEXTS] });
        let near_three = 3.0 * (1.0 + 1e-5);
        assert_eq!(flat.labels[flat.nearest_in(near_three, Context::Cyclic, TOL, LitMode::F64).unwrap().entry as usize], "3");
    }

    /// Literals a hair either side of every entry's band edge, in f32.
    fn edge_literals(t: &SnapTable) -> Vec<f32> {
        let mut out = Vec::new();
        for v in &t.values {
            for side in [-1.0, 1.0] {
                let edge = (v * (1.0 + side * TOL)) as f32;
                for step in [-2i32, -1, 0, 1, 2] {
                    out.push(f32::from_bits((edge.to_bits() as i32 + step) as u32));
                }
            }
        }
        out
    }

    /// Where F32 and F64 part ways, the literal sits on an edge: a candidate's
    /// relative error within 1e-6 of the tolerance, or two candidates' energies
    /// within 1e-5 of each other.
    fn on_an_edge(t: &SnapTable, v: f64, context: Context) -> bool {
        let x = v.abs();
        let near: Vec<u32> = (0..t.len() as u32)
            .filter(|i| ((x - t.values[*i as usize]) / t.values[*i as usize]).abs() <= TOL + 1e-6)
            .collect();
        let errs = near.iter().map(|i| (x - t.values[*i as usize]).abs() / t.values[*i as usize]);
        let energies: Vec<f64> =
            near.iter().map(|i| t.objectives(x, *i, context, TOL).iter().map(|o| o * o).sum()).collect();
        errs.into_iter().any(|e| (e - TOL).abs() <= 1e-6)
            || energies.iter().enumerate().any(|(k, a)| energies[..k].iter().any(|b| (a - b).abs() <= 1e-5))
    }

    #[test]
    fn f32_and_f64_part_ways_only_on_an_edge() {
        let t = table();
        for (name, lits) in [("random + SRBench", literals()), ("band edges", edge_literals(&t))] {
            let mut differ = 0usize;
            let mut unconfirmed = 0usize;
            for (row, v) in lits.iter().enumerate() {
                let (v, context) = (f64::from(*v), context_of(row));
                let narrow = t.nearest_in(v, context, TOL, LitMode::F32);
                if narrow != t.nearest_in(v, context, TOL, LitMode::F64) {
                    differ += 1;
                    assert!(on_an_edge(&t, v, context), "{v} in {context:?}: F32 and F64 differ away from any edge");
                }
                // The hook the later stages call: it turns down exactly the
                // proposals f64 would not have made.
                if let Some(hit) = narrow {
                    let ok = t.confirm_f64_in(v, context, hit, TOL);
                    assert_eq!(ok, t.nearest_in(v, context, TOL, LitMode::F64) == Some(hit));
                    unconfirmed += usize::from(!ok);
                }
            }
            eprintln!("F32 vs F64 on {} {name} literals: {differ} differ, {unconfirmed} F32 proposals refused by confirm_f64_in", lits.len());
            // The edge set is built to disagree; the random one must hardly ever.
            if name != "band edges" {
                assert!(differ * 1000 <= lits.len(), "{name}: {differ} of {}", lits.len());
            }
        }
    }

    #[test]
    fn two_builds_and_two_searches_are_identical() {
        let (a, b) = (table(), table());
        assert_eq!(a.values_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.values_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!((&a.info, &a.templates, &a.names, &a.labels), (&b.info, &b.templates, &b.names, &b.labels));
        // Input order does not matter either.
        let mut reversed = crate::snap_karva::lattice();
        reversed.reverse();
        let c = SnapTable::build(&reversed).unwrap();
        assert_eq!((&a.info, &a.templates, &a.labels), (&c.info, &c.templates, &c.labels));
        let run = |t: &SnapTable| {
            literals().iter().enumerate().map(|(row, v)| t.nearest_in(f64::from(*v), context_of(row), TOL, LitMode::F32)).collect::<Vec<_>>()
        };
        assert_eq!(run(&a), run(&c));
    }

    /// The kernel's layout constants are the host's.
    #[test]
    fn wgsl_constants_match_the_host() {
        for needle in [
            format!("const INFO_STRIDE: u32 = {INFO_STRIDE}u;"),
            format!("const FLAG_NEGATED: u32 = {FLAG_NEGATED}u;"),
            "const NONE: u32 = 0xffffffffu;".to_string(),
            format!("const MANTISSA: u32 = {F32_MANTISSA:#010x}u;"),
            format!("const FRAME_EXPONENT: u32 = {FRAME_EXPONENT}u;"),
            format!("const BAND: u32 = {BAND}u;"),
            format!("const BAND_OVERFLOW: u32 = {BAND_OVERFLOW}u;"),
            format!("const FAMILIES: u32 = {FAMILIES}u;"),
            format!("const FIT_WORDS: u32 = {FIT_WORDS}u;"),
        ] {
            assert!(SNAP_WGSL.contains(&needle), "snap.wgsl lacks `{needle}`");
        }
    }

    /// The F32 twin's word on each literal, contexts taking turns.
    #[cfg(feature = "gpu")]
    fn twin(t: &SnapTable, lits: &[f32], rel_tol: f64) -> Vec<Found> {
        lits.iter().enumerate().map(|(row, v)| t.search(f64::from(*v), context_of(row), rel_tol, LitMode::F32)).collect()
    }

    #[cfg(feature = "gpu")]
    fn turns(n: usize) -> Vec<Context> {
        (0..n).map(context_of).collect()
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn device_agrees_with_the_f32_twin() {
        let kernel = SnapKernel::new(table()).expect("a GPU adapter");
        let t = kernel.table().clone();
        assert_eq!(kernel.run(&[], TOL).unwrap(), Vec::new());
        assert_eq!(kernel.run_in(&[], &[], TOL).unwrap(), Vec::new());
        let one = [printed("3.1416") as f32];
        assert_eq!(kernel.run_in(&one, &turns(1), TOL).unwrap(), twin(&t, &one, TOL));
        assert_eq!(kernel.run(&one, TOL).unwrap(), [t.nearest(f64::from(one[0]), TOL, LitMode::F32)]);
        assert_eq!(kernel.run(&one, TOL).unwrap()[0].map(|h| t.labels[h.entry as usize].as_str()), Some("pi"));

        let specials = [0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE / 2.0, f32::MAX, f32::MIN_POSITIVE];
        assert_eq!(kernel.run_in(&specials, &turns(specials.len()), TOL).unwrap(), twin(&t, &specials, TOL));

        for rel_tol in [TOL, 5e-3] {
            for (name, lits) in [("random + SRBench", literals()), ("band edges", edge_literals(&t))] {
                let start = std::time::Instant::now();
                let got = kernel.run_in(&lits, &turns(lits.len()), rel_tol).unwrap();
                let took = start.elapsed();
                let again = kernel.run_in(&lits, &turns(lits.len()), rel_tol).unwrap();
                let want = twin(&t, &lits, rel_tol);
                let differ = got.iter().zip(&want).filter(|(g, w)| g != w).count();
                let hits = got.iter().filter(|f| matches!(f, Found::Hit(_))).count();
                let overflows = got.iter().filter(|f| **f == Found::BandOverflow).count();
                eprintln!(
                    "device snap at {rel_tol}, {} {name} literals in turning contexts: {hits} hits, {overflows} band overflows, {differ} differ from the F32 twin, {took:?}",
                    lits.len()
                );
                assert_eq!(got, want, "{name}");
                assert_eq!(got, again, "{name}: two dispatches");
                assert_eq!(overflows, 0);
            }
        }
        // Past the supported tolerances the device reports the overfull band too.
        let dense = (0..t.len()).max_by_key(|i| band(&t, t.values[*i], 0.1).len()).unwrap();
        let lit = [t.values_f32[dense]];
        assert_eq!(kernel.run_in(&lit, &turns(1), 0.1).unwrap(), [Found::BandOverflow]);
        assert_eq!(twin(&t, &lit, 0.1), [Found::BandOverflow]);
        assert_eq!(kernel.run(&lit, 0.1).unwrap(), [None]);
    }

    /// The affinity table reaches the device as data: another table, another
    /// choice, on the device as on the twin.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_reads_the_affinity_table() {
        let mut affinity = SnapAffinity::default();
        affinity.fit[Context::Algebraic as usize] = [0.9, 0.9, 0.9, 0.9, 0.0, 0.9];
        let standard = SnapKernel::new(table()).expect("a GPU adapter");
        let tuned = SnapKernel::new(table().with_affinity(affinity)).expect("a GPU adapter");
        let lits = literals();
        let algebraic = vec![Context::Algebraic; lits.len()];
        let (a, b) = (standard.run_in(&lits, &algebraic, TOL).unwrap(), tuned.run_in(&lits, &algebraic, TOL).unwrap());
        let want: Vec<Found> = lits.iter().map(|v| tuned.table().search(f64::from(*v), Context::Algebraic, TOL, LitMode::F32)).collect();
        assert_eq!(b, want);
        let moved = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        eprintln!("device snap, physical forms made welcome in algebraic context: {moved} of {} choices move", lits.len());
        assert!(moved > 0);
        // A kernel on a device somebody else owns gives the same words.
        let shared = SnapKernel::on_device(standard.device(), standard.queue(), table()).unwrap();
        assert_eq!(shared.run_in(&lits, &algebraic, TOL).unwrap(), a);
    }

    /// A zero entry and a negated entry on the device.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_handles_zero_and_negated_entries() {
        let t = SnapTable::build(&[
            entry(0.0, "(Num 0.0)", "0"),
            entry(1.0, "(Num 1.0)", "1"),
            entry(-5.0, r#"(Neg (Var "five"))"#, "-five"),
        ])
        .unwrap();
        let kernel = SnapKernel::new(t.clone()).expect("a GPU adapter");
        let lits = [0.0f32, -0.0, 9.9e-4, -9.9e-4, 1.1e-3, 1.0005, -1.0005, 5.0, -5.0, 3.0];
        let got = kernel.run_in(&lits, &turns(lits.len()), TOL).unwrap();
        assert_eq!(got, twin(&t, &lits, TOL));
        assert_eq!(got[3], Found::Hit(Hit { entry: 0, negative: false }));
        assert_eq!(got[7], Found::Hit(Hit { entry: 2, negative: true }));
        assert_eq!(got[8], Found::Hit(Hit { entry: 2, negative: false }));
        // An empty table matches nothing.
        let empty = SnapKernel::new(SnapTable::build(&[]).unwrap()).expect("a GPU adapter");
        assert_eq!(empty.run(&lits, TOL).unwrap(), vec![None; lits.len()]);
    }

    /// 200,000 literals in one row of groups; 4,200,000 wrap into a second.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_large_batches_match() {
        let kernel = SnapKernel::new(table()).expect("a GPU adapter");
        let t = kernel.table().clone();
        let base = literals();
        for n in [200_000usize, 4_200_000] {
            let lits: Vec<f32> = (0..n).map(|i| base[i % base.len()] * (1.0 + (i / base.len()) as f32 * 1e-4)).collect();
            assert_eq!(n.div_ceil(64) > crate::gpu_eval::MAX_GROUPS_PER_DIM as usize, n > 4_194_240);
            let start = std::time::Instant::now();
            let got = kernel.run_in(&lits, &turns(n), TOL).unwrap();
            let device = start.elapsed();
            let start = std::time::Instant::now();
            let want = twin(&t, &lits, TOL);
            let cpu = start.elapsed();
            let hits = got.iter().filter(|f| matches!(f, Found::Hit(_))).count();
            eprintln!("device snap, {n} literals: {hits} hits, device {device:?}, CPU F32 twin {cpu:?}");
            assert_eq!(got.len(), n);
            assert!(got == want, "{n} literals: device and F32 twin differ");
        }
    }
}
