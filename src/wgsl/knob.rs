//! THE KNOB MUTATION (`docs/PLAN_constrained_mutation.md` task 0, sections
//! 3, 3.1 and 5.1): a coordinated change of one tuning knob of a safe
//! kernel. The knob is a declaration (`params.rs`): the grid's x, y, bx, by,
//! the tile depth, or a workgroup array's length; every position that
//! depends on it is a reference the renderer writes from it, so one change
//! rewrites all of them in step.
//!
//! Firing: pick a knob of the individual that has a legal new value (a
//! LOOKUP in its value set), pick the value, and keep it only when the
//! JOINS hold for the child: the workgroup within the adapter's
//! invocations and per-axis limits, the block within the register budget,
//! the workgroup memory the child asks for within the adapter's (never
//! halved by the renderer), and every reference of the kernel resolving
//! EXACTLY into its slot's range (a load loop still covers the tile: T a
//! whole multiple of what the loop divides it by). The child then passes
//! the layered checker as far as it exists ([`check`]): rendered, validated
//! for the adapter (`target.rs`), and read back by the reader's ownership
//! proof with the knobs it was given. A child that fails [`check`] is a bug
//! in the rule, never dispatched.

use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::params::{exact_value, ktile_of, parse_ref, Knobs};
use crate::wgsl::render::{grid_of, gram_role, validate_text, Tree, BLK, MAX_THREADS};
use crate::wgsl::safe_reader::read_safe;
use crate::wgsl::target::Target;

/// The register budget of a thread's block: the private accumulator arrays
/// hold 64 values (`pa`..`pf` of f32, `pv`..`py` of 16 vec4).
pub const BLOCK_REGISTERS: u32 = 64;

/// A knob a kernel declares.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    /// The grid's workgroup x (`@workgroup_size` x).
    Wx,
    /// The grid's workgroup y.
    Wy,
    /// The column block factor.
    Bx,
    /// The row block factor.
    By,
    /// The tile depth T (`grid.with_ktile`).
    Ktile,
    /// The declared length of a workgroup array (its leaf, `wg.va`), when it
    /// is a literal (a length that references T moves with T).
    Len(String),
}

impl Site {
    pub fn name(&self) -> String {
        match self {
            Site::Wx => "wx".into(),
            Site::Wy => "wy".into(),
            Site::Bx => "bx".into(),
            Site::By => "by".into(),
            Site::Ktile => "ktile".into(),
            Site::Len(a) => format!("len:{a}"),
        }
    }

    /// The values the knob may take (the lookup): powers of two within the
    /// adapter's per-axis limit for a workgroup dimension, {1, 2, 4, 8} for
    /// a block factor, powers of two to 4,096 for the tile depth (the item
    /// loop's stride) and to 8,192 for an array's length.
    pub fn values(&self, target: &Target) -> Vec<u32> {
        let pow2 = |max: u32| (0..=13).map(|j| 1u32 << j).filter(|&v| v <= max).collect::<Vec<u32>>();
        match self {
            Site::Wx => pow2(target.max_size[0].min(MAX_THREADS)),
            Site::Wy => pow2(target.max_size[1].min(MAX_THREADS)),
            Site::Bx | Site::By => BLK.to_vec(),
            Site::Ktile => pow2(4096),
            Site::Len(_) => pow2(8192),
        }
    }
}

/// The grid row (`grid.of` / `grid.of_rows_x`) inside a grid's wrappers.
fn grid_row(g: &Tree) -> Option<&Tree> {
    match g.name.as_str() {
        "grid.of" | "grid.of_rows_x" => Some(g),
        "grid.with_wg" | "grid.with_ktile" => g.kids.first().and_then(grid_row),
        _ => None,
    }
}

fn plain_literal(t: &Tree) -> Option<u32> {
    (t.kids.is_empty() && t.name.starts_with("literal.") && parse_ref(&t.name).is_none() && t.konst.is_finite() && t.konst >= 0.0).then(|| t.konst.round() as u32)
}

/// The workgroup array declarations of a grid, (array leaf, length leaf).
fn decls(g: &Tree, out: &mut Vec<(String, Tree)>) {
    if g.name.starts_with("wg.decl_") && g.kids.len() == 2 {
        out.push((g.kids[0].name.clone(), g.kids[1].clone()));
    }
    for k in &g.kids {
        if k.name != "grid.of" && k.name != "grid.of_rows_x" {
            decls(k, out);
        }
    }
}

/// The knobs a safe kernel declares, with their current values.
pub fn sites(root: &Tree) -> Vec<(Site, u32)> {
    let Some(g) = root.kids.first().filter(|_| root.name == "kernel.entry_safe") else { return Vec::new() };
    let mut out = Vec::new();
    if let (Some(row), Ok(grid)) = (grid_row(g), grid_of(g)) {
        if row.kids.len() == 4 && row.kids.iter().all(|k| plain_literal(k).is_some()) {
            out.extend([(Site::Wx, grid.x), (Site::Wy, grid.y), (Site::Bx, grid.bx), (Site::By, grid.by)]);
        }
    }
    if let Some(t) = ktile_of(g) {
        out.push((Site::Ktile, t));
    }
    let mut d = Vec::new();
    decls(g, &mut d);
    for (arr, len) in d {
        if let Some(v) = plain_literal(&len) {
            out.push((Site::Len(arr), v));
        }
    }
    out
}

/// `t` with the literal at the node chosen by `pick` given the value `v`.
fn rewrite(t: &Tree, pick: &dyn Fn(&Tree) -> Option<usize>, v: u32) -> Tree {
    let mut kids: Vec<Tree> = t.kids.iter().map(|k| rewrite(k, pick, v)).collect();
    if let Some(i) = pick(t) {
        kids[i].konst = v as f32;
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

/// The kernel with the knob `site` declared as `v` (every reference to it
/// then renders from `v`).
pub fn set(root: &Tree, site: &Site, v: u32) -> Tree {
    let Some(g) = root.kids.first() else { return root.clone() };
    let g2 = match site {
        Site::Wx | Site::Wy | Site::Bx | Site::By => {
            let i = match site {
                Site::Wx => 0,
                Site::Wy => 1,
                Site::Bx => 2,
                _ => 3,
            };
            rewrite(g, &|n: &Tree| (n.name == "grid.of" || n.name == "grid.of_rows_x").then_some(i), v)
        }
        Site::Ktile => rewrite(g, &|n: &Tree| (n.name == "grid.with_ktile").then_some(1), v),
        Site::Len(arr) => rewrite(g, &|n: &Tree| (n.name.starts_with("wg.decl_") && n.kids.len() == 2 && n.kids[0].name == *arr && plain_literal(&n.kids[1]).is_some()).then_some(1), v),
    };
    let mut kids = root.kids.clone();
    kids[0] = g2;
    Tree { name: root.name.clone(), ty: root.ty, konst: root.konst, kids }
}

/// The largest value a reference of this dual may write (the renderer
/// clamps beyond it, which would break the coupling).
fn slot_max(kingdom: &WgslKingdom, t: &Tree) -> u32 {
    match kingdom.dual(t.ty).map(|d| d.form.as_str()) {
        Some("lit64") => 64,
        Some("lit4096") => 4096,
        Some("wglen") => 8192,
        _ => u32::MAX,
    }
}

/// THE JOINS of a knob child: every constraint that relates the knob to
/// the rest of the kernel and to the adapter (the workgroup memory is the
/// kernel's own arrays: no interface plays a part). Ok, or the first that fails.
pub fn joins(child: &Tree, kingdom: &WgslKingdom, target: &Target) -> Result<(), String> {
    let knobs = Knobs::of_root(child)?;
    let g = knobs.grid;
    let row = child.kids.first().and_then(grid_row).ok_or("no grid row")?;
    // The workgroup as written is the workgroup rendered (the renderer cuts y beyond the invocations).
    let written: Vec<u32> = row.kids.iter().take(2).filter_map(plain_literal).collect();
    if written != [g.x, g.y] {
        return Err(format!("workgroup {written:?} is not the {} x {} the renderer writes", g.x, g.y));
    }
    if g.x > target.max_size[0] || g.y > target.max_size[1] || g.x * g.y > target.max_invocations {
        return Err(format!("workgroup {} x {} beyond the adapter's {:?} per axis and {} invocations", g.x, g.y, target.max_size, target.max_invocations));
    }
    if g.bx * g.by > BLOCK_REGISTERS {
        return Err(format!("block {} x {} beyond the {BLOCK_REGISTERS} block registers", g.bx, g.by));
    }
    fn refs(t: &Tree, knobs: &Knobs, kingdom: &WgslKingdom) -> Result<(), String> {
        if let Some((k, form)) = parse_ref(&t.name) {
            let kv = knobs.value(k);
            let v = exact_value(kv, t.konst, form).ok_or_else(|| format!("{} x {} / 256 of {} = {} is not exact", t.name, t.konst, k.name(), kv))?;
            if v > slot_max(kingdom, t) {
                return Err(format!("{} = {v} beyond its slot's {}", t.name, slot_max(kingdom, t)));
            }
        }
        t.kids.iter().try_for_each(|k| refs(k, knobs, kingdom))
    }
    refs(child, &knobs, kingdom)?;
    let bytes = crate::wgsl::render::workgroup_request(kingdom, child)?;
    if bytes > u64::from(target.max_workgroup_storage) {
        return Err(format!("workgroup memory {bytes} bytes beyond the adapter's {}", target.max_workgroup_storage));
    }
    Ok(())
}

/// The values of `site` other than its current one whose child passes the
/// joins and the layered check ([`check`]: a wider workgroup over a
/// workgroup array indexed `lid.x + 16u * lid.y` has two threads write one
/// element, the checker's layer 7, 2026-10-14). A workgroup array's literal
/// length only grows: without a range proof of its indices (checker layer
/// 3, not built) the length the kernel declares is the only bound known to
/// cover them (48 of 50 shortened arrays of the saved populations computed
/// wrong values on the device).
pub fn legal_values(root: &Tree, site: &Site, current: u32, kingdom: &WgslKingdom, target: &Target, under: &Under) -> Vec<u32> {
    let covers = |v: u32| !matches!(site, Site::Len(_)) || v >= current;
    site.values(target)
        .into_iter()
        .filter(|&v| {
            if v == current || !covers(v) {
                return false;
            }
            let child = set(root, site, v);
            joins(&child, kingdom, target).is_ok() && check_legal_under(&child, kingdom, under).is_ok()
        })
        .collect()
}

/// A knob child: the tree and the change.
#[derive(Clone, Debug)]
pub struct KnobChild {
    pub tree: Tree,
    pub site: Site,
    pub old: u32,
    pub new: u32,
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One knob mutation under the kernel's interface: a knob with a legal new
/// value chosen uniformly, then its value uniformly among the legal ones. An
/// error when no knob of the kernel has a legal new value.
pub fn mutate(root: &Tree, kingdom: &WgslKingdom, target: &Target, under: &Under, rng: &mut u64) -> Result<KnobChild, String> {
    let options: Vec<(Site, u32, Vec<u32>)> = sites(root).into_iter().map(|(s, cur)| { let vs = legal_values(root, &s, cur, kingdom, target, under); (s, cur, vs) }).filter(|(_, _, vs)| !vs.is_empty()).collect();
    if options.is_empty() {
        return Err("no knob of this kernel has a legal new value".into());
    }
    let (site, old, values) = &options[(next(rng) % options.len() as u64) as usize];
    let new = values[(next(rng) % values.len() as u64) as usize];
    Ok(KnobChild { tree: set(root, site, new), site: site.clone(), old: *old, new })
}

/// The checker layer a child failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The legality checker (`check.rs`: typed tree, grammar, range,
    /// barrier uniformity, adapter, bounded loops).
    Legal,
    Render,
    Validate,
    ReadBack,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Legal => "legality",
            Stage::Render => "render",
            Stage::Validate => "validate",
            Stage::ReadBack => "read back",
        }
    }
}

/// THE INTERFACE A KERNEL IS CHECKED AND RENDERED UNDER: the kingdom's role
/// for it (the Gram's, a declared one, or the one inferred from a problem's
/// kernel, `render::Role::inferred`) and the dispatch contract the legality
/// checker's layer 7 proves under (`safety::Contract`; None: the role's
/// default, `Contract::for_role`). A problem directory gives both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Under {
    pub role: crate::wgsl::render::Role,
    pub contract: Option<crate::wgsl::safety::Contract>,
}

impl Under {
    /// The Gram's interface under its default contract: the Gram problem's
    /// own, for its tests and tools (no check defaults to it).
    pub fn gram() -> Under {
        Under { role: gram_role(), contract: None }
    }

    /// The interface a kernel was read under (`SafeRead::role`), the
    /// contract the checker derives from it.
    pub fn of(role: &crate::wgsl::render::Role) -> Under {
        Under { role: role.clone(), contract: None }
    }
}

/// THE LAYERED CHECK on the run path (2026-10-13: the evolution path is
/// genes, tree, text, run; nothing is parsed back), under the kernel's
/// interface: the legality checker's six layers (`check.rs`, the ownership
/// proof among them) under its role and contract, then the text rendered
/// under its role (with its `// grid` header, from the child's own grid) and
/// validated by naga for the adapter. The text, or the stage and what failed.
/// (No role-less form since 2026-10-19: the Gram's was the default.)
pub fn check_legal_under(child: &Tree, kingdom: &WgslKingdom, under: &Under) -> Result<String, (Stage, String)> {
    // The legality checker reads values: the references written as their literals.
    let plain = crate::wgsl::params::to_literals(child, kingdom).map_err(|e| (Stage::Legal, e))?;
    let checker = match &under.contract {
        Some(c) => crate::wgsl::check::Checker::with_contract(kingdom, &under.role, &Target::current(), c.clone()),
        None => crate::wgsl::check::Checker::new(kingdom, &under.role, &Target::current()),
    };
    checker.check(&plain).map_err(|e| (Stage::Legal, e.to_string()))?;
    let text = crate::wgsl::safe_karva::render_tree_with(child, kingdom, &under.role).map_err(|e| (Stage::Render, e))?.text;
    validate_text(&text).map_err(|e| (Stage::Validate, e.lines().next().unwrap_or("").to_string()))?;
    Ok(text)
}

/// [`check_legal_under`], then the text read back by the safe reader: a
/// second, independent ownership proof, kept for the tests and the tools (it
/// once found a race the checker passed, 2698df7), off the run path; the
/// text read back must read under the same interface. The text and the tree
/// read back, or the stage and what failed.
pub fn check_text_under(child: &Tree, kingdom: &WgslKingdom, under: &Under) -> Result<(String, crate::wgsl::safe_reader::SafeRead), (Stage, String)> {
    let text = check_legal_under(child, kingdom, under)?;
    let back = read_safe(&text, kingdom).map_err(|e| (Stage::ReadBack, e))?;
    if back.role != under.role {
        return Err((Stage::ReadBack, format!("the text reads back under the interface {}, written under {}", back.role.name, under.role.name)));
    }
    Ok((text, back))
}

/// [`check_legal_under`] with the read-back: the text read back declares the
/// grid and tile depth the child does. For the tests and the tools. (A knob
/// child's check on the run path is [`check_legal_under`]: its grid and tile
/// depth are the child's own, the text rendered from them.)
pub fn check_read_back(child: &Tree, kingdom: &WgslKingdom, under: &Under) -> Result<String, (Stage, String)> {
    let (text, back) = check_text_under(child, kingdom, under)?;
    let (want, got) = (Knobs::of_root(child).map_err(|e| (Stage::ReadBack, e))?, Knobs::of_root(&back.tree).map_err(|e| (Stage::ReadBack, e))?);
    // A tile depth of 1 is the items stepped one at a time: read back, it is
    // a kernel that declares none, whose tile depth is 1.
    let depth = |k: &Knobs| k.value(crate::wgsl::params::Knob::Ktile);
    if want.grid != got.grid || depth(&want) != depth(&got) {
        return Err((Stage::ReadBack, format!("read back with grid {:?} tile depth {}, written with {:?} {}", got.grid, depth(&got), want.grid, depth(&want))));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::safe_reader::read_safe;

    fn kernel(k: &WgslKingdom) -> Tree {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/safewgsl"));
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        files.iter().find_map(|f| read_safe(&std::fs::read_to_string(f).ok()?, k).ok().filter(|r| ktile_of(&r.tree.kids[0]).is_some())).map(|r| r.tree).expect("a corpus kernel with a tile depth")
    }

    #[test]
    fn a_kernel_declares_its_grid_and_tile_depth_as_sites() {
        let k = WgslKingdom::load_safe();
        let t = kernel(&k);
        let s: Vec<Site> = sites(&t).into_iter().map(|(s, _)| s).collect();
        for want in [Site::Wx, Site::Wy, Site::Bx, Site::By, Site::Ktile] {
            assert!(s.contains(&want), "{want:?} missing from {s:?}");
        }
    }

    #[test]
    fn the_joins_refuse_what_the_adapter_or_the_references_cannot_take() {
        let k = WgslKingdom::load_safe();
        let target = Target::recorded();
        let t = kernel(&k);
        joins(&t, &k, &target).unwrap();
        let wide = set(&t, &Site::Wx, 2048);
        assert!(joins(&wide, &k, &target).is_err());
        let small = Target { max_invocations: 64, ..target };
        assert!(joins(&set(&t, &Site::Wx, 128), &k, &small).is_err());
        // A tile depth of 1 leaves a load loop of a fraction of a pass.
        assert!(legal_values(&t, &Site::Ktile, ktile_of(&t.kids[0]).unwrap(), &k, &target, &Under::gram()).iter().all(|&v| joins(&set(&t, &Site::Ktile, v), &k, &target).is_ok()));
    }

    #[test]
    fn a_literal_array_length_only_grows() {
        let k = WgslKingdom::load_safe();
        let target = Target::recorded();
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/safewgsl"));
        let mut checked = 0usize;
        for f in std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")) {
            let Ok(r) = read_safe(&std::fs::read_to_string(&f).unwrap(), &k) else { continue };
            for (site, cur) in sites(&r.tree).into_iter().filter(|(s, _)| matches!(s, Site::Len(_))) {
                let legal = legal_values(&r.tree, &site, cur, &k, &target, &Under::gram());
                assert!(legal.iter().all(|&v| v > cur), "{}: {} {cur} -> {legal:?}", f.display(), site.name());
                checked += 1;
            }
        }
        assert!(checked > 0, "no corpus kernel declares a literal array length");
    }

    #[test]
    fn a_tile_depth_change_rewrites_every_reference_in_step() {
        let k = WgslKingdom::load_safe();
        let target = Target::recorded();
        let t = kernel(&k);
        let old = ktile_of(&t.kids[0]).unwrap();
        let legal = legal_values(&t, &Site::Ktile, old, &k, &target, &Under::gram());
        assert!(!legal.is_empty(), "no legal tile depth beside {old}");
        let new = legal[0];
        let child = set(&t, &Site::Ktile, new);
        let text = check_read_back(&child, &k, &Under::gram()).unwrap_or_else(|(s, e)| panic!("{}: {e}", s.name()));
        assert!(text.contains(&format!("t = t + {new}u")), "{text}");
    }

    #[test]
    fn knob_mutations_are_deterministic_and_pass_the_check() {
        let k = WgslKingdom::load_safe();
        let target = Target::recorded();
        let t = kernel(&k);
        let (mut a, mut b) = (7u64, 7u64);
        for _ in 0..8 {
            let (x, y) = (mutate(&t, &k, &target, &Under::gram(), &mut a).unwrap(), mutate(&t, &k, &target, &Under::gram(), &mut b).unwrap());
            assert_eq!((x.site.clone(), x.old, x.new), (y.site, y.old, y.new));
            check_read_back(&x.tree, &k, &Under::gram()).unwrap_or_else(|(s, e)| panic!("{} {} -> {}: {}: {e}", x.site.name(), x.old, x.new, s.name()));
        }
    }
}
