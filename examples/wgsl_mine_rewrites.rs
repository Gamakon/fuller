//! RULES MINED FROM MODEL EDITS (`docs/PLAN_constrained_mutation.md` §6,
//! task 9), a first measured pass. Reads the saved model calls of an HGP
//! log directory, pairs each child that reads and is better than a parent
//! (lower HFF) with that parent, diffs the two typed trees
//! (`src/wgsl/tree_diff.rs`), classifies the edits, generalises them into
//! candidate rules (typed holes) and writes the report.
//!
//!   cargo run --release --features wgsl --example wgsl_mine_rewrites -- <logs/HGP/001> [--out docs/MINED_REWRITES.md] [--data <dir>]
//!
//! Child HFF: a reply's tree is matched by content to the kernels the run
//! wrote (`kernels/gNNN_rK[.pP].wgsl`, re-read by the same reader), whose
//! HFF is the row's line in that run's `fitness.log` (the P-th line of the
//! row in that generation). Parent HFF: the prompt's `HFF` line under each
//! `===== PARENT` header. Nothing here is wired into evolution.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::Tree;
use fuller::wgsl::safe_reader::read_safe;
use fuller::wgsl::tree_diff::{anti_unify, canonical_registers, class_of, diff, flatten_blocks, generalise, rule_of, rule_text, similarity, site, Edit, EditKind};

/// The similarity a reply must have to a kernel of its run and generation
/// to take that kernel's HFF when no kernel equals it.
const MATCH_SIMILARITY: f64 = 0.9;

/// The reply's kernel text: a fenced block if present, else the text from
/// the first `// grid` line (the run's own `llm::strip`, which is compiled
/// only with the `gpu` feature).
fn strip(reply: &str) -> String {
    let mut text = reply.trim().to_string();
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        let end = body.find("```").unwrap_or(body.len());
        text = body[..end].to_string();
    }
    if let Some(i) = text.find("// grid") {
        text = text[i..].to_string();
    }
    text.trim().to_string() + "\n"
}

/// A parent from a prompt: its label, HFF and kernel text.
struct Parent {
    label: String,
    hff: f64,
    text: String,
}

fn parse_prompt(prompt: &str) -> Vec<Parent> {
    let lines: Vec<&str> = prompt.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if l.starts_with("===== PARENT") {
            let label = l.trim_matches('=').trim().to_string();
            let mut hff = None;
            let mut j = i + 1;
            while j < lines.len() && !lines[j].starts_with("=====") && !lines[j].starts_with("```wgsl") {
                if let Some(rest) = lines[j].strip_prefix("HFF ") {
                    hff = rest.split(';').next().and_then(|v| v.trim().parse::<f64>().ok());
                }
                j += 1;
            }
            if j < lines.len() && lines[j].starts_with("```wgsl") {
                let start = j + 1;
                let mut end = start;
                while end < lines.len() && !lines[end].starts_with("```") {
                    end += 1;
                }
                if let Some(h) = hff {
                    out.push(Parent { label, hff: h, text: lines[start..end].join("\n") + "\n" });
                }
                i = end;
            }
        }
        i += 1;
    }
    out
}

/// `gNNN_rK[.pP][.refill|.e0].wgsl` → (gen, row, pass, refill, e0), e0
/// marking an infeasible first evaluation; None for other names.
fn kernel_name(stem: &str) -> Option<(u32, u32, u32, bool, bool)> {
    let rest = stem.strip_prefix('g')?;
    let (g, rest) = rest.split_once("_r")?;
    let mut parts = rest.split('.');
    let row = parts.next()?;
    let (mut pass, mut refill, mut e0) = (1u32, false, false);
    for p in parts {
        if p == "e0" {
            e0 = true;
        } else if p == "refill" {
            refill = true;
        } else if let Some(n) = p.strip_prefix('p') {
            pass = n.parse().ok()?;
        } else {
            return None;
        }
    }
    Some((g.parse().ok()?, row.parse().ok()?, pass, refill, e0))
}

/// The run's `fitness.log`: (gen, row, refill) → the fitness of each pass
/// in order.
fn fitness_log(path: &Path) -> BTreeMap<(u32, u32, bool), Vec<f64>> {
    let mut out: BTreeMap<(u32, u32, bool), Vec<f64>> = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(path) else { return out };
    for l in text.lines() {
        let w: Vec<&str> = l.split_whitespace().collect();
        if w.len() < 4 || w[0] == "time" || w[0] == "---" {
            continue;
        }
        let (Ok(g), Some(row)) = (w[1].parse::<u32>(), w[2].strip_prefix('r').and_then(|r| r.parse::<u32>().ok())) else { continue };
        let tag = if w.len() > 4 { w[3] } else { "" };
        if tag == "elite" {
            continue;
        }
        let Ok(f) = w[w.len() - 1].parse::<f64>() else { continue };
        out.entry((g, row, tag == "refill")).or_default().push(f);
    }
    out
}

/// One reply that read: its tree and where it came from.
struct Child {
    run: String,
    call: String,
    model: String,
    kind: String,
    tree: Tree,
    /// The tree with registers canonical (`tree_diff::canonical_registers`).
    canon: Tree,
    /// The generation in the call's id.
    gen: Option<u32>,
    hff: Option<f64>,
    how: &'static str,
    sim: f64,
    parents: Vec<(String, f64, Tree)>,
}

fn call_kind(id: &str) -> String {
    let base = id.split('_').skip(1).collect::<Vec<_>>().join("_");
    // A grammar repair's id ends `_g<attempt>` (wgsl_fit's read_model_reply).
    let grammar = base.rsplit_once("_g").is_some_and(|(_, n)| n.parse::<u32>().is_ok());
    let k = if base.contains("x_union") {
        "crossover (union)"
    } else if base.contains("x_reinvent") {
        "crossover (reinvent)"
    } else if base.contains("fresh") {
        "fresh"
    } else if base.contains('p') {
        "repair"
    } else {
        "other"
    };
    if grammar {
        format!("{k}, grammar repair")
    } else {
        k.to_string()
    }
}

/// Every row name in a tree.
fn names(t: Option<&Tree>) -> BTreeSet<String> {
    fn walk(t: &Tree, s: &mut BTreeSet<String>) {
        s.insert(t.name.clone());
        for k in &t.kids {
            walk(k, s);
        }
    }
    let mut s = BTreeSet::new();
    if let Some(t) = t {
        walk(t, &mut s);
    }
    s
}

/// A statement that sets a register or private array to a constant (a
/// literal or a splat of one): bookkeeping, not work.
fn is_constant_init(t: &Tree) -> bool {
    let constant = |v: &Tree| v.name.starts_with("literal.") || (v.name.starts_with("shape.splat") && v.kids.iter().all(|k| k.name.starts_with("literal.")));
    (t.name.starts_with("reg.set") || t.name.starts_with("parr.set")) && t.kids.last().is_some_and(constant)
}

/// Whether `inner` occurs as a sub-tree of `outer`.
fn contains(outer: &Tree, inner: &Tree) -> bool {
    outer == inner || outer.kids.iter().any(|k| contains(k, inner))
}

/// The multiset of leaves (rows and literal values) of a tree.
fn leaves(t: &Tree, out: &mut Vec<String>) {
    if t.kids.is_empty() {
        out.push(t.to_sexpr());
    }
    for k in &t.kids {
        leaves(k, out);
    }
}

/// A run's name and a generation.
type RunGen = (String, u32);

/// A kernel a run wrote: its register-canonical tree, its HFF and the
/// tree's text.
type Measured = (Tree, f64, String);

/// The literal values of a tree in preorder.
fn literals(t: &Tree, out: &mut Vec<f64>) {
    if t.name.starts_with("literal.") {
        out.push(t.konst);
    }
    for k in &t.kids {
        literals(k, out);
    }
}

/// THE EDIT CLASSES. One edit and its site in the parent (`find`).
fn classify(e: &Edit, find: &Tree) -> &'static str {
    let (b, a) = (e.before.as_ref(), e.after.as_ref());
    let (nb, na) = (names(b), names(a));
    let bname = b.map_or("", |t| t.name.as_str());
    let aname = a.map_or("", |t| t.name.as_str());
    let head = if bname.is_empty() { aname } else { bname };
    let lit = |t: Option<&Tree>| t.is_some_and(|t| t.kids.is_empty() && t.name.starts_with("literal."));
    let in_ctx = |p: &str| e.context.iter().any(|n| n.starts_with(p));
    // Literal changes, by the row that holds them.
    if e.kind == EditKind::Replace && lit(b) && lit(a) {
        return match (bname, find.name.as_str()) {
            ("literal.wgdim", _) => "workgroup shape changed",
            ("literal.blk", _) => "register block changed (cells per thread)",
            ("literal.wglen", _) => "workgroup tile length changed",
            (_, f) if f.starts_with("loop.tile") => "tile loop trip count changed",
            (_, f) if f.starts_with("loop.items") => "item loop stride or unroll changed",
            (_, f) if f.starts_with("bound.") || f.starts_with("loop.") => "loop bound changed",
            ("literal.real", _) => "float constant changed",
            _ => "index constant changed (stride, offset or tile size)",
        };
    }
    // The grid header and its workgroup declarations.
    if head.starts_with("grid.") || head.starts_with("wg.decl") || find.name.starts_with("grid.") || (in_ctx("grid.") && !in_ctx("block.")) {
        if nb.iter().any(|n| n.starts_with("wg.decl")) != na.iter().any(|n| n.starts_with("wg.decl")) || head.starts_with("wg.decl") {
            return "workgroup memory tile added, removed or resized";
        }
        return "grid header restructured";
    }
    let both_regs = |t: Option<&Tree>| t.is_some_and(|t| t.kids.is_empty() && class_of(&t.name) == "reg");
    if both_regs(b) && both_regs(a) {
        return "register renamed";
    }
    // Statements inserted, deleted or replaced, by kind.
    let statement = |t: Option<&Tree>| t.is_some_and(|t| t.ty == find.kids.first().map_or(t.ty, |k| k.ty) && find.name.starts_with("block."));
    if e.kind == EditKind::Insert || e.kind == EditKind::Delete || (e.kind == EditKind::Replace && statement(b) && statement(a)) {
        let kinds = [bname, aname];
        let init = b.is_none_or(is_constant_init) && a.is_none_or(is_constant_init);
        if init {
            return "register initialisation added, removed or changed";
        }
        if kinds.iter().any(|n| n.starts_with("stmt.barrier")) {
            return "barrier added or removed";
        }
        if kinds.iter().any(|n| n.starts_with("loop.")) {
            return "loop added, removed or replaced";
        }
        if kinds.iter().any(|n| n.starts_with("store.")) {
            return "store epilogue restructured";
        }
        if kinds.iter().any(|n| n.starts_with("stmt.if")) {
            return "guard (if) added, removed or replaced";
        }
        if kinds.iter().any(|n| n.starts_with("wg.set") || n.starts_with("wg.store")) {
            return "workgroup tile load added, removed or replaced";
        }
        return "assignment added, removed or replaced";
    }
    // Loops.
    if bname.starts_with("loop.") && aname.starts_with("loop.") && bname != aname {
        let unrolled = |n: &str| n.contains("unrolled");
        return match (unrolled(bname), unrolled(aname)) {
            (false, true) => "loop unrolled",
            (true, false) => "loop re-rolled",
            _ => "loop kind changed (strided, bounded)",
        };
    }
    if head.starts_with("bound.") {
        return "loop bound changed";
    }
    // Loads.
    let vec_load = |s: &BTreeSet<String>| s.iter().any(|n| n == "load.phi4_safe" || n.contains("get_v"));
    let scalar_load = |s: &BTreeSet<String>| s.iter().any(|n| n == "load.phi_safe" || n == "wg.get" || n == "wg.get_safe");
    if vec_load(&na) && !vec_load(&nb) && scalar_load(&nb) {
        return "load vectorised";
    }
    if vec_load(&nb) && !vec_load(&na) && scalar_load(&na) {
        return "load de-vectorised";
    }
    if head.starts_with("store.") || in_ctx("store.") {
        return "store epilogue restructured";
    }
    // Index guards: the old expression kept inside a new mask or clamp, or
    // the reverse.
    let guard = |n: &str| n.starts_with("safe.") || n == "index.min" || n == "index.clamp" || n == "bits.and" || n == "index.mod" || n.starts_with("logic.") || n.starts_with("compare.");
    if let (Some(x), Some(y)) = (b, a) {
        if guard(aname) && contains(y, x) {
            return "index masked or guarded";
        }
        if guard(bname) && contains(x, y) {
            return "index mask or guard removed";
        }
        let (mut lx, mut ly) = (Vec::new(), Vec::new());
        leaves(x, &mut lx);
        leaves(y, &mut ly);
        lx.sort();
        ly.sort();
        if lx == ly && (bname.starts_with("arith.") || aname.starts_with("arith.")) {
            return "accumulation reordered (same operands)";
        }
        if lx == ly {
            return "expression reassociated (same operands)";
        }
    }
    if e.kind == EditKind::Update {
        return "row changed in place";
    }
    if head.starts_with("index.") || head.starts_with("safe.") || head.starts_with("axis.") {
        return "index expression rewritten";
    }
    if head.starts_with("compare.") || head.starts_with("logic.") {
        return "condition rewritten";
    }
    if head.starts_with("arith.") || head.starts_with("shape.") || head.starts_with("geom.") || head.starts_with("parr.get") || head.starts_with("reg.get") || head.starts_with("wg.get") || head.starts_with("load.") {
        return "value expression rewritten";
    }
    "other"
}

struct Pair<'c> {
    child: &'c Child,
    parent_label: String,
    parent_hff: f64,
    edits: Vec<(Edit, &'static str, Tree, Tree)>,
}

fn main() -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or("usage: wgsl_mine_rewrites <logs/HGP/001> [--out <md>] [--data <dir>] | --probe <a.wgsl|reply.txt> <b.wgsl>")?;
    if first == "--probe" {
        // Two kernels' edits, the first read as a reply.
        let kingdom = WgslKingdom::load_safe();
        let a = PathBuf::from(args.next().ok_or("--probe needs two files")?);
        let b = PathBuf::from(args.next().ok_or("--probe needs two files")?);
        let ta = read_safe(&strip(&std::fs::read_to_string(&a).map_err(|e| e.to_string())?), &kingdom)?.tree;
        let tb = read_safe(&strip(&std::fs::read_to_string(&b).map_err(|e| e.to_string())?), &kingdom)?.tree;
        println!("a {} nodes, b {} nodes, equal {}", ta.size(), tb.size(), ta == tb);
        for e in diff(&flatten_blocks(&ta), &flatten_blocks(&tb)) {
            let s = |x: &Option<Tree>| x.as_ref().map_or("-".to_string(), |t| t.to_sexpr().chars().take(240).collect());
            println!("{:?}\t{}\t{}\t=>\t{}", e.kind, e.context.iter().rev().take(3).cloned().collect::<Vec<_>>().join(" < "), s(&e.before), s(&e.after));
        }
        return Ok(());
    }
    let dir = PathBuf::from(first);
    let (mut out_md, mut data): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_md = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--data" => data = Some(PathBuf::from(args.next().ok_or("--data needs a path")?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let kingdom = WgslKingdom::load_safe();
    let type_name = |t: &Tree| match t.ty {
        fuller::geneframe::Ty::Wgsl(i) => kingdom.duals.get(i as usize).map_or(format!("{:?}", t.ty), |d| d.dual.clone()),
        other => format!("{other:?}"),
    };
    let mut runs: Vec<PathBuf> = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().map(|e| e.path()).filter(|p| p.join("kernels").is_dir() || p.join("calls").is_dir()).collect();
    runs.sort();
    // THE KERNEL INDEX: register-canonical tree text → (run, HFF) of every
    // kernel a run wrote (a row's final kernel and the infeasible first
    // evaluations, `.e0`, at HFF pi/2); for the runs with saved calls, the
    // canonical trees by (run, generation) for the similarity match.
    let mut index: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    let mut by_gen: HashMap<RunGen, Vec<Measured>> = HashMap::new();
    let (mut kernels_read, mut kernels_unread, mut kernels_unscored) = (0usize, 0usize, 0usize);
    for run in &runs {
        let run_name = run.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let with_calls = run.join("calls").is_dir();
        let fit = fitness_log(&run.join("fitness.log"));
        let Ok(rd) = std::fs::read_dir(run.join("kernels")) else { continue };
        let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        files.sort();
        for f in files {
            let stem = f.file_name().map(|s| s.to_string_lossy().trim_end_matches(".wgsl").to_string()).unwrap_or_default();
            let Some((g, row, pass, refill, e0)) = kernel_name(&stem) else { continue };
            let hff = if e0 {
                std::f64::consts::FRAC_PI_2
            } else {
                match fit.get(&(g, row, refill)).and_then(|v| v.get(pass as usize - 1)).copied() {
                    Some(h) => h,
                    None => {
                        kernels_unscored += 1;
                        continue;
                    }
                }
            };
            let Ok(src) = std::fs::read_to_string(&f) else { continue };
            match read_safe(&src, &kingdom) {
                Ok(r) => {
                    kernels_read += 1;
                    let canon = canonical_registers(&r.tree);
                    let key = canon.to_sexpr();
                    index.entry(key.clone()).or_default().push((run_name.clone(), hff));
                    if with_calls {
                        by_gen.entry((run_name.clone(), g)).or_default().push((canon, hff, key));
                    }
                }
                Err(_) => kernels_unread += 1,
            }
        }
    }
    eprintln!("{:.0} s\tkernel index: {kernels_read} read, {kernels_unread} not read, {kernels_unscored} without a fitness line, {} distinct trees", t0.elapsed().as_secs_f64(), index.len());
    // THE REPLIES.
    let mut children: Vec<Child> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let (mut replies, mut distinct_texts, mut unread) = (0usize, BTreeSet::new(), 0usize);
    let mut parent_cache: HashMap<String, Option<Tree>> = HashMap::new();
    let (mut parents_total, mut parents_unread) = (0usize, 0usize);
    let mut match_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    for run in &runs {
        let run_name = run.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let Ok(rd) = std::fs::read_dir(run.join("calls")) else { continue };
        let mut calls: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".reply.txt")).collect();
        calls.sort();
        for c in calls {
            replies += 1;
            let fname = c.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let stem = fname.trim_end_matches(".reply.txt");
            let (id, model) = stem.rsplit_once('.').unwrap_or((stem, "?"));
            let reply = std::fs::read_to_string(&c).map_err(|e| e.to_string())?;
            let text = strip(&reply);
            if !distinct_texts.insert(text.clone()) {
                continue;
            }
            let Ok(r) = read_safe(&text, &kingdom) else {
                unread += 1;
                continue;
            };
            let canon = canonical_registers(&r.tree);
            let key = canon.to_sexpr();
            if !seen.insert(key.clone()) {
                continue;
            }
            // The child's HFF, first pass: the same kernel (registers aside)
            // in this run, else in any run. The similarity pass follows.
            let gen: Option<u32> = id.strip_prefix('g').and_then(|x| x.get(..3)).and_then(|x| x.parse().ok());
            let (hff, how, sim) = match index.get(&key) {
                Some(v) => match v.iter().find(|(rn, _)| *rn == run_name) {
                    Some(x) => (Some(x.1), "same kernel, this run", 1.0),
                    None => (v.first().map(|x| x.1), "same kernel, another run", 1.0),
                },
                None => (None, "no match", 0.0),
            };
            let prompt = std::fs::read_to_string(run.join("calls").join(format!("{stem}.prompt.txt"))).unwrap_or_default();
            let mut parents = Vec::new();
            for p in parse_prompt(&prompt) {
                parents_total += 1;
                let t = parent_cache.entry(p.text.clone()).or_insert_with(|| read_safe(&p.text, &kingdom).ok().map(|r| r.tree)).clone();
                match t {
                    Some(t) => parents.push((p.label, p.hff, t)),
                    None => parents_unread += 1,
                }
            }
            children.push(Child { run: run_name.clone(), call: id.to_string(), model: model.to_string(), kind: call_kind(id), tree: r.tree, canon, gen, hff, how, sim, parents });
        }
    }
    // The similarity pass, one to one: a kernel of the call's run and
    // generation that no reply equals goes to the reply most like it (at
    // least MATCH_SIMILARITY), most similar pairs first. A failed child
    // whose repair overwrote its kernel file thus does not take the
    // repair's HFF: the repair reply is more like that kernel and claims it.
    let exact: BTreeSet<String> = children.iter().filter(|c| c.hff.is_some()).map(|c| c.canon.to_sexpr()).collect();
    let mut cands: Vec<(f64, usize, (String, u32), usize)> = Vec::new();
    for (ci, c) in children.iter().enumerate() {
        if c.hff.is_some() {
            continue;
        }
        let Some(g) = c.gen else { continue };
        let k = (c.run.clone(), g);
        for (ki, (t, _, tk)) in by_gen.get(&k).map_or(&[][..], |v| v.as_slice()).iter().enumerate() {
            if exact.contains(tk) {
                continue;
            }
            let ratio = t.size().min(c.canon.size()) as f64 / t.size().max(c.canon.size()) as f64;
            if ratio < MATCH_SIMILARITY {
                continue;
            }
            let sim = similarity(&c.canon, t);
            if sim >= MATCH_SIMILARITY {
                cands.push((sim, ci, k.clone(), ki));
            }
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.3.cmp(&b.3)));
    let mut claimed: BTreeSet<((String, u32), usize)> = BTreeSet::new();
    for (sim, ci, k, ki) in cands {
        if children[ci].hff.is_some() || claimed.contains(&(k.clone(), ki)) {
            continue;
        }
        let h = by_gen[&k][ki].1;
        claimed.insert((k, ki));
        let c = &mut children[ci];
        c.hff = Some(h);
        c.how = "most similar unclaimed kernel of its run and generation";
        c.sim = sim;
    }
    for c in &children {
        *match_kinds.entry(c.how).or_default() += 1;
    }
    eprintln!("{:.0} s\tmatches: {match_kinds:?}", t0.elapsed().as_secs_f64());
    let matched = children.iter().filter(|c| c.hff.is_some()).count();
    let feasible = children.iter().filter(|c| c.hff.is_some_and(|h| h < 1.5)).count();
    eprintln!("{:.0} s\treplies {replies}, distinct texts {}, not read {unread}, distinct trees read {}, with a measured HFF {matched} (feasible {feasible}); parents {parents_total} ({parents_unread} not read)", t0.elapsed().as_secs_f64(), distinct_texts.len(), children.len());
    // THE PAIRS: a child better than a parent; every such parent.
    let mut pairs: Vec<Pair> = Vec::new();
    for c in &children {
        let Some(h) = c.hff else { continue };
        if h >= 1.5 {
            continue;
        }
        for (label, ph, pt) in &c.parents {
            if h < *ph {
                let pf = flatten_blocks(pt);
                let edits = diff(&pf, &flatten_blocks(&c.tree))
                    .into_iter()
                    .filter_map(|e| {
                        let (f, r) = site(&pf, &e)?;
                        let class = classify(&e, &f);
                        Some((e, class, f, r))
                    })
                    .collect();
                pairs.push(Pair { child: c, parent_label: label.clone(), parent_hff: *ph, edits });
            }
        }
    }
    eprintln!("{:.0} s\tpairs {} (children {})", t0.elapsed().as_secs_f64(), pairs.len(), pairs.iter().map(|p| p.child.call.clone() + &p.child.run).collect::<BTreeSet<_>>().len());
    // THE CLASSES: per class, the pairs showing it, its edits, the mean
    // gain over those pairs, and the gain over pairs showing only it.
    struct ClassStat {
        pairs: BTreeSet<usize>,
        edits: usize,
        examples: Vec<(usize, usize)>,
    }
    let mut classes: BTreeMap<&'static str, ClassStat> = BTreeMap::new();
    for (pi, p) in pairs.iter().enumerate() {
        for (ei, (_, class, _, _)) in p.edits.iter().enumerate() {
            let s = classes.entry(class).or_insert_with(|| ClassStat { pairs: BTreeSet::new(), edits: 0, examples: Vec::new() });
            s.pairs.insert(pi);
            s.edits += 1;
            s.examples.push((pi, ei));
        }
    }
    let gain = |pi: usize| pairs[pi].parent_hff - pairs[pi].child.hff.unwrap_or(0.0);
    let mut class_rows: Vec<(&'static str, &ClassStat, f64, usize, f64)> = classes
        .iter()
        .map(|(k, s)| {
            let mean = s.pairs.iter().map(|&p| gain(p)).sum::<f64>() / s.pairs.len() as f64;
            let without: Vec<usize> = (0..pairs.len()).filter(|p| !s.pairs.contains(p)).collect();
            let without_mean = if without.is_empty() { f64::NAN } else { without.iter().map(|&p| gain(p)).sum::<f64>() / without.len() as f64 };
            (*k, s, mean, without.len(), without_mean)
        })
        .collect();
    class_rows.sort_by(|a, b| b.1.pairs.len().cmp(&a.1.pairs.len()).then(b.1.edits.cmp(&a.1.edits)).then(a.0.cmp(b.0)));
    // THE RULES: edits grouped by generalised shape; support = distinct
    // pairs (and distinct children); the rule shown is the anti-unification
    // of the group's concrete (context-holed) instances.
    struct RuleGroup {
        class: &'static str,
        place: String,
        instances: Vec<(Tree, Tree)>,
        pairs: BTreeSet<usize>,
        children: BTreeSet<String>,
        shape: (Tree, Tree),
    }
    let mut groups: BTreeMap<String, RuleGroup> = BTreeMap::new();
    for (pi, p) in pairs.iter().enumerate() {
        for (e, class, f, r) in &p.edits {
            let (cf, cr) = rule_of(f, r);
            if cf.size() + cr.size() > 120 {
                continue;
            }
            let (gf, gr) = generalise(&cf, &cr);
            // Where: the nearest enclosing row that is not a statement list
            // (for a site that is a list window, the loop or branch it is in).
            let place = if f.name.starts_with("block.") { e.context.iter().rev().find(|n| !n.starts_with("block.")).cloned().unwrap_or_default() } else { String::new() };
            let key = format!("{place} | {} => {}", gf.to_sexpr(), gr.to_sexpr());
            let g = groups.entry(key).or_insert_with(|| RuleGroup { class, place, instances: Vec::new(), pairs: BTreeSet::new(), children: BTreeSet::new(), shape: (gf, gr) });
            g.instances.push((cf, cr));
            g.pairs.insert(pi);
            g.children.insert(format!("{}/{}", p.child.run, p.child.call));
        }
    }
    let mut rules: Vec<(&RuleGroup, f64, String)> = groups
        .values()
        .filter(|g| g.children.len() >= 2)
        .map(|g| {
            let mean = g.pairs.iter().map(|&p| gain(p)).sum::<f64>() / g.pairs.len() as f64;
            let text = anti_unify(&g.instances).map_or_else(|| rule_text(&g.shape.0, &g.shape.1, &type_name), |(f, r)| rule_text(&f, &r, &type_name));
            // The literal values the instances changed, most frequent first
            // (where a literal hole hides them).
            let mut seen: BTreeMap<String, usize> = BTreeMap::new();
            for (f, r) in &g.instances {
                let (mut lf, mut lr) = (Vec::new(), Vec::new());
                literals(f, &mut lf);
                literals(r, &mut lr);
                if lf.len() == lr.len() {
                    let changed: Vec<String> = lf.iter().zip(&lr).filter(|(x, y)| x != y).map(|(x, y)| format!("{x}→{y}")).collect();
                    if !changed.is_empty() {
                        *seen.entry(changed.join(",")).or_default() += 1;
                    }
                }
            }
            let mut seen: Vec<(String, usize)> = seen.into_iter().collect();
            seen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let observed = seen.iter().take(4).map(|(k, n)| format!("{k} ×{n}")).collect::<Vec<_>>().join("; ");
            let text = if observed.is_empty() { text } else { format!("{text}  [values: {observed}]") };
            (g, mean, text)
        })
        .collect();
    rules.sort_by(|a, b| b.0.children.len().cmp(&a.0.children.len()).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
    eprintln!("{:.0} s\tclasses {}, rule shapes {}, with support from two children or more {}", t0.elapsed().as_secs_f64(), class_rows.len(), groups.len(), rules.len());
    // THE DATA FILES.
    if let Some(d) = &data {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        let mut t = String::from("run\tcall\tmodel\tkind\tmatch\tmatch_similarity\tparent\tparent_hff\tchild_hff\tgain\tparent_nodes\tchild_nodes\tedits\tclasses\n");
        for p in &pairs {
            let cls: BTreeSet<&str> = p.edits.iter().map(|e| e.1).collect();
            let pn = p.child.parents.iter().find(|x| x.0 == p.parent_label).map_or(0, |x| x.2.size());
            writeln!(t, "{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}", p.child.run, p.child.call, p.child.model, p.child.kind, p.child.how, p.child.sim, p.parent_label, p.parent_hff, p.child.hff.unwrap_or(f64::NAN), p.parent_hff - p.child.hff.unwrap_or(0.0), pn, p.child.tree.size(), p.edits.len(), cls.into_iter().collect::<Vec<_>>().join("; ")).map_err(|e| e.to_string())?;
        }
        std::fs::write(d.join("pairs.tsv"), t).map_err(|e| e.to_string())?;
        let mut t = String::from("children\tpairs\tinstances\tmean_gain\tclass\tin\trule\n");
        for (g, mean, text) in &rules {
            writeln!(t, "{}\t{}\t{}\t{:.4}\t{}\t{}\t{}", g.children.len(), g.pairs.len(), g.instances.len(), mean, g.class, g.place, text).map_err(|e| e.to_string())?;
        }
        std::fs::write(d.join("rules.tsv"), t).map_err(|e| e.to_string())?;
        let mut t = String::from("pair\tclass\tkind\tcontext\tbefore\tafter\n");
        for (pi, p) in pairs.iter().enumerate() {
            for (e, class, _, _) in &p.edits {
                let s = |x: &Option<Tree>| x.as_ref().map_or("-".to_string(), |t| t.to_sexpr().chars().take(300).collect());
                writeln!(t, "{pi}\t{class}\t{:?}\t{}\t{}\t{}", e.kind, e.context.iter().rev().take(3).cloned().collect::<Vec<_>>().join(" < "), s(&e.before), s(&e.after)).map_err(|e| e.to_string())?;
            }
        }
        std::fs::write(d.join("edits.tsv"), t).map_err(|e| e.to_string())?;
    }
    // THE REPORT.
    let Some(md) = out_md else { return Ok(()) };
    let mut s = String::new();
    let n_children = pairs.iter().map(|p| format!("{}/{}", p.child.run, p.child.call)).collect::<BTreeSet<_>>().len();
    let edits_total: usize = pairs.iter().map(|p| p.edits.len()).sum();
    let mut sizes: Vec<usize> = pairs.iter().map(|p| p.edits.len()).collect();
    sizes.sort_unstable();
    let median = sizes.get(sizes.len() / 2).copied().unwrap_or(0);
    writeln!(s, "# Rewrite rules mined from model edits (first pass)\n").map_err(|e| e.to_string())?;
    writeln!(s, "Generated by `cargo run --release --features wgsl --example wgsl_mine_rewrites -- {} --out docs/MINED_REWRITES.md --data docs/mined_rewrites` (`src/wgsl/tree_diff.rs`), `docs/PLAN_constrained_mutation.md` §6, task 9. The rules below are candidates only; none is wired into evolution.\n", dir.display()).map_err(|e| e.to_string())?;
    writeln!(s, "## Data\n").map_err(|e| e.to_string())?;
    writeln!(s, "| measure | count |\n|---|---|").map_err(|e| e.to_string())?;
    for (k, v) in [
        ("model replies saved (all runs)", replies),
        ("distinct reply texts", distinct_texts.len()),
        ("distinct texts the reader does not read", unread),
        ("distinct trees read", children.len()),
        ("of those, matched to a measured kernel (HFF known)", matched),
        ("of those, feasible", feasible),
        ("parent kernels in the prompts", parents_total),
        ("parent kernels the reader does not read", parents_unread),
        ("pairs: child feasible and lower HFF than the parent", pairs.len()),
        ("children in those pairs", n_children),
        ("edits in those pairs", edits_total),
        ("median edits per pair", median),
    ] {
        writeln!(s, "| {k} | {v} |").map_err(|e| e.to_string())?;
    }
    for (k, v) in &match_kinds {
        writeln!(s, "| distinct trees whose HFF came from: {k} | {v} |").map_err(|e| e.to_string())?;
    }
    let paired_how: BTreeMap<&str, usize> = pairs.iter().fold(BTreeMap::new(), |mut m, p| {
        *m.entry(p.child.how).or_default() += 1;
        m
    });
    for (k, v) in &paired_how {
        writeln!(s, "| pairs whose child HFF came from: {k} | {v} |").map_err(|e| e.to_string())?;
    }
    writeln!(s, "\nA child is paired with every parent in its prompt that it beats (a crossover child with both parents when it beats both). Kernels the run wrote were re-read by the same reader to match a reply to its measured HFF; a reply whose kernel was varied before scoring (deduplication), or replaced after an infeasible first evaluation, has no match and is left out. The gain of a pair is parent HFF minus child HFF (higher is better).\n").map_err(|e| e.to_string())?;
    writeln!(s, "## Edit classes\n").map_err(|e| e.to_string())?;
    writeln!(s, "Each edit is the smallest pair of differing sub-trees with everything around them identical (GumTree matching, then an aligned top-down walk). Most replies rewrite the whole kernel, so a pair carries many edits of many classes; the mean gain of a class is over the pairs that show it and is confounded by the other edits in those pairs. No pair here has edits of one class only, so the last two columns give the mean gain of the pairs that do not show the class, for comparison. Register initialisation and register renaming are bookkeeping of the grammar (every register a kernel uses is set to zero at the top of its block; allocation order differs between kernels), not changes of the computation.\n").map_err(|e| e.to_string())?;
    let all_mean = (0..pairs.len()).map(gain).sum::<f64>() / pairs.len().max(1) as f64;
    writeln!(s, "| class | pairs | edits | mean gain (pairs showing it) | pairs without it | mean gain (pairs without it) |\n|---|---|---|---|---|---|").map_err(|e| e.to_string())?;
    writeln!(s, "| (all pairs) | {} | {edits_total} | {all_mean:.4} | - | - |", pairs.len()).map_err(|e| e.to_string())?;
    for (k, st, mean, without, without_mean) in &class_rows {
        let wm = if without_mean.is_nan() { "-".to_string() } else { format!("{without_mean:.4}") };
        writeln!(s, "| {k} | {} | {} | {mean:.4} | {without} | {wm} |", st.pairs.len(), st.edits).map_err(|e| e.to_string())?;
    }
    writeln!(s, "\n## Three examples per class\n").map_err(|e| e.to_string())?;
    writeln!(s, "Before and after are the edit's site in the parent's typed tree and the same site with only this edit applied (s-expressions over the kingdom's rows; long sites cut at 400 characters). Examples are the edits of the class with the smallest sites, from different children.\n").map_err(|e| e.to_string())?;
    for (k, st, _, _, _) in &class_rows {
        writeln!(s, "### {k}\n").map_err(|e| e.to_string())?;
        let mut ex: Vec<&(usize, usize)> = st.examples.iter().collect();
        ex.sort_by_key(|(pi, ei)| {
            let (_, _, f, r) = &pairs[*pi].edits[*ei];
            (f.size() + r.size(), *pi, *ei)
        });
        let mut used: BTreeSet<String> = BTreeSet::new();
        let mut shown = 0;
        for (pi, ei) in ex {
            let p = &pairs[*pi];
            let who = format!("{}/{}", p.child.run, p.child.call);
            if used.contains(&who) {
                continue;
            }
            let (e, _, f, r) = &p.edits[*ei];
            // A site too small to say anything (a lone leaf) is skipped.
            if f.size() + r.size() < 3 {
                continue;
            }
            used.insert(who.clone());
            let cut = |t: &Tree| -> String {
                let x = t.to_sexpr();
                if x.chars().count() > 400 {
                    x.chars().take(400).collect::<String>() + " …"
                } else {
                    x
                }
            };
            writeln!(s, "{}. `{who}` ({}, {}), {} HFF {:.4} → child {:.4}; {:?} at `{}`\n   - before: `{}`\n   - after: `{}`", shown + 1, p.child.model, p.child.kind, p.parent_label, p.parent_hff, p.child.hff.unwrap_or(f64::NAN), e.kind, e.context.iter().rev().take(3).cloned().collect::<Vec<_>>().join(" < "), cut(f), cut(r)).map_err(|e| e.to_string())?;
            shown += 1;
            if shown == 3 {
                break;
            }
        }
        writeln!(s).map_err(|e| e.to_string())?;
    }
    writeln!(s, "## Candidate rules\n").map_err(|e| e.to_string())?;
    writeln!(s, "An edit's site becomes `find ==> replace`: sub-trees common to both sides become shared holes, then literals and variables (registers, private arrays) become typed holes (the magic symbol table's generalisation). Edits with the same generalised shape form one candidate; the rule shown is the anti-unification of its concrete instances, so what every instance agrees on (a literal 4 → 8, a row name) stays concrete. `?n:type` is a hole of that kingdom type. A site inside a statement list is a window (the changed statement and its neighbours) and the rule records the enclosing row it applies in (`in`). Ranked by support (distinct children, two or more), then mean gain; the mean gain is over the pairs showing the rule, which carry many other edits (see the classes). Sites over 120 nodes are left out.\n").map_err(|e| e.to_string())?;
    let bookkeeping = |c: &str| c.starts_with("register initialisation") || c == "register renamed";
    let table = |s: &mut String, rows: &[&(&RuleGroup, f64, String)]| -> Result<(), String> {
        writeln!(s, "| rank | children | pairs | instances | mean gain | class | in | rule |\n|---|---|---|---|---|---|---|---|").map_err(|e| e.to_string())?;
        for (i, (g, mean, text)) in rows.iter().enumerate() {
            let t: String = text.replace('|', "\\|");
            let t = if t.chars().count() > 600 { t.chars().take(600).collect::<String>() + " …" } else { t };
            writeln!(s, "| {} | {} | {} | {} | {mean:.4} | {} | {} | `{t}` |", i + 1, g.children.len(), g.pairs.len(), g.instances.len(), g.class, g.place).map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let work: Vec<&(&RuleGroup, f64, String)> = rules.iter().filter(|r| !bookkeeping(r.0.class)).take(40).collect();
    table(&mut s, &work)?;
    writeln!(s, "\n### Bookkeeping rules (register initialisation and renaming)\n\nThe same mining over the bookkeeping classes, top ten; these change no computation and are listed so their support is visible, not as candidates.\n").map_err(|e| e.to_string())?;
    let books: Vec<&(&RuleGroup, f64, String)> = rules.iter().filter(|r| bookkeeping(r.0.class)).take(10).collect();
    table(&mut s, &books)?;
    writeln!(s, "\n{} candidate rules have support from two children or more; the full list is `docs/mined_rewrites/rules.tsv`, the pairs `pairs.tsv`, every edit `edits.tsv`.", rules.len()).map_err(|e| e.to_string())?;
    std::fs::write(&md, s).map_err(|e| e.to_string())?;
    eprintln!("{:.0} s\twrote {}", t0.elapsed().as_secs_f64(), md.display());
    Ok(())
}
