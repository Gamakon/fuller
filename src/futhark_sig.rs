//! The SIGNATURE FAMILY of the shape kingdom's generator: a Futhark kingdom whose interface is generated from a
//! program's declared signature (`kingdoms/futhark_shapes/gen_tables.py`, section "THE SIGNATURE FAMILY"; one instance
//! a problem, under `kingdoms/futhark_shapes/sig/<problem>/`), in the direction of "any Futhark program: inputs and
//! outputs are any typed values".
//!
//! An instance declares its inputs and output as typed arrays over named dimensions — Rodinia's pathfinder is
//! `walls: [rows][cols]i32 -> [cols]i32` — and the generator derives the rest, every rule of a legal program a type:
//! a class (`a.rows.cols`, `a.cols`, `ix.cols`, `i32`) in a context (the binders around the node: `tabulate`, a
//! `loop` over a dimension carrying a state, the same loop from 1). An index is in range by its type (its terminals
//! are `0`, `D - 1` and the counters of binders over `D`; its functions clamp), a variable is read only in a context
//! that binds it, and a binder, a reduction or a `replicate` exists only above the family's depth, so a node runs at
//! most `max(D) ** depth` times. Every gene expresses a program Futhark accepts; the tests here check that Futhark
//! agrees and that the compiled program computes what the host interpreter says.
//!
//! What this module holds: the loader and the declared signature, Futhark's binary value format for the cases, the
//! program around an expressed tree (the entry point binding the interface), the host interpreter (the meaning of a
//! gene, a test oracle, exact `i32` arithmetic as Futhark's: wrapping), the exact work of a tree on given sizes, a
//! builder that writes a program given as an s-expression of the family's rows as a tree, the problem registry (the
//! instance, its reference written as a gene, the published program as the speed yardstick, and its stored cases),
//! and the instance's e-graph (one sort a type, constants by their pool index).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use egglog::extract::hff_extract;
use egglog::prelude::exprs;
use egglog::{EGraph, Term, TermDag, TermId};

use crate::futhark_kingdom::{FutharkFamily, FutharkKind, FutharkKingdom, GeneRng};
use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::saturated_fold::EgraphKingdom;
use crate::typed_gep::{Domain, GeneLayout, Tree};

/// One instance of the family: the generated artefacts of one declared signature.
#[derive(Debug, Clone, Copy)]
pub struct Instance {
    pub name: &'static str,
    types: &'static str,
    functions: &'static str,
    interface: &'static str,
    pub datatype: &'static str,
    pub rules: &'static str,
    egraph_rows: &'static str,
}

/// Rodinia pathfinder: `walls: [rows][cols]i32 -> [cols]i32`.
pub const PATHFINDER: Instance = Instance {
    name: "pathfinder",
    types: include_str!("../kingdoms/futhark_shapes/sig/pathfinder/types.tsv"),
    functions: include_str!("../kingdoms/futhark_shapes/sig/pathfinder/functions.tsv"),
    interface: include_str!("../kingdoms/futhark_shapes/sig/pathfinder/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark_shapes/sig/pathfinder/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark_shapes/sig/pathfinder/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark_shapes/sig/pathfinder/egraph.rows"),
};

/// The study's micro benchmark reduce (`micro/reduce.fut`, `sum_i32`): `xs: [n]i32 -> i32`.
pub const REDUCE: Instance = Instance {
    name: "reduce",
    types: include_str!("../kingdoms/futhark_shapes/sig/reduce/types.tsv"),
    functions: include_str!("../kingdoms/futhark_shapes/sig/reduce/functions.tsv"),
    interface: include_str!("../kingdoms/futhark_shapes/sig/reduce/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark_shapes/sig/reduce/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark_shapes/sig/reduce/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark_shapes/sig/reduce/egraph.rows"),
};

/// The study's micro benchmark scan (`micro/scan.fut`, `sum_i32`): `xs: [n]i32 -> [n]i32`, inclusive prefix sums.
pub const SCAN: Instance = Instance {
    name: "scan",
    types: include_str!("../kingdoms/futhark_shapes/sig/scan/types.tsv"),
    functions: include_str!("../kingdoms/futhark_shapes/sig/scan/functions.tsv"),
    interface: include_str!("../kingdoms/futhark_shapes/sig/scan/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark_shapes/sig/scan/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark_shapes/sig/scan/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark_shapes/sig/scan/egraph.rows"),
};

/// The study's histogram (`micro/reduce_by_index.fut`, `sum_i32`): `(k: i32) (is: [n]i32) (vs: [n]i32) -> [k]i32`,
/// the sum of the values in each bin `is % k`.
pub const HISTOGRAM: Instance = Instance {
    name: "histogram",
    types: include_str!("../kingdoms/futhark_shapes/sig/histogram/types.tsv"),
    functions: include_str!("../kingdoms/futhark_shapes/sig/histogram/functions.tsv"),
    interface: include_str!("../kingdoms/futhark_shapes/sig/histogram/interface.tsv"),
    datatype: include_str!("../kingdoms/futhark_shapes/sig/histogram/egraph.datatype"),
    rules: include_str!("../kingdoms/futhark_shapes/sig/histogram/egraph.rules"),
    egraph_rows: include_str!("../kingdoms/futhark_shapes/sig/histogram/egraph.rows"),
};

/// The ruleset name the generated rules are in.
pub const RULESET: &str = "futhark_sig";

/// A problem: a signature instance, its published program written as a gene of the kingdom (an s-expression of the
/// family's rows, for [`build`]), the published program itself as the speed yardstick (a Futhark source whose `main`
/// has the declared signature), and the directory of its stored cases.
#[derive(Debug, Clone, Copy)]
pub struct Problem {
    pub name: &'static str,
    pub instance: Instance,
    pub reference: &'static str,
    pub yardstick: &'static str,
    /// THE PLANTED SLOWDOWN (a control of regime P): an equal program that does the work twice, which the search must
    /// recover from. For pathfinder: the elementwise least of the reference and its re-associated twin (the `min`s
    /// grouped the other way), which Futhark cannot merge and the e-graph can (associativity puts the twins in one
    /// class, and `min a a = a` removes the `map2`).
    pub planted: &'static str,
    /// Relative to the crate's root.
    pub corpus: &'static str,
    /// THE TIMING INPUT's size (`n`; 0: time on the largest training case). A micro benchmark is timed on its own
    /// large input ([`timing_input`]), used only for timing and never as a quality case: measured 2026-10-27, reduce
    /// timed on its 1,000-element training case gave A/A ratios in three clusters (0.30, 1.00, 3.3), the GPU's clock
    /// states on a 10-30 us kernel, so selection in stage 2 chose on noise. At most 65,535 x 256: wgpu allows 65,535
    /// workgroups a dispatch dimension, and an elementwise map over `n` dispatches `n / 256` (measured 2026-10-28: at
    /// n = 2^24 every champion with a map over `n` failed, "[65536, 1, 1] must be less or equal to 65535"), so the
    /// micro benchmarks use 15 x 2^20 (61,440 workgroups).
    pub timing_n: usize,
}

/// The timing input of a problem with `timing_n > 0`, in Futhark's binary format (the entry point's parameters):
/// `n` values uniform over the whole `i32` range (the histogram: k = 100, indices and values so drawn), a fixed seed.
pub fn timing_input(p: &Problem) -> Result<Option<Vec<u8>>, String> {
    let n = p.timing_n;
    if n == 0 {
        return Ok(None);
    }
    let mut rng = GeneRng::new(0x7131_6E00 ^ n as u64);
    let mut any = |n: usize| Array { dims: vec![n], data: (0..n).map(|_| rng.next_u64() as u32 as i32).collect() };
    match p.name {
        "reduce" | "scan" => Ok(Some(write_binary(&[&any(n)]))),
        "histogram" => {
            let (is, vs) = (any(n), any(n));
            Ok(Some(write_binary(&[&Array { dims: vec![], data: vec![100] }, &is, &vs])))
        }
        other => Err(format!("no timing input recipe for {other}")),
    }
}

/// Pathfinder's published algorithm (`rodinia/pathfinder/pathfinder.fut`) as a gene: the state starts as row 0 of
/// the walls; for each later row `i` (1 .. rows-1), each column `j` becomes the wall plus the least of the state's
/// `j`, `j - 1` and `j + 1`, the published order of the two `min`s. The published kernel skips a neighbour that is off
/// the edge; here the index is clamped, so the off-edge neighbour is `j` itself, which the least already holds.
pub const PATHFINDER_REFERENCE: &str = "(loop1.rows (row.rows.cols walls zero)
  (tab (add (get.cols (row.rows.cols walls i1) j2)
            (min (min (get.cols s1 j2) (get.cols s1 (prev j2))) (get.cols s1 (next j2))))))";

pub const PATHFINDER_PLANTED: &str = "(map2.min
  (loop1.rows (row.rows.cols walls zero)
    (tab (add (get.cols (row.rows.cols walls i1) j2)
              (min (min (get.cols s1 j2) (get.cols s1 (prev j2))) (get.cols s1 (next j2))))))
  (loop1.rows (row.rows.cols walls zero)
    (tab (add (get.cols (row.rows.cols walls i1) j2)
              (min (get.cols s1 j2) (min (get.cols s1 (prev j2)) (get.cols s1 (next j2))))))))";

pub const PATHFINDER_PROBLEM: Problem = Problem {
    name: "pathfinder",
    instance: PATHFINDER,
    reference: PATHFINDER_REFERENCE,
    yardstick: include_str!("../tests/corpus/futhark_problems/pathfinder/yardstick.fut"),
    planted: PATHFINDER_PLANTED,
    corpus: "tests/corpus/futhark_problems/pathfinder",
    timing_n: 0,
};

/// Reduce's published program, `i32.sum` (`reduce (+) 0`), as a gene.
pub const REDUCE_REFERENCE: &str = "(sum.n xs)";
/// Reduce's planted slowdown: the sum of the elementwise least of the input and itself (a `map2` Futhark keeps; the
/// e-graph's `map2 min a a = a` removes it).
pub const REDUCE_PLANTED: &str = "(sum.n (map2.min xs xs))";

pub const REDUCE_PROBLEM: Problem = Problem {
    name: "reduce",
    instance: REDUCE,
    reference: REDUCE_REFERENCE,
    yardstick: include_str!("../tests/corpus/futhark_problems/reduce/yardstick.fut"),
    planted: REDUCE_PLANTED,
    corpus: "tests/corpus/futhark_problems/reduce",
    timing_n: 15 << 20,
};

/// Scan's published program, `scan (+) 0`, as a gene.
pub const SCAN_REFERENCE: &str = "(scan.add xs)";
/// Scan's planted slowdown: the scan of the elementwise greatest of the input and itself.
pub const SCAN_PLANTED: &str = "(scan.add (map2.max xs xs))";

pub const SCAN_PROBLEM: Problem = Problem {
    name: "scan",
    instance: SCAN,
    reference: SCAN_REFERENCE,
    yardstick: include_str!("../tests/corpus/futhark_problems/scan/yardstick.fut"),
    planted: SCAN_PLANTED,
    corpus: "tests/corpus/futhark_problems/scan",
    timing_n: 15 << 20,
};

/// The histogram's published program, `hist (+) 0 k (map (% k) is) vs`, as a gene: the bins are the indices modulo
/// `k`, tabulated over `n`, and the values are summed into them.
pub const HISTOGRAM_REFERENCE: &str = "(hist.add.n (tab (mod.k (get.n is j1))) vs)";
/// The histogram's planted slowdown: the values replaced by the elementwise least of the values and themselves.
pub const HISTOGRAM_PLANTED: &str = "(hist.add.n (tab (mod.k (get.n is j1))) (map2.min vs vs))";

pub const HISTOGRAM_PROBLEM: Problem = Problem {
    name: "histogram",
    instance: HISTOGRAM,
    reference: HISTOGRAM_REFERENCE,
    yardstick: include_str!("../tests/corpus/futhark_problems/histogram/yardstick.fut"),
    planted: HISTOGRAM_PLANTED,
    corpus: "tests/corpus/futhark_problems/histogram",
    timing_n: 15 << 20,
};

/// A problem by name.
pub fn problem(name: &str) -> Result<Problem, String> {
    match name {
        "pathfinder" => Ok(PATHFINDER_PROBLEM),
        "reduce" => Ok(REDUCE_PROBLEM),
        "scan" => Ok(SCAN_PROBLEM),
        "histogram" => Ok(HISTOGRAM_PROBLEM),
        other => Err(format!("no Futhark signature problem {other} (pathfinder, reduce, scan, histogram)")),
    }
}

/// Load an instance's tables.
pub fn load(inst: &Instance) -> Result<FutharkKingdom, String> {
    FutharkKingdom::from_tables(FutharkFamily::Shape, &format!("Futhark Sig {}", inst.name), inst.types, inst.functions)
}

// ---- the declared signature -------------------------------------------------------------------------------------

/// A typed array of the signature: its name, element type and dimension roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub elem: String,
    pub dims: Vec<String>,
}

/// The declared signature, read from `interface.tsv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub inputs: Vec<Param>,
    pub output: Param,
    /// The family's binder depth: a node runs at most `max(D) ** depth` times.
    pub depth: u32,
    /// Every dimension role, in order of first appearance.
    pub dims: Vec<String>,
    /// The SIZE INPUTS: dimensions given as an `i32` value parameter of the entry point (a `size` line of the
    /// interface; the parameter is in `inputs`, a scalar), rather than by an array's shape.
    pub size_inputs: Vec<String>,
}

pub fn signature(inst: &Instance) -> Result<Signature, String> {
    let mut inputs = Vec::new();
    let mut size_inputs = Vec::new();
    let (mut output, mut depth, mut dims) = (None, None, None);
    for line in inst.interface.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            return Err(format!("{}: the interface line {line:?} has {} fields, wants 4", inst.name, f.len()));
        }
        let list = |s: &str| s.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>();
        match f[0] {
            "input" => inputs.push(Param { name: f[1].into(), elem: f[2].into(), dims: list(f[3]) }),
            "size" => {
                if f[2] != "i32" || !f[3].is_empty() {
                    return Err(format!("{}: the size input {} is not an i32 scalar", inst.name, f[1]));
                }
                inputs.push(Param { name: f[1].into(), elem: "i32".into(), dims: vec![] });
                size_inputs.push(f[1].to_string());
            }
            "output" => output = Some(Param { name: f[1].into(), elem: f[2].into(), dims: list(f[3]) }),
            "depth" => depth = Some(f[1].parse::<u32>().map_err(|e| format!("{}: depth: {e}", inst.name))?),
            "dims" => dims = Some(list(f[1])),
            other => return Err(format!("{}: an interface line of kind {other}", inst.name)),
        }
    }
    Ok(Signature {
        inputs,
        output: output.ok_or_else(|| format!("{}: the interface has no output", inst.name))?,
        depth: depth.ok_or_else(|| format!("{}: the interface has no depth", inst.name))?,
        dims: dims.ok_or_else(|| format!("{}: the interface has no dims", inst.name))?,
        size_inputs,
    })
}

impl Signature {
    /// The sizes of every dimension role, bound from the inputs (each input's shape must have its declared rank;
    /// a role named twice must have one size; every size is at least 1, the family's precondition).
    pub fn sizes(&self, inputs: &[Array]) -> Result<BTreeMap<String, usize>, String> {
        if inputs.len() != self.inputs.len() {
            return Err(format!("{} inputs, the signature has {}", inputs.len(), self.inputs.len()));
        }
        let mut out = BTreeMap::new();
        for (p, a) in self.inputs.iter().zip(inputs) {
            if self.size_inputs.contains(&p.name) {
                let n = match (a.dims.as_slice(), a.data.as_slice()) {
                    ([], [n]) if *n >= 1 => *n as usize,
                    _ => return Err(format!("the size input {} is {:?} {:?}, not one i32 of at least 1", p.name, a.dims, a.data)),
                };
                if let Some(old) = out.insert(p.name.clone(), n) {
                    if old != n {
                        return Err(format!("the dimension {} is {old} and {n}", p.name));
                    }
                }
                continue;
            }
            if a.dims.len() != p.dims.len() {
                return Err(format!("the input {} has rank {}, declared {}", p.name, a.dims.len(), p.dims.len()));
            }
            for (d, &n) in p.dims.iter().zip(&a.dims) {
                if n == 0 {
                    return Err(format!("the input {} has an empty dimension {d}: every dimension is at least 1", p.name));
                }
                if let Some(old) = out.insert(d.clone(), n) {
                    if old != n {
                        return Err(format!("the dimension {d} is {old} and {n}"));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The dimensions bound by an array's shape (the entry point's size parameters): every one but the size inputs.
    fn shape_dims(&self) -> impl Iterator<Item = &String> {
        self.dims.iter().filter(|d| !self.size_inputs.contains(d))
    }

    /// A parameter's or the output's Futhark type, its dimensions suffixed with `i` (a case of the quality program;
    /// empty for the plain program); a size input's dimension is `i64.i32` of its parameter.
    fn futhark_type(&self, p: &Param, i: &str) -> String {
        let dim = |d: &String| if self.size_inputs.contains(d) { format!("[i64.i32 {d}{i}]") } else { format!("[{d}{i}]") };
        format!("{}{}", p.dims.iter().map(dim).collect::<String>(), p.elem)
    }
}

// ---- values: Futhark's binary format ----------------------------------------------------------------------------

/// An `i32` array of any rank (a scalar has no dims), row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array {
    pub dims: Vec<usize>,
    pub data: Vec<i32>,
}

/// Values in Futhark's binary format (`b`, version 2, rank, ` i32`, the dims as `i64`, the data), one after another:
/// what a Futhark executable reads on stdin for its entry point's parameters, in order.
pub fn write_binary(values: &[&Array]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        out.extend_from_slice(&[b'b', 2, v.dims.len() as u8]);
        out.extend_from_slice(b" i32");
        for &d in &v.dims {
            out.extend_from_slice(&(d as i64).to_le_bytes());
        }
        for x in &v.data {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Every value of a stream in Futhark's binary format (`i32` only; whitespace between values is skipped).
pub fn read_binary(raw: &[u8]) -> Result<Vec<Array>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        while i < raw.len() && raw[i].is_ascii_whitespace() {
            i += 1;
        }
        if i == raw.len() {
            return Ok(out);
        }
        let head = raw.get(i..i + 7).ok_or("a value ends inside its header")?;
        if head[0] != b'b' || head[1] != 2 {
            return Err(format!("not Futhark's binary format: {:?}", &head[..2]));
        }
        if &head[3..7] != b" i32" {
            return Err(format!("a value that is not i32: {:?}", String::from_utf8_lossy(&head[3..7])));
        }
        let rank = head[2] as usize;
        i += 7;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = raw.get(i..i + 8).ok_or("a value ends inside its shape")?;
            let n = i64::from_le_bytes(d.try_into().map_err(|_| "a shape that is not 8 bytes")?);
            dims.push(usize::try_from(n).map_err(|_| format!("a negative dimension {n}"))?);
            i += 8;
        }
        let n: usize = dims.iter().product();
        let data = raw.get(i..i + 4 * n).ok_or("a value ends inside its data")?;
        out.push(Array { dims, data: data.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() });
        i += 4 * n;
    }
}

/// The one value of a program's output.
pub fn read_one(raw: &[u8]) -> Result<Array, String> {
    let mut v = read_binary(raw)?;
    if v.len() != 1 {
        return Err(format!("the program wrote {} values, wants 1", v.len()));
    }
    Ok(v.remove(0))
}

// ---- the program ------------------------------------------------------------------------------------------------

/// The whole program: the entry point with the declared signature (the dimension roles as size parameters, a size
/// input as an `i32` parameter rebound as the size `i64.i32` of it), the zero array of every rank-1 class bound once,
/// around the rendered tree.
pub fn program(k: &FutharkKingdom, sig: &Signature, tree: &Tree) -> Result<String, String> {
    program_named(k, sig, tree, "main")
}

/// [`program`] with its entry point named `entry` (a member of a batch program, `crate::futhark_batch`): the text is
/// [`program`]'s but for the name.
pub fn program_named(k: &FutharkKingdom, sig: &Signature, tree: &Tree, entry: &str) -> Result<String, String> {
    let root = k.rows.get(tree.symbol as usize).ok_or("the root is not a row")?;
    if !k.dual(root.output).is_some_and(|d| d.root) {
        return Err(format!("the root {} is not the declared output", root.name));
    }
    let mut s = format!("entry {entry} {} ", sig.shape_dims().map(|d| format!("[{d}]")).collect::<String>());
    for p in &sig.inputs {
        s.push_str(&format!("({}: {}) ", p.name, sig.futhark_type(p, "")));
    }
    s.push_str(&format!(": {} =\n", sig.futhark_type(&sig.output, "")));
    for d in &sig.size_inputs {
        s.push_str(&format!("  let {d} = i64.i32 {d}\n"));
    }
    for d in &sig.dims {
        s.push_str(&format!("  let zeros_{d} = replicate {d} 0{}\n", sig.output.elem));
    }
    s.push_str(&format!("  in {}\n", k.render(tree)?));
    Ok(s)
}

/// An entry point `entry` with the declared signature that returns `callee` applied to its parameters, in order (its
/// output's sizes left to the callee): how a program given as text, wrapped in a module so its own definitions cannot
/// clash with another copy's, becomes a member of a batch (`crate::futhark_batch`).
pub fn entry_calling(sig: &Signature, entry: &str, callee: &str) -> String {
    let mut s = format!("entry {entry} {}", sig.shape_dims().map(|d| format!("[{d}]")).collect::<String>());
    for p in &sig.inputs {
        s.push_str(&format!(" ({}: {})", p.name, sig.futhark_type(p, "")));
    }
    let args: Vec<&str> = sig.inputs.iter().map(|p| p.name.as_str()).collect();
    format!("{s} : {}{} =\n  {callee} {}\n", "[]".repeat(sig.output.dims.len()), sig.output.elem, args.join(" "))
}

/// THE QUALITY PROGRAM: the tree as a function of the declared signature (`#[noinline]`, so it is compiled once), and
/// an entry point that applies it to `n` cases at once, each with its own sizes, returning the `n` outputs. A GPU
/// process costs about 0.25 s to start whatever it runs (measured 2026-10-27, M3 Max), so one process for every case
/// instead of one a case is what makes the evaluation rate. The function's text is [`program`]'s body unchanged;
/// the timed program is [`program`] itself.
pub fn program_cases(k: &FutharkKingdom, sig: &Signature, tree: &Tree, n: usize) -> Result<String, String> {
    program_cases_named(k, sig, tree, n, "kernel", "main")
}

/// [`program_cases`] with its function named `kernel` and its entry point `entry` (a member of a batch program,
/// `crate::futhark_batch`): the text is [`program_cases`]'s but for the two names.
pub fn program_cases_named(k: &FutharkKingdom, sig: &Signature, tree: &Tree, n: usize, kernel: &str, entry: &str) -> Result<String, String> {
    if n == 0 {
        return Err("a quality program of no cases".into());
    }
    let single = program(k, sig, tree)?;
    let body = single.split_once("=\n").map(|(_, b)| b).ok_or("our software has a bug: the program has no body")?;
    let ty = |p: &Param, i: usize| sig.futhark_type(p, &i.to_string());
    let mut s = format!("#[noinline]\ndef {kernel} ");
    s.push_str(&sig.shape_dims().map(|d| format!("[{d}]")).collect::<String>());
    for p in &sig.inputs {
        s.push_str(&format!(" ({}: {})", p.name, sig.futhark_type(p, "")));
    }
    s.push_str(&format!(" : {} =\n{body}\nentry {entry} ", sig.futhark_type(&sig.output, "")));
    for i in 0..n {
        s.push_str(&sig.shape_dims().map(|d| format!("[{d}{i}]")).collect::<String>());
    }
    for i in 0..n {
        for p in &sig.inputs {
            s.push_str(&format!(" ({}{i}: {})", p.name, ty(p, i)));
        }
    }
    let outs: Vec<String> = (0..n).map(|i| ty(&sig.output, i)).collect();
    let calls: Vec<String> = (0..n).map(|i| format!("{kernel} {}", sig.inputs.iter().map(|p| format!("{}{i}", p.name)).collect::<Vec<_>>().join(" "))).collect();
    s.push_str(&format!(" : ({}) =\n  ({})\n", outs.join(", "), calls.join(", ")));
    Ok(s)
}

// ---- the host interpreter ---------------------------------------------------------------------------------------

/// What a row does, parsed once from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Min,
    Max,
    Neg,
    Abs,
    Sel,
    Get,
    Sum,
    Minimum,
    Maximum,
    /// An index as a value.
    Ival,
    /// The size of a dimension as a value.
    DimVal(usize),
    Const,
    /// The state of frame k.
    State(usize),
    /// The index of frame k (a tabulate's `jk`, a loop's `ik`).
    Index(usize),
    Prev,
    Next(usize),
    Clamp(usize),
    Zero,
    Last(usize),
    /// A tabulate over a dimension, binding frame k.
    Tab(usize, usize),
    Replicate(usize),
    /// An elementwise operation of two arrays of one class (`map2`): add, sub, min, max.
    Map2(Bin),
    /// The inclusive scan of an array (the `scan` family): add, min, max.
    Scan(Bin),
    /// The histogram into a dimension's array of values at indices (the `hist` family): add, min, max.
    Hist(Bin, usize),
    /// A value modulo a dimension's size (the `mod` family), Futhark's `%`: in `[0, D)`.
    Mod(usize),
    /// The row of an array.
    Row,
    Input(usize),
    Zeros(usize),
    /// A loop over a dimension binding frame k; `from1` runs its counter over 1 .. D-1.
    Loop { dim: usize, k: usize, from1: bool },
}

/// The elementwise operations of `map2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bin {
    Add,
    Sub,
    Min,
    Max,
}

impl Bin {
    /// The neutral element Futhark's `scan` and `hist` start from (`sub` has none and is no scan or histogram).
    fn neutral(self) -> i32 {
        match self {
            Bin::Add | Bin::Sub => 0,
            Bin::Min => i32::MAX,
            Bin::Max => i32::MIN,
        }
    }

    fn of(name: &str) -> Option<Bin> {
        match name {
            "add" => Some(Bin::Add),
            "sub" => Some(Bin::Sub),
            "min" => Some(Bin::Min),
            "max" => Some(Bin::Max),
            _ => None,
        }
    }

    fn apply(self, a: i32, b: i32) -> i32 {
        match self {
            Bin::Add => a.wrapping_add(b),
            Bin::Sub => a.wrapping_sub(b),
            Bin::Min => a.min(b),
            Bin::Max => a.max(b),
        }
    }
}

/// The interpreter of one instance: each row's meaning, parsed once.
pub struct Interp<'a> {
    k: &'a FutharkKingdom,
    sig: &'a Signature,
    ops: Vec<Op>,
    literals: Vec<Option<Vec<i32>>>,
}

/// An array value as a view: shared data, an offset, the dims.
#[derive(Debug, Clone)]
struct View {
    data: Rc<Vec<i32>>,
    off: usize,
    dims: Vec<usize>,
}

impl View {
    fn len(&self) -> usize {
        self.dims.iter().product()
    }
    fn slice(&self) -> &[i32] {
        &self.data[self.off..self.off + self.len()]
    }
}

#[derive(Debug, Clone)]
enum Val {
    S(i32),
    I(i64),
    A(View),
}

/// The frames' variables: a state and an index a frame (frame 0 is unused).
struct Env {
    sizes: Vec<usize>,
    inputs: Vec<View>,
    zeros: Vec<View>,
    state: Vec<Option<Val>>,
    index: Vec<i64>,
}

fn depth_of(dual: &str) -> usize {
    dual.split_once(':').map_or(0, |(_, ctx)| ctx.matches('/').count())
}

impl<'a> Interp<'a> {
    pub fn new(k: &'a FutharkKingdom, sig: &'a Signature) -> Result<Interp<'a>, String> {
        let dim = |d: &str| sig.dims.iter().position(|x| x == d).ok_or_else(|| format!("no dimension {d}"));
        let class_dim = |dual: &str, prefix: &str| -> Result<usize, String> {
            let cls = dual.split(':').next().unwrap_or("");
            dim(cls.strip_prefix(prefix).ok_or_else(|| format!("{dual} is not a {prefix} class"))?)
        };
        let mut ops = Vec::with_capacity(k.rows.len());
        let mut literals = Vec::with_capacity(k.duals.len());
        for d in &k.duals {
            literals.push(if d.literals.is_empty() {
                None
            } else {
                Some(d.literals.iter().map(|l| l.trim_end_matches(sig.output.elem.as_str()).parse::<i32>().map_err(|e| format!("the literal {l}: {e}"))).collect::<Result<Vec<_>, _>>()?)
            });
        }
        for r in &k.rows {
            let out = &k.dual(r.output).ok_or("a row with no dual")?.dual;
            let (op, _) = r.name.split_once('@').unwrap_or((r.name.as_str(), ""));
            let kk = depth_of(out) + 1;
            let num = |s: &str| s.get(1..).and_then(|n| n.parse::<usize>().ok());
            let o = match op {
                "add" => Op::Add,
                "sub" => Op::Sub,
                "mul" => Op::Mul,
                "min" => Op::Min,
                "max" => Op::Max,
                "neg" => Op::Neg,
                "abs" => Op::Abs,
                "sel" => Op::Sel,
                "prev" => Op::Prev,
                "next" => Op::Next(class_dim(out, "ix.")?),
                "clamp" => Op::Clamp(class_dim(out, "ix.")?),
                "zero" => Op::Zero,
                "last" => Op::Last(class_dim(out, "ix.")?),
                "tab" => Op::Tab(class_dim(out, "a.")?, kk),
                "replicate" => Op::Replicate(class_dim(out, "a.")?),
                "zeros" => Op::Zeros(class_dim(out, "a.")?),
                _ if r.kind == FutharkKind::Const => Op::Const,
                _ if op.starts_with("get.") => Op::Get,
                _ if op.starts_with("sum.") => Op::Sum,
                _ if op.starts_with("minimum.") => Op::Minimum,
                _ if op.starts_with("maximum.") => Op::Maximum,
                _ if op.starts_with("ival.") => Op::Ival,
                _ if op.starts_with("row.") => Op::Row,
                "map2.add" => Op::Map2(Bin::Add),
                "map2.sub" => Op::Map2(Bin::Sub),
                "map2.min" => Op::Map2(Bin::Min),
                "map2.max" => Op::Map2(Bin::Max),
                _ if op.starts_with("scan.") => Op::Scan(Bin::of(&op[5..]).filter(|b| *b != Bin::Sub).ok_or_else(|| format!("our software has a bug: the scan row {}", r.name))?),
                _ if op.starts_with("hist.") => {
                    let b = op.split('.').nth(1).and_then(Bin::of).filter(|b| *b != Bin::Sub).ok_or_else(|| format!("our software has a bug: the histogram row {}", r.name))?;
                    Op::Hist(b, class_dim(out, "a.")?)
                }
                _ if op.starts_with("mod.") => Op::Mod(dim(&op[4..])?),
                // A size input's row is its dimension as a value, not an input array.
                _ if sig.size_inputs.iter().any(|d| d == op) => Op::DimVal(dim(op)?),
                _ if op.starts_with("loop1.") => Op::Loop { dim: dim(&op[6..])?, k: kk, from1: true },
                _ if op.starts_with("loop.") => Op::Loop { dim: dim(&op[5..])?, k: kk, from1: false },
                _ if op.starts_with('s') && num(op).is_some() => Op::State(num(op).unwrap_or(0)),
                _ if (op.starts_with('i') || op.starts_with('j')) && num(op).is_some() => Op::Index(num(op).unwrap_or(0)),
                _ => {
                    if let Some(i) = sig.inputs.iter().position(|p| p.name == op) {
                        Op::Input(i)
                    } else if let Ok(d) = dim(op) {
                        Op::DimVal(d)
                    } else {
                        return Err(format!("our software has a bug: the row {} has no meaning in the interpreter", r.name));
                    }
                }
            };
            ops.push(o);
        }
        Ok(Interp { k, sig, ops, literals })
    }

    /// The value of an expressed tree on the inputs: the program's output, as Futhark computes it.
    pub fn eval(&self, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
        let sizes_map = self.sig.sizes(inputs)?;
        let sizes: Vec<usize> = self.sig.dims.iter().map(|d| sizes_map[d]).collect();
        let depth = self.sig.depth as usize + 2;
        let mut env = Env {
            inputs: inputs.iter().map(|a| View { data: Rc::new(a.data.clone()), off: 0, dims: a.dims.clone() }).collect(),
            zeros: sizes.iter().map(|&n| View { data: Rc::new(vec![0; n]), off: 0, dims: vec![n] }).collect(),
            sizes,
            state: vec![None; depth],
            index: vec![0; depth],
        };
        match self.at(tree, &mut env)? {
            Val::A(v) => Ok(Array { dims: v.dims.clone(), data: v.slice().to_vec() }),
            Val::S(x) => Ok(Array { dims: vec![], data: vec![x] }),
            Val::I(_) => Err("our software has a bug: a program whose value is an index".into()),
        }
    }

    fn s(&self, t: &Tree, env: &mut Env) -> Result<i32, String> {
        match self.at(t, env)? {
            Val::S(x) => Ok(x),
            other => Err(format!("our software has a bug: a scalar slot holds {other:?}")),
        }
    }
    fn i(&self, t: &Tree, env: &mut Env) -> Result<i64, String> {
        match self.at(t, env)? {
            Val::I(x) => Ok(x),
            other => Err(format!("our software has a bug: an index slot holds {other:?}")),
        }
    }
    fn a(&self, t: &Tree, env: &mut Env) -> Result<View, String> {
        match self.at(t, env)? {
            Val::A(x) => Ok(x),
            other => Err(format!("our software has a bug: an array slot holds {other:?}")),
        }
    }

    fn at(&self, t: &Tree, env: &mut Env) -> Result<Val, String> {
        let op = *self.ops.get(t.symbol as usize).ok_or("a symbol that is not a row")?;
        let kid = |i: usize| t.kids.get(i).ok_or_else(|| format!("{} lacks argument {i}", self.k.rows[t.symbol as usize].name));
        Ok(match op {
            Op::Add => Val::S(self.s(kid(0)?, env)?.wrapping_add(self.s(kid(1)?, env)?)),
            Op::Sub => Val::S(self.s(kid(0)?, env)?.wrapping_sub(self.s(kid(1)?, env)?)),
            Op::Mul => Val::S(self.s(kid(0)?, env)?.wrapping_mul(self.s(kid(1)?, env)?)),
            Op::Min => Val::S(self.s(kid(0)?, env)?.min(self.s(kid(1)?, env)?)),
            Op::Max => Val::S(self.s(kid(0)?, env)?.max(self.s(kid(1)?, env)?)),
            Op::Neg => Val::S(self.s(kid(0)?, env)?.wrapping_neg()),
            Op::Abs => Val::S(self.s(kid(0)?, env)?.wrapping_abs()),
            Op::Sel => {
                // Futhark evaluates only the branch taken.
                let c = self.s(kid(0)?, env)?;
                Val::S(if c < 0 { self.s(kid(1)?, env)? } else { self.s(kid(2)?, env)? })
            }
            Op::Get => {
                let a = self.a(kid(0)?, env)?;
                let i = self.i(kid(1)?, env)?;
                let n = a.len();
                if a.dims.len() != 1 || i < 0 || i as usize >= n {
                    return Err(format!("our software has a bug: the index {i} is out of range for {:?}", a.dims));
                }
                Val::S(a.data[a.off + i as usize])
            }
            Op::Sum => Val::S(self.a(kid(0)?, env)?.slice().iter().fold(0i32, |s, x| s.wrapping_add(*x))),
            Op::Minimum => Val::S(self.a(kid(0)?, env)?.slice().iter().fold(i32::MAX, |s, x| s.min(*x))),
            Op::Maximum => Val::S(self.a(kid(0)?, env)?.slice().iter().fold(i32::MIN, |s, x| s.max(*x))),
            Op::Ival => Val::S(self.i(kid(0)?, env)? as i32),
            Op::DimVal(d) => Val::S(env.sizes[d] as i32),
            Op::Const => {
                let row = &self.k.rows[t.symbol as usize];
                let ty = match row.output {
                    Ty::FutharkShape(i) => i as usize,
                    _ => return Err("a constant outside the family".into()),
                };
                let pool = self.literals[ty].as_ref().ok_or("a constant of a type with no pool")?;
                Val::S(*pool.get(t.const_index.ok_or("a constant with no index")? as usize).ok_or("a constant index past the pool")?)
            }
            Op::State(k) => env.state.get(k).cloned().flatten().ok_or_else(|| format!("our software has a bug: s{k} is read unbound"))?,
            Op::Index(k) => Val::I(*env.index.get(k).ok_or("an index frame past the depth")?),
            Op::Prev => Val::I((self.i(kid(0)?, env)? - 1).max(0)),
            Op::Next(d) => Val::I((self.i(kid(0)?, env)? + 1).min(env.sizes[d] as i64 - 1)),
            Op::Clamp(d) => Val::I(i64::from(self.s(kid(0)?, env)?).min(env.sizes[d] as i64 - 1).max(0)),
            Op::Zero => Val::I(0),
            Op::Last(d) => Val::I(env.sizes[d] as i64 - 1),
            Op::Tab(d, k) => {
                let n = env.sizes[d];
                let mut out = Vec::with_capacity(n);
                for j in 0..n {
                    env.index[k] = j as i64;
                    out.push(self.s(kid(0)?, env)?);
                }
                Val::A(View { data: Rc::new(out), off: 0, dims: vec![n] })
            }
            Op::Replicate(d) => {
                let n = env.sizes[d];
                let x = self.s(kid(0)?, env)?;
                Val::A(View { data: Rc::new(vec![x; n]), off: 0, dims: vec![n] })
            }
            Op::Map2(b) => {
                let x = self.a(kid(0)?, env)?;
                let y = self.a(kid(1)?, env)?;
                if x.dims != y.dims || x.dims.len() != 1 {
                    return Err(format!("our software has a bug: map2 over {:?} and {:?}", x.dims, y.dims));
                }
                let out: Vec<i32> = x.slice().iter().zip(y.slice()).map(|(p, q)| b.apply(*p, *q)).collect();
                Val::A(View { data: Rc::new(out), off: 0, dims: x.dims.clone() })
            }
            Op::Scan(b) => {
                let x = self.a(kid(0)?, env)?;
                let out: Vec<i32> = x
                    .slice()
                    .iter()
                    .scan(b.neutral(), |acc, v| {
                        *acc = b.apply(*acc, *v);
                        Some(*acc)
                    })
                    .collect();
                Val::A(View { data: Rc::new(out), off: 0, dims: x.dims.clone() })
            }
            Op::Hist(b, d) => {
                let is = self.a(kid(0)?, env)?;
                let vs = self.a(kid(1)?, env)?;
                if is.dims != vs.dims || is.dims.len() != 1 {
                    return Err(format!("our software has a bug: a histogram of {:?} at {:?}", vs.dims, is.dims));
                }
                let n = env.sizes[d];
                let mut out = vec![b.neutral(); n];
                for (i, v) in is.slice().iter().zip(vs.slice()) {
                    // Futhark's hist ignores an index outside [0, n).
                    if let Ok(i) = usize::try_from(*i) {
                        if i < n {
                            out[i] = b.apply(out[i], *v);
                        }
                    }
                }
                Val::A(View { data: Rc::new(out), off: 0, dims: vec![n] })
            }
            Op::Mod(d) => Val::S(self.s(kid(0)?, env)?.rem_euclid(env.sizes[d] as i32)),
            Op::Row => {
                let m = self.a(kid(0)?, env)?;
                let i = self.i(kid(1)?, env)?;
                if m.dims.len() < 2 || i < 0 || i as usize >= m.dims[0] {
                    return Err(format!("our software has a bug: the row {i} is out of range for {:?}", m.dims));
                }
                let dims = m.dims[1..].to_vec();
                let stride: usize = dims.iter().product();
                Val::A(View { data: m.data.clone(), off: m.off + i as usize * stride, dims })
            }
            Op::Input(i) => Val::A(env.inputs[i].clone()),
            Op::Zeros(d) => Val::A(env.zeros[d].clone()),
            Op::Loop { dim, k, from1 } => {
                let mut state = self.at(kid(0)?, env)?;
                let n = env.sizes[dim];
                let trips = if from1 { n.saturating_sub(1) } else { n };
                for t in 0..trips {
                    env.index[k] = (t + usize::from(from1)) as i64;
                    env.state[k] = Some(state);
                    state = self.at(kid(1)?, env)?;
                }
                env.state[k] = None;
                state
            }
        })
    }

    /// The node executions of a tree on given sizes: every node once, times the trips of the binders it is inside
    /// (a binder's body is its last argument); a reduction, a `map2` and a `replicate` also cost their array's length.
    pub fn work(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<u128, String> {
        let n: Vec<u128> = self.sig.dims.iter().map(|d| sizes.get(d).map(|&x| x as u128).ok_or_else(|| format!("no size for {d}"))).collect::<Result<_, _>>()?;
        fn walk(me: &Interp<'_>, t: &Tree, mult: u128, n: &[u128]) -> Result<u128, String> {
            let op = me.ops[t.symbol as usize];
            let (body_mult, own) = match op {
                Op::Tab(d, _) => (n[d], 1),
                Op::Loop { dim, from1, .. } => (n[dim].saturating_sub(u128::from(from1)), 1),
                Op::Replicate(d) | Op::Hist(_, d) => (1, 1 + n[d]),
                _ => (1, 1),
            };
            let mut sum = mult * own;
            if let (Op::Sum | Op::Minimum | Op::Maximum | Op::Map2(_) | Op::Scan(_) | Op::Hist(..), Some(kid)) = (op, t.kids.first()) {
                // The array reduced has the length of its class's dimension.
                let dual = &me.k.dual(me.k.rows[kid.symbol as usize].output).ok_or("a row with no dual")?.dual;
                let cls = dual.split(':').next().unwrap_or("");
                let d = me.sig.dims.iter().position(|x| Some(x.as_str()) == cls.strip_prefix("a.")).ok_or("a reduction of a non-array")?;
                sum += mult * n[d];
            }
            let last = t.kids.len().saturating_sub(1);
            for (i, kid) in t.kids.iter().enumerate() {
                let binder = matches!(op, Op::Tab(..) | Op::Loop { .. }) && i == last;
                sum += walk(me, kid, if binder { mult * body_mult } else { mult }, n)?;
            }
            Ok(sum)
        }
        walk(self, tree, 1, &n)
    }

    /// The bound on any tree of a layout: its most nodes, each run at most `max(D) ** depth` times and each costing at
    /// most `1 + max(D)` (a reduction, a `map2` or a replicate), on the given largest dimension.
    pub fn layout_bound(&self, layout: &GeneLayout, max_dim: usize) -> u128 {
        let d = max_dim as u128;
        layout.max_nodes() as u128 * d.pow(self.sig.depth) * (1 + d)
    }
}

// ---- the builder ------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn parse_sexp(text: &str) -> Result<Sexp, String> {
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let mut pos = 0;
    let out = parse_at(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!("text after the expression: {:?}", &tokens[pos..tokens.len().min(pos + 5)]));
    }
    Ok(out)
}

fn parse_at(tokens: &[&str], pos: &mut usize) -> Result<Sexp, String> {
    let t = *tokens.get(*pos).ok_or("the expression ends early")?;
    *pos += 1;
    match t {
        "(" => {
            let mut items = Vec::new();
            loop {
                match tokens.get(*pos) {
                    None => return Err("an unclosed (".into()),
                    Some(&")") => {
                        *pos += 1;
                        return Ok(Sexp::List(items));
                    }
                    Some(_) => items.push(parse_at(tokens, pos)?),
                }
            }
        }
        ")" => Err("an unopened )".into()),
        _ => Ok(Sexp::Atom(t.to_string())),
    }
}

/// A program written as an s-expression of the family's rows, as a tree of the root type. A head `op` in a slot of
/// dual `T` is the row `op@T` (its arguments' duals are the row's own), an atom is the terminal `atom@T`, and an
/// integer is the literal of `T`'s pool. A construct with no row is an error that names it, never a substitute.
pub fn build(k: &FutharkKingdom, text: &str) -> Result<Tree, String> {
    let root = *k.roots().first().ok_or("the kingdom has no root")?;
    let by_name: HashMap<&str, u32> = k.rows.iter().enumerate().map(|(i, r)| (r.name.as_str(), i as u32)).collect();
    build_at(k, &by_name, &parse_sexp(text)?, root)
}

fn build_at(k: &FutharkKingdom, by_name: &HashMap<&str, u32>, e: &Sexp, ty: Ty) -> Result<Tree, String> {
    let dual = k.dual(ty).ok_or("a slot of no dual")?;
    let row = |op: &str| -> Result<u32, String> { by_name.get(format!("{op}@{}", dual.dual).as_str()).copied().ok_or_else(|| format!("the kingdom has no row {op}@{}", dual.dual)) };
    match e {
        Sexp::Atom(a) => {
            if let Ok(v) = a.parse::<i64>() {
                let i = dual.literals.iter().position(|l| l.trim_end_matches(dual.slot.as_str()).parse::<i64>() == Ok(v)).ok_or_else(|| format!("the literal {a} is not in the pool of {}", dual.dual))?;
                let sym = by_name.get(format!("?_{}", dual.dual).as_str()).copied().ok_or_else(|| format!("{} has no constant", dual.dual))?;
                return Ok(Tree { symbol: sym, const_index: Some(i as u32), kids: vec![] });
            }
            Ok(Tree { symbol: row(a)?, const_index: None, kids: vec![] })
        }
        Sexp::List(items) => {
            let Some(Sexp::Atom(head)) = items.first() else { return Err(format!("a list with no head in {}", dual.dual)) };
            let sym = row(head)?;
            let inputs = k.rows[sym as usize].inputs.clone();
            if inputs.len() != items.len() - 1 {
                return Err(format!("{head}@{} takes {} arguments, given {}", dual.dual, inputs.len(), items.len() - 1));
            }
            let kids = items[1..].iter().zip(inputs).map(|(a, t)| build_at(k, by_name, a, t)).collect::<Result<Vec<_>, _>>()?;
            Ok(Tree { symbol: sym, const_index: None, kids })
        }
    }
}

/// A tree as the builder's s-expression (the inverse of [`build`]): for the hall of fame and the reports.
pub fn sexp(k: &FutharkKingdom, t: &Tree) -> String {
    let row = &k.rows[t.symbol as usize];
    let op = row.name.split_once('@').map_or(row.name.as_str(), |(o, _)| o);
    if row.kind == FutharkKind::Const {
        let lit = k.dual(row.output).and_then(|d| t.const_index.and_then(|i| d.literals.get(i as usize)).map(|l| l.trim_end_matches(d.slot.as_str()).to_string()));
        return lit.unwrap_or_else(|| "?".into());
    }
    if t.kids.is_empty() {
        return op.to_string();
    }
    format!("({op} {})", t.kids.iter().map(|c| sexp(k, c)).collect::<Vec<_>>().join(" "))
}

// ---- random genes -----------------------------------------------------------------------------------------------

/// A random gene of the layout: each head cell a function with probability one half and a terminal otherwise (each
/// uniform over its kind; a cell whose alphabet has one kind draws from it), tail and Dc cells uniform; the root cell
/// is a function of the root type (phylu's `root_function`: a root that is a terminal is a constant program).
pub fn random_gene(contract: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut GeneRng) -> Result<Vec<u32>, String> {
    let mut gene = vec![0u32; layout.width];
    for span in layout.spans() {
        let a = layout.alphabet(contract, span.off, const_len);
        if a.is_empty() {
            return Err(format!("our software has a bug: the span at {} has an empty alphabet", span.off));
        }
        let (f, t): (Vec<u32>, Vec<u32>) = a.iter().partition(|s| !contract.is_terminal(**s));
        let split = span.domain == Domain::Head && !f.is_empty() && !t.is_empty();
        for cell in &mut gene[span.off..span.off + span.len] {
            *cell = if split {
                let pick = if rng.below(2) == 0 { &f } else { &t };
                pick[rng.below(pick.len())]
            } else {
                a[rng.below(a.len())]
            };
        }
    }
    let pos = layout.frame(layout.root).and_then(|f| f.row_pos(0)).ok_or("the root has no first row")?;
    let a = layout.alphabet(contract, pos, const_len);
    let f: Vec<u32> = a.into_iter().filter(|s| !contract.is_terminal(*s)).collect();
    if !f.is_empty() {
        gene[pos] = f[rng.below(f.len())];
    }
    Ok(gene)
}

// ---- the stored cases and the executables -------------------------------------------------------------------------

/// Which part of the fitness a case is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    /// Held out from the training sizes; a fitness input all the same (Andrew's rule: the quality group holds the
    /// held-out cases, its validation column).
    Held,
    /// THE TEST SET: never a fitness input, never seen by selection; an exactness claim is made on it (fault 11 of
    /// `docs/reviews/2026-10-28_futhark_evolve_faults.md`).
    Test,
}

impl Split {
    /// Whether the case is a fitness input (training or held-out).
    pub fn in_fitness(self) -> bool {
        self != Split::Test
    }
}

/// One stored case: the inputs and the published program's output on them.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub split: Split,
    pub inputs: Vec<Array>,
    pub output: Array,
    /// The inputs in Futhark's binary format, as the executable reads them.
    pub stdin: Vec<u8>,
}

/// The case directory of a problem.
pub fn corpus_dir(p: &Problem) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(p.corpus)
}

/// The stored cases (`cases.tsv`: name, split, then `<name>.in` and `<name>.out` in Futhark's binary format).
pub fn load_cases(p: &Problem) -> Result<Vec<Case>, String> {
    let dir = corpus_dir(p);
    let index = std::fs::read_to_string(dir.join("cases.tsv")).map_err(|e| format!("{}: {e}", dir.join("cases.tsv").display()))?;
    let mut out = Vec::new();
    for line in index.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let [name, split, ..] = f.as_slice() else { return Err(format!("the case line {line:?}")) };
        let split = match *split {
            "train" => Split::Train,
            "held" => Split::Held,
            "test" => Split::Test,
            other => return Err(format!("the case {name} has the split {other}")),
        };
        let stdin = std::fs::read(dir.join(format!("{name}.in"))).map_err(|e| format!("{name}.in: {e}"))?;
        let inputs = read_binary(&stdin)?;
        let output = read_one(&std::fs::read(dir.join(format!("{name}.out"))).map_err(|e| format!("{name}.out: {e}"))?)?;
        out.push(Case { name: name.to_string(), split, inputs, output, stdin });
    }
    Ok(out)
}

/// Compile a program with Futhark's sequential C backend (`futhark c`, the released compiler) into `dir`.
pub fn compile_c(source: &str, dir: &Path) -> Result<PathBuf, String> {
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

/// Run a compiled executable (either backend's) on a case's binary input and read its one output value.
pub fn run_exe(exe: &Path, stdin: &[u8]) -> Result<Array, String> {
    let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(exe).arg("-b"), std::time::Duration::from_secs(120), Some(stdin))?;
    if !out.status.success() {
        return Err(format!("{} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
    }
    read_one(&out.stdout)
}

/// What a program's output is against the expected one, each on `[0, 1]`, 0 exact: the share of elements that
/// differ; the mean absolute error over the expected values' mean magnitude, and the largest absolute error over
/// their largest magnitude, each mapped `e -> e / (1 + e)`. An output of another shape differs everywhere (1, 1, 1).
pub fn compare(got: &Array, want: &Array) -> [f64; 3] {
    if got.dims != want.dims {
        return [1.0; 3];
    }
    let n = want.data.len().max(1) as f64;
    let (mut differ, mut sum, mut max) = (0usize, 0f64, 0f64);
    for (g, w) in got.data.iter().zip(&want.data) {
        let d = (i64::from(*g) - i64::from(*w)).unsigned_abs() as f64;
        differ += usize::from(d > 0.0);
        sum += d;
        max = max.max(d);
    }
    let mean_mag = (want.data.iter().map(|w| f64::from(*w).abs()).sum::<f64>() / n).max(1.0);
    let max_mag = want.data.iter().map(|w| f64::from(*w).abs()).fold(1.0, f64::max);
    let squash = |e: f64| e / (1.0 + e);
    [differ as f64 / n, squash(sum / n / mean_mag), squash(max / max_mag)]
}

/// One kind of quality input ([`QUALITY_PARTS`] lists the ones the fitness uses, in order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPart {
    /// One input a case: the share of cells not exactly right (the one exactness input).
    Wrong,
    /// One input a cell: `ln(1 + |d|) / ln(1 + M)` (at most 1), `d` the cell's error and `M` the largest magnitude of
    /// the correct output (at least 1). Graded, no threshold, and an error's size counts relative to the answer's
    /// magnitude, never to its range (fault 3).
    CellLog,
    /// One input a cell: `e / (1 + e)` with `e = |d| / m`, `m` the mean magnitude of the correct output (at least 1).
    CellLevel,
    /// One input a neighbouring pair of cells: 1 if the step between them has the wrong sign (up, down, flat) against
    /// the correct output's, else 0. Structure, not level: it credits a program whose output rises and falls where the
    /// answer does (fault 3). Absent for a one-cell output.
    Step,
    /// Two inputs a case whose correct output has at least two distinct values: `(1 - r) / 2` for Spearman's and for
    /// Pearson's correlation of the output with the correct one (a constant output has no shape: 0.5 each).
    Shape,
    /// One input a case: the share of neighbouring pairs of cells whose step has the wrong sign (up, down, flat)
    /// against the correct output's; 0 for a one-cell output; absent, as the shape inputs are, when the correct output
    /// is constant over two or more cells (it has no order to agree with, and "is the output flat" would only repeat
    /// the exactness share).
    Order,
    /// One input a case: the share of cells off by over 10 % of the correct value's magnitude (at least 1).
    Outside10,
    /// The study's seven inputs a case (2026-10-27): the share of cells not exact; the shares off by over 1 %, 10 %
    /// and 50 % of the correct value's magnitude; the mean and largest error over the correct output's range, each
    /// `e / (1 + e)`; the share of neighbouring pairs in the wrong order (0 for one cell).
    Study,
}

/// THE QUALITY INPUTS' FORMULA: the parts each case contributes, in order (plan section B, chosen 2026-10-29 by the
/// slope check on the GPU's outputs of histogram's new cases; `docs/PLAN_futhark_fitness_fixes.md` records every
/// formula tried). The exactness share, the two shape inputs, the order share and the share off by over 10 %: it
/// passes the check (the median neighbour beats 97.71 % of random genes, Spearman +0.077) and does not reward the
/// all-zeros output (68.8 % of the non-zero wrong random outputs score better than it; under the study's seven
/// inputs 28.8 %, so the dead point scored better than most real-looking wrong outputs). The order input is what
/// makes an output off by one everywhere beat all zeros on a small-valued case. Graded per-cell inputs were measured and lose:
/// histogram's whole-range cases give hundreds of cells of unrelated large numbers that drown the exactness signal.
pub const QUALITY_PARTS: &[QualityPart] = &[QualityPart::Wrong, QualityPart::Shape, QualityPart::Order, QualityPart::Outside10];

/// The names of [`quality_inputs`] against a correct output `want`, in order: their number depends on the case
/// (one input a cell or a step).
pub fn quality_input_names(want: &Array) -> Vec<String> {
    let n = want.data.len();
    let distinct = want.data.iter().collect::<std::collections::BTreeSet<_>>().len();
    let mut out = Vec::new();
    for part in QUALITY_PARTS {
        match part {
            QualityPart::Wrong => out.push("wrong_share".to_string()),
            QualityPart::CellLog => out.extend((0..n).map(|i| format!("cell{i}_log_error"))),
            QualityPart::CellLevel => out.extend((0..n).map(|i| format!("cell{i}_level_error"))),
            QualityPart::Step => out.extend((1..n).map(|i| format!("step{i}_sign_wrong"))),
            QualityPart::Shape if distinct >= 2 => out.extend(["rank_disagreement".to_string(), "linear_disagreement".to_string()]),
            QualityPart::Shape => {}
            QualityPart::Order if n >= 2 && distinct < 2 => {}
            QualityPart::Order => out.push("order_wrong".to_string()),
            QualityPart::Outside10 => out.push("outside_10pct".to_string()),
            QualityPart::Study => out.extend(STUDY_INPUTS.iter().map(|s| s.to_string())),
        }
    }
    out
}

/// The names of [`QualityPart::Study`]'s inputs.
const STUDY_INPUTS: [&str; 7] = ["wrong_share", "outside_1pct", "outside_10pct", "outside_50pct", "mean_abs_over_range", "max_abs_over_range", "order_wrong"];

/// [`QualityPart::Study`]'s inputs, exactly the study's (checked against its ledgers).
fn study_inputs(got: &Array, want: &Array) -> [f64; 7] {
    let n = want.data.len().max(1) as f64;
    let lo = want.data.iter().copied().min().unwrap_or(0);
    let hi = want.data.iter().copied().max().unwrap_or(0);
    let range = (i64::from(hi) - i64::from(lo)) as f64;
    let mag = want.data.iter().map(|w| f64::from(*w).abs()).fold(0.0, f64::max);
    let scale = if range > 0.0 { range } else { mag }.max(1.0);
    let (mut wrong, mut o1, mut o10, mut o50, mut sum, mut max) = (0.0, 0.0, 0.0, 0.0, 0.0, 0f64);
    for (g, w) in got.data.iter().zip(&want.data) {
        let d = (i64::from(*g) - i64::from(*w)).unsigned_abs() as f64;
        let rel = d / f64::from(*w).abs().max(1.0);
        wrong += f64::from(u8::from(d > 0.0));
        o1 += f64::from(u8::from(rel > 0.01));
        o10 += f64::from(u8::from(rel > 0.10));
        o50 += f64::from(u8::from(rel > 0.50));
        sum += d;
        max = max.max(d);
    }
    let squash = |e: f64| e / (1.0 + e);
    let steps = want.data.len().saturating_sub(1);
    let order = if steps == 0 {
        0.0
    } else {
        let sign = |a: i32, b: i32| (i64::from(b) - i64::from(a)).signum();
        (0..steps).filter(|&j| sign(got.data[j], got.data[j + 1]) != sign(want.data[j], want.data[j + 1])).count() as f64 / steps as f64
    };
    [wrong / n, o1 / n, o10 / n, o50 / n, squash(sum / n / scale), squash(max / scale), order]
}

/// THE QUALITY INPUTS of one case, each its own input on `[0, 1]`, 0 exact, in the order of
/// [`quality_input_names`]: the parts of [`QUALITY_PARTS`]. Every cell and every step is its own input; nothing is
/// averaged before HFF but the one exactness share. An output of the wrong shape is 1 everywhere.
pub fn quality_inputs(got: &Array, want: &Array) -> Vec<f64> {
    let names = quality_input_names(want).len();
    if got.dims != want.dims {
        return vec![1.0; names];
    }
    let n = want.data.len();
    let d: Vec<f64> = got.data.iter().zip(&want.data).map(|(g, w)| (i64::from(*g) - i64::from(*w)).unsigned_abs() as f64).collect();
    let big = want.data.iter().map(|w| f64::from(*w).abs()).fold(1.0, f64::max);
    let mean = (want.data.iter().map(|w| f64::from(*w).abs()).sum::<f64>() / n.max(1) as f64).max(1.0);
    let distinct = want.data.iter().collect::<std::collections::BTreeSet<_>>().len();
    let mut out = Vec::with_capacity(names);
    for part in QUALITY_PARTS {
        match part {
            QualityPart::Wrong => out.push(d.iter().filter(|x| **x > 0.0).count() as f64 / n.max(1) as f64),
            QualityPart::CellLog => out.extend(d.iter().map(|x| (x.ln_1p() / big.ln_1p()).min(1.0))),
            QualityPart::CellLevel => out.extend(d.iter().map(|x| (x / mean) / (1.0 + x / mean))),
            QualityPart::Step => {
                let sign = |v: &[i32], j: usize| (i64::from(v[j + 1]) - i64::from(v[j])).signum();
                out.extend((0..n.saturating_sub(1)).map(|j| f64::from(u8::from(sign(&got.data, j) != sign(&want.data, j)))));
            }
            QualityPart::Shape if distinct >= 2 => {
                let g: Vec<f64> = got.data.iter().map(|x| f64::from(*x)).collect();
                let w: Vec<f64> = want.data.iter().map(|x| f64::from(*x)).collect();
                for r in [crate::runner::spearman(&g, &w), crate::runner::pearson(&g, &w)] {
                    out.push(r.map_or(0.5, |r| (1.0 - r) / 2.0));
                }
            }
            QualityPart::Shape => {}
            QualityPart::Order if n >= 2 && distinct < 2 => {}
            QualityPart::Order => out.push(study_inputs(got, want)[6]),
            QualityPart::Outside10 => out.push(got.data.iter().zip(&want.data).filter(|(g, w)| (i64::from(**g) - i64::from(**w)).unsigned_abs() as f64 / f64::from(**w).abs().max(1.0) > 0.10).count() as f64 / n.max(1) as f64),
            QualityPart::Study => out.extend(study_inputs(got, want)),
        }
    }
    out
}

// ---- the slope check's measures (docs/PLAN_futhark_fitness_fixes.md, section A) -----------------------------------

/// THE TREE EDIT DISTANCE between two expressed trees (Zhang and Shasha, 1989): the least number of node relabels,
/// deletions and insertions, each of cost 1, that turn one tree into the other. A node's label is its row and, for a
/// constant, its pool index. The distance of a program to the reference gene's tree is the slope check's distance.
pub fn tree_distance(a: &Tree, b: &Tree) -> usize {
    /// A tree in postorder: each node's label, its leftmost leaf's postorder index, and the keyroots (the highest
    /// node of each leftmost leaf), ascending.
    struct Post {
        labels: Vec<(u32, Option<u32>)>,
        leftmost: Vec<usize>,
        keyroots: Vec<usize>,
    }
    fn walk(t: &Tree, labels: &mut Vec<(u32, Option<u32>)>, leftmost: &mut Vec<usize>) -> usize {
        let mut first = None;
        for k in &t.kids {
            let i = walk(k, labels, leftmost);
            first.get_or_insert(leftmost[i]);
        }
        let me = labels.len();
        labels.push((t.symbol, t.const_index));
        leftmost.push(first.unwrap_or(me));
        me
    }
    fn post(t: &Tree) -> Post {
        let (mut labels, mut leftmost) = (Vec::new(), Vec::new());
        walk(t, &mut labels, &mut leftmost);
        let mut seen = std::collections::HashSet::new();
        let mut keyroots: Vec<usize> = (0..labels.len()).rev().filter(|i| seen.insert(leftmost[*i])).collect();
        keyroots.sort_unstable();
        Post { labels, leftmost, keyroots }
    }
    let (pa, pb) = (post(a), post(b));
    let (na, nb) = (pa.labels.len(), pb.labels.len());
    let mut tree = vec![vec![0usize; nb]; na];
    for &i in &pa.keyroots {
        for &j in &pb.keyroots {
            let (li, lj) = (pa.leftmost[i], pb.leftmost[j]);
            let (m, n) = (i - li + 2, j - lj + 2);
            let mut forest = vec![vec![0usize; n]; m];
            for x in 1..m {
                forest[x][0] = forest[x - 1][0] + 1;
            }
            for y in 1..n {
                forest[0][y] = forest[0][y - 1] + 1;
            }
            for x in 1..m {
                for y in 1..n {
                    let (u, v) = (li + x - 1, lj + y - 1);
                    let edit = (forest[x - 1][y] + 1).min(forest[x][y - 1] + 1);
                    if pa.leftmost[u] == li && pb.leftmost[v] == lj {
                        forest[x][y] = edit.min(forest[x - 1][y - 1] + usize::from(pa.labels[u] != pb.labels[v]));
                        tree[u][v] = forest[x][y];
                    } else {
                        forest[x][y] = edit.min(forest[pa.leftmost[u] - li][pb.leftmost[v] - lj] + tree[u][v]);
                    }
                }
            }
        }
    }
    tree[na - 1][nb - 1]
}

/// FNV-1a over bytes, as 16 hex digits.
fn fnv(parts: &[&[u8]]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        for b in p.iter().chain(&[0xff]) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

/// The identity of a problem's stored cases: every case's name, split, inputs and output.
pub fn cases_fingerprint(cases: &[Case]) -> String {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    for c in cases {
        parts.push(format!("{}\t{:?}", c.name, c.split).into_bytes());
        parts.push(c.stdin.clone());
        parts.push(write_binary(&[&c.output]));
    }
    fnv(&parts.iter().map(Vec::as_slice).collect::<Vec<_>>())
}

/// The identity of the quality inputs' formula: their values, exactly, on fixed outputs against fixed answers (a
/// level shift, a reversal, a constant, a shape mismatch, a near miss), so any change to the formula changes it.
pub fn quality_fingerprint() -> String {
    let want = Array { dims: vec![7], data: vec![3, -40, 900, 7, 7, 0, 123_456] };
    let probes = [
        Array { dims: vec![7], data: want.data.iter().map(|x| x + 5).collect() },
        Array { dims: vec![7], data: want.data.iter().rev().copied().collect() },
        Array { dims: vec![7], data: vec![0; 7] },
        Array { dims: vec![6], data: vec![1; 6] },
        Array { dims: vec![7], data: vec![3, -40, 901, 7, 7, 0, 123_456] },
    ];
    let text: String = probes.iter().flat_map(|g| quality_inputs(g, &want)).map(|x| format!("{x:?},")).collect();
    fnv(&[text.as_bytes()])
}

// ---- the instance's e-graph -------------------------------------------------------------------------------------

/// The row <-> constructor mapping (`egraph.rows`).
#[derive(Debug, Clone)]
pub struct Bridge {
    ctor_of_row: BTreeMap<String, String>,
    row_of_ctor: BTreeMap<String, String>,
}

impl Bridge {
    /// Read an instance's mapping; a repeated row or constructor is an error (the generator's witness refuses one).
    pub fn load(inst: &Instance) -> Result<Bridge, String> {
        let mut ctor_of_row = BTreeMap::new();
        let mut row_of_ctor = BTreeMap::new();
        for line in inst.egraph_rows.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 3 {
                return Err(format!("the mapping line {line:?} is malformed"));
            }
            if ctor_of_row.insert(f[0].to_string(), f[1].to_string()).is_some() || row_of_ctor.insert(f[1].to_string(), f[0].to_string()).is_some() {
                return Err(format!("the mapping line {line:?} repeats a row or a constructor"));
            }
        }
        Ok(Bridge { ctor_of_row, row_of_ctor })
    }
}

/// An expressed tree as an egglog term; a constant placeholder is its constructor applied to the pool index.
pub fn import(tree: &Tree, k: &FutharkKingdom, bridge: &Bridge) -> Result<String, String> {
    let row = k.rows.get(tree.symbol as usize).ok_or_else(|| format!("symbol {} is not a row", tree.symbol))?;
    let ctor = bridge.ctor_of_row.get(&row.name).ok_or_else(|| format!("the row {} has no constructor", row.name))?;
    if row.kind == FutharkKind::Const {
        return Ok(format!("({ctor} {})", tree.const_index.ok_or_else(|| format!("{} has no constant index", row.name))?));
    }
    let kids = tree.kids.iter().map(|t| import(t, k, bridge)).collect::<Result<Vec<_>, _>>()?;
    Ok(if kids.is_empty() { format!("({ctor})") } else { format!("({ctor} {})", kids.join(" ")) })
}

/// An egglog term as a tree of the kingdom's rows; `None` when it is not one.
pub fn to_tree(termdag: &TermDag, id: TermId, k: &FutharkKingdom, bridge: &Bridge, symbols: &HashMap<String, u32>) -> Option<Tree> {
    let Term::App(op, kids) = termdag.get(id) else { return None };
    let symbol = *symbols.get(bridge.row_of_ctor.get(op.as_str())?)?;
    if k.rows[symbol as usize].kind == FutharkKind::Const {
        let [kid] = kids.as_slice() else { return None };
        let Term::Lit(egglog::ast::Literal::Int(i)) = termdag.get(*kid) else { return None };
        return Some(Tree { symbol, const_index: Some(u32::try_from(*i).ok()?), kids: vec![] });
    }
    let kids = kids.iter().map(|t| to_tree(termdag, *t, k, bridge, symbols)).collect::<Option<Vec<_>>>()?;
    Some(Tree { symbol, const_index: None, kids })
}

fn term_cost(termdag: &TermDag, id: TermId) -> u64 {
    match termdag.get(id) {
        Term::App(_, kids) => 1 + kids.iter().map(|k| term_cost(termdag, *k)).sum::<u64>(),
        _ => 1,
    }
}

/// The e-graph variants of a tree: `iters` iterations of the instance's ruleset, the ranked members of the root's
/// class (HFF extraction, `members` a class), up to `n` distinct trees other than the tree itself, smallest first and
/// then by their rendered text, so the order is deterministic. Every rule is an identity of wrapping `i32`
/// arithmetic and of `min`/`max`, so every variant computes the same values.
pub fn variants(inst: &Instance, k: &FutharkKingdom, bridge: &Bridge, tree: &Tree, iters: u32, members: usize, n: usize) -> Result<Vec<Tree>, String> {
    let term = import(tree, k, bridge)?;
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, inst.datatype).map_err(|e| format!("the datatype does not load: {e}"))?;
    egraph.parse_and_run_program(None, inst.rules).map_err(|e| format!("the ruleset does not load: {e}"))?;
    egraph.parse_and_run_program(None, &format!("(let __root {term})\n(run-schedule (repeat {iters} (run {RULESET})))")).map_err(|e| format!("the term does not load: {e}"))?;
    let (sort, value) = egraph.eval_expr(&exprs::var("__root")).map_err(|e| format!("the root does not evaluate: {e}"))?;
    let mut termdag = TermDag::default();
    let score = |td: &TermDag, t: TermId| term_cost(td, t) as f64;
    let ranked = hff_extract(&egraph, &mut termdag, value, sort, &score, members);
    let symbols: HashMap<String, u32> = k.rows.iter().enumerate().map(|(i, r)| (r.name.clone(), i as u32)).collect();
    let mut out: Vec<(u64, String, Tree)> = Vec::new();
    for (cost, id) in ranked {
        let Some(t) = to_tree(&termdag, id, k, bridge, &symbols) else { continue };
        if &t == tree {
            continue;
        }
        let text = k.render(&t)?;
        if out.iter().any(|(_, s, _)| s == &text) {
            continue;
        }
        out.push((cost as u64, text, t));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    out.truncate(n);
    Ok(out.into_iter().map(|(_, _, t)| t).collect())
}

/// The instance's e-graph as the saturated fold needs it.
pub struct FoldEgraph<'a> {
    pub inst: &'a Instance,
    pub kingdom: &'a FutharkKingdom,
    pub bridge: &'a Bridge,
}

impl EgraphKingdom for FoldEgraph<'_> {
    fn programs(&self) -> Vec<&str> {
        vec![self.inst.datatype, self.inst.rules]
    }
    fn rulesets(&self) -> Vec<&str> {
        vec![RULESET]
    }
    fn import(&self, tree: &Tree) -> Result<String, String> {
        import(tree, self.kingdom, self.bridge)
    }
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)> {
        let name = self.bridge.row_of_ctor.get(constructor)?;
        let symbol = self.kingdom.rows.iter().position(|r| &r.name == name)? as SymbolId;
        if self.kingdom.rows[symbol as usize].kind == FutharkKind::Const {
            let [text] = literals else { return None };
            return Some((symbol, Some(text.parse().ok()?)));
        }
        literals.is_empty().then_some((symbol, None))
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_micro;
