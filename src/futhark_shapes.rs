//! The shape-classed Futhark kingdom: the interface, the shape-polymorphic
//! program, and the checks that the class table and the Futhark compiler
//! agree.
//!
//! The 1-D kingdom decides that shape is a *value*, not a type: `arr.f32` is
//! one type however long the array is. This kingdom decides that a shape
//! **class** is a type: `mat.n.m` is the array the interface calls `[n][m]`,
//! where `n` and `m` are dimension *roles*. The sizes stay symbolic -- the
//! generated source is shape-polymorphic -- while the *relations* between the
//! dimensions become part of the type system.
//!
//! The two kingdoms coexist, and they are separate kingdoms on purpose: the
//! 1-D one keeps the programs whose shapes are only equal at run time, which
//! static classes cannot express.
//!
//! The unification happens in the generator
//! (`kingdoms/futhark_shapes/gen_tables.py`), not here. This module only
//! reads the tables, renders a program, and checks that the result is legal
//! and that Futhark accepts it.

use std::collections::BTreeMap;

use crate::futhark_kingdom::{FutharkDual, FutharkKingdom, FutharkKind};
use crate::geneframe::Ty;
use crate::typed_chromosome::{AdfSet, Epoch};
use crate::typed_gep::Tree;

/// The helper the `matmul` row needs.
///
/// Futhark 0.28.1 has no `matmul` in its prelude, and its `map` cannot map
/// over a matrix's rows with a *scalar* function -- so the kingdom's `matmul`
/// row needs a helper of its own. The helper is **shape-polymorphic**, which
/// is the point: Futhark's own type checker then enforces the
/// inner-dimension relation the class table states, so a table and a compiler
/// that disagree cannot both be right.
pub const MATMUL_HELPER: &str = "def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =\n  map (\\a -> map (\\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A\n";

/// The dimension roles of the kingdom's interface, in a deterministic order.
///
/// A role is a dimension variable in the generated source, so `mat.n.m` and
/// `mat.m.p` share the `m`.
pub fn roles(k: &FutharkKingdom) -> Vec<String> {
    let mut set = std::collections::BTreeSet::new();
    for d in &k.duals {
        if d.slot == "mat" {
            for r in d.form.split('.') {
                set.insert(r.to_string());
            }
        }
    }
    set.into_iter().collect()
}

/// The dimensions of a matrix class, from its dual's `form` (`n.m`).
fn dims_of(d: &FutharkDual) -> Vec<String> {
    d.form.split('.').map(|s| s.to_string()).collect()
}

/// The interface's inputs: each matrix class's terminal, which is the Futhark
/// variable that class's input binds.
fn inputs(k: &FutharkKingdom) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for d in &k.duals {
        if d.slot != "mat" {
            continue;
        }
        let ty = k.ty(&d.dual).expect("a dual's own type");
        let term = k
            .rows
            .iter()
            .find(|r| r.kind == FutharkKind::Terminal && r.output == ty)
            .expect("every matrix class has its input terminal");
        out.push((term.spelling.clone(), dims_of(d)));
    }
    out
}

/// The whole Futhark source for a program: the helper, the shape-polymorphic
/// kernel, and the entry point with concrete sizes.
///
/// The kernel is polymorphic in the interface's roles, so the same gene
/// describes a program for *any* sizes the interface allows; the entry point
/// picks one set.
pub fn program(
    k: &FutharkKingdom,
    root: Ty,
    tree: &Tree,
    sizes: &BTreeMap<String, usize>,
) -> Result<String, String> {
    program_text(k, root, "", &k.render(tree)?, sizes)
}

/// The whole Futhark source for a FOLDED program (a tree of the epoch
/// alphabet): the helper, each ADF the tree calls as a top-level
/// `def adf_<id>` in level order, the kernel calling them, and the entry
/// point.
pub fn program_folded(
    k: &FutharkKingdom,
    root: Ty,
    epoch: &Epoch,
    adfs: &AdfSet,
    tree: &Tree,
    sizes: &BTreeMap<String, usize>,
) -> Result<String, String> {
    let f = k.adf_functions_for(epoch, adfs, tree)?;
    let expr = k.render_folded(epoch, &f, tree)?;
    program_text(k, root, &f.definitions, &expr, sizes)
}

fn program_text(
    k: &FutharkKingdom,
    root: Ty,
    definitions: &str,
    expr: &str,
    sizes: &BTreeMap<String, usize>,
) -> Result<String, String> {
    let root_dims = dims_of(k.dual(root).ok_or_else(|| "the root has no dual".to_string())?);
    let ins = inputs(k);
    // A role with no size is a mistake, not a default: the sizes are what
    // the entry point binds, and a silent 1 would make the program mean
    // something other than what the caller asked for.
    let size_of = |r: &String| -> Result<String, String> {
        sizes
            .get(r)
            .map(|s| s.to_string())
            .ok_or_else(|| format!("the role {r} has no size"))
    };

    let params = ins
        .iter()
        .map(|(v, d)| format!("({v}: [{}]f32)", d.join("][")))
        .collect::<Vec<_>>()
        .join(" ");
    let mut entry_params = Vec::new();
    for (v, d) in &ins {
        let dims: Vec<String> = d.iter().map(&size_of).collect::<Result<Vec<_>, String>>()?;
        entry_params.push(format!("({v}: [{}]f32)", dims.join("][")));
    }
    let entry_params = entry_params.join(" ");
    let dims = format!("[{}]", roles(k).join("]["));
    let result = format!("[{}]f32", root_dims.join("]["));
    let entry_result = format!("[{}]f32", root_dims.iter().map(&size_of).collect::<Result<Vec<_>, _>>()?.join("]["));
    let args = ins.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>().join(" ");

    Ok(format!(
        "{MATMUL_HELPER}\n{definitions}-- the kernel: shape-polymorphic over the interface's roles\n\
         def kernel {dims} {params} : {result} =\n  {expr}\n\n\
         -- the entry point: concrete sizes\n\
         entry main {entry_params} : {entry_result} = kernel {args}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_exec::{compile_program, run_text_program, Verdict};
    use crate::futhark_shapes_egraph::Bridge;
    use crate::futhark_shapes_interp::{equal, format_matrix, identical, parse_matrix, Mat, MValue};
    use crate::typed_gep::{express, GeneLayout};
    use std::process::Command;
    use std::time::Duration;

    fn kingdom() -> FutharkKingdom {
        FutharkKingdom::load_shapes().expect("the shape tables load")
    }

    /// The sizes the harness picks for the roles. The source is polymorphic in
    /// them; these are only the sizes one run uses.
    fn sizes() -> BTreeMap<String, usize> {
        [("m".to_string(), 3usize), ("n".to_string(), 2), ("p".to_string(), 4)].into_iter().collect()
    }

    #[test]
    fn the_shape_kingdom_passes_the_contract_check() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        assert_eq!(k.duals.len(), 11, "nine matrix classes and two function classes");
        assert_eq!(k.rows.len(), 71, "54 operators, 11 terminals, 6 elementary row functions");
        assert_eq!(k.roots().len(), 1, "the interface's result class");
        // Every demanded class has a terminal, which the contract check
        // enforces; and every class is in the SHAPE family, so the two
        // kingdoms' duals cannot be confused.
        for ty in contract.types() {
            assert!(k.family.owns(*ty), "{ty:?} is not in the kingdom's family");
        }
    }

    /// The design's central numeric claim, checked by the engine itself: the
    /// class set's frames and the gene's width.
    #[test]
    fn the_engine_derives_the_counts_the_design_predicted() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let heads = contract.heads_from_program_size(root, 3).expect("heads");
        let bounds = contract.bounds(root, &heads).expect("bounds");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        assert_eq!(bounds.types.len(), 11, "eleven root-reachable frames");
        assert_eq!(layout.width, 322, "the gene's width at program size 3");
        assert_eq!(bounds.l, 75, "the leaf bound");
    }

    /// Legality by construction: every random gene expresses, with no repair
    /// and no rejection. The shape classes make an ill-shaped program
    /// UNREPRESENTABLE, so there is nothing to check after the fact.
    #[test]
    fn every_random_gene_expresses_a_legal_program() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let mut rng: u64 = 0x5EED_1234_ABCD_0002;
        let mut expressed = 0usize;
        for _ in 0..500 {
            let mut gene = vec![0u32; layout.width];
            for (pos, cell) in gene.iter_mut().enumerate() {
                let alphabet = layout.alphabet(&contract, pos, &const_len);
                assert!(!alphabet.is_empty(), "cell {pos} has an empty alphabet");
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
            }
            let e = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal");
            let tree = e.tree();
            // The rendered program must mention every argument of every node:
            // the renderer checks that, so a silent drop is impossible.
            k.render(&tree).expect("the program renders");
            expressed += 1;
        }
        assert_eq!(expressed, 500);
    }

    /// The decisive check: Futhark accepts the rendered programs. The helper
    /// is shape-polymorphic, so a class table that disagreed with Futhark
    /// about the inner dimensions could not compile.
    #[test]
    fn futhark_accepts_the_rendered_programs() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let dir = std::env::temp_dir().join("futhark_shapes_test").join(format!("t{}", std::process::id()));
        let mut rng: u64 = 0x5EED_1234_ABCD_0003;
        let mut compiled = 0usize;
        for _ in 0..3 {
            let mut gene = vec![0u32; layout.width];
            for (pos, cell) in gene.iter_mut().enumerate() {
                let alphabet = layout.alphabet(&contract, pos, &const_len);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
            }
            let e = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal");
            let source = program(&k, root, &e.tree(), &sizes()).expect("the program renders");
            let verdict = compile_program(&source, &dir, Duration::from_secs(120)).0;
            match verdict {
                Verdict::Compiled | Verdict::BoundedFusion => compiled += 1,
                Verdict::Refused(e) => panic!("futhark refused a legal program: {e}\n{source}"),
                Verdict::TimedOut => panic!("futhark did not finish on a legal program\n{source}"),
            }
        }
        assert_eq!(compiled, 3);
    }

    /// The interface's inputs: one matrix per class, with the values a fixed
    /// sequence gives, and the same matrices in Futhark's textual format.
    ///
    /// The first value of **every** matrix is **-0.0**, on purpose: it puts
    /// the sign of a zero through every path, which is where the compiled
    /// program and the source semantics part company. Futhark's simplifier
    /// folds `x + 0` away, so a row that added zero would compile to the
    /// identity and keep `-0.0`'s sign, where the source semantics turns it
    /// into `+0.0` — a difference no comparison could see. No row does that
    /// any more: the `const` rows are a genuine constant, so they differ from
    /// the identity by VALUE. Futhark reads and writes the sign of a zero
    /// (`-0.0f32` in, `-0.0f32` out), so the value survives the round trip
    /// through the text.
    fn inputs(k: &FutharkKingdom, sizes: &BTreeMap<String, usize>) -> (BTreeMap<Ty, Mat>, String) {
        let mut map: BTreeMap<Ty, Mat> = BTreeMap::new();
        let mut text = String::new();
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        for d in &k.duals {
            if d.slot != "mat" {
                continue;
            }
            let dims: Vec<usize> = d.form.split('.').map(|r| *sizes.get(r).unwrap_or(&1)).collect();
            let mut data = Vec::with_capacity(dims[0] * dims[1]);
            for _ in 0..dims[0] * dims[1] {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                data.push(((rng >> 40) as f32) / 16777216.0 * 16.0 - 8.0);
            }
            if !data.is_empty() {
                data[0] = -0.0;
            }
            let mat = Mat::new(dims[0], dims[1], data).expect("the shape matches the values");
            let ty = k.ty(&d.dual).expect("a dual's own type");
            text.push_str(&format_matrix(&mat));
            map.insert(ty, mat);
        }
        (map, text)
    }

    /// The agreement between a compiled program and the interpreter.
    ///
    /// Two comparisons, and which one is used depends on what the program
    /// does:
    ///
    /// * [`assert_matrices_identical`] — bit for bit, NaN excepted. This is
    ///   for the paths whose arithmetic is exactly specified: a row function, a
    ///   `map`, a `map2`, a `transpose`. It tells the signed zeros apart, so a
    ///   row that differs from another only in the sign of a zero cannot slip
    ///   through;
    /// * [`assert_matrices_agree`] — exact values, NaN excepted, and no
    ///   tolerance. This is for the reduction path (`matmul`), where the
    ///   compiled program may legitimately differ from the source semantics:
    ///   Futhark's simplifier folds `0.0 + x` away when the reduction has a
    ///   single element, so the compiled product of a `[-0.0]` row is `-0.0`
    ///   where the source semantics gives `+0.0`.
    ///
    /// The tolerance that used to sit under both of them is **gone**: its
    /// stated reason was false. Futhark 0.28.1's `c` backend reduces with a
    /// sequential left fold, measured bit-identical to the interpreter's over
    /// 600 of 600 elements, and the order-sensitive probe
    /// `[16777216, 1, -16777216, 1]` tells a left fold from a tree sum in
    /// both. A margin that is never needed is a blind spot: it would hide a
    /// last-ulp wrong result from a spelling that drifted.
    ///
    /// The **shapes** must agree too, both dimensions, with no exception:
    /// Futhark's textual format carries a matrix with no elements as
    /// `empty([rows][cols]f32)`, so even a degenerate dimension survives the
    /// round trip through the text.
    fn assert_matrices_agree(got: &Mat, want: &Mat, source: &str) {
        assert_matrices(got, want, source, equal);
    }

    /// The same agreement, but bit for bit.
    fn assert_matrices_identical(got: &Mat, want: &Mat, source: &str) {
        assert_matrices(got, want, source, identical);
    }

    /// The shapes and the values, with the comparison the caller chose.
    fn assert_matrices(got: &Mat, want: &Mat, source: &str, same: fn(f32, f32) -> bool) {
        assert_eq!(got.rows, want.rows, "the program and the interpreter disagree on the row count\n{source}");
        assert_eq!(got.cols, want.cols, "the program and the interpreter disagree on the column count\n{source}");
        for i in 0..want.rows {
            for j in 0..want.cols {
                let (x, y) = (got.get(i, j), want.get(i, j));
                assert!(
                    same(x, y),
                    "element ({i}, {j}): futhark says {x}, the interpreter says {y}\n{source}"
                );
            }
        }
    }

    /// The degenerate shapes, checked against the real compiler.
    ///
    /// A dimension of zero is where the two representations differ most: a
    /// `Vec<Vec<f32>>` cannot hold a `[0][m]` matrix at all, and Futhark's
    /// `transpose` of one is `[m][0]` — `m` empty rows, not none. Futhark's
    /// textual format carries such a matrix as `empty([rows][cols]f32)`, so
    /// the shape survives the round trip and the comparison can be exact.
    #[test]
    fn the_degenerate_shapes_agree_with_futhark() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let dir = std::env::temp_dir().join("futhark_shapes_degen").join(format!("t{}", std::process::id()));

        // (the sizes, the tree, the class the result has)
        let cases: Vec<(BTreeMap<String, usize>, &str, &str)> = vec![
            // The transpose of a [0][3] matrix is [3][0]: three empty rows.
            (
                [("n".to_string(), 0usize), ("m".to_string(), 3), ("p".to_string(), 2)].into_iter().collect(),
                "transpose@mat.n.m",
                "mat.m.n",
            ),
            // A matmul over an EMPTY inner dimension is a zero matrix, not an
            // empty one: [2][0] times [0][2] is [2][2] of zeros.
            (
                [("n".to_string(), 2usize), ("m".to_string(), 0), ("p".to_string(), 2)].into_iter().collect(),
                "matmul@mat.n.mxmat.m.p",
                "mat.n.p",
            ),
        ];

        for (sizes, row, root) in &cases {
            let (inputs, text) = inputs(&k, sizes);
            let kids: Vec<&str> = match *row {
                "transpose@mat.n.m" => vec!["mat.n.m.input"],
                _ => vec!["mat.n.m.input", "mat.m.p.input"],
            };
            let tree = tree_of(&k, row, &kids);
            let source = program(&k, k.ty(root).expect("the root class"), &tree, sizes)
                .unwrap_or_else(|e| panic!("the program for {row} does not render: {e}"));
            let out = run_text_program(&source, &text, &dir, Duration::from_secs(120))
                .unwrap_or_else(|e| panic!("the program for {row} did not run: {e}\n{source}"));
            let got = parse_matrix(&out)
                .unwrap_or_else(|e| panic!("the output for {row} is not a matrix: {e}\n{out}"));
            let want = match k.eval_matrices(&tree, &inputs).expect("the interpreter evaluates") {
                MValue::Mat(m) => m,
                other => panic!("the value for {row} is not a matrix: {other:?}"),
            };
            assert_matrices_agree(&got, &want, &source);
        }
    }

    /// The proposal's round-trip invariant, for the shape kingdom: a tree the
    /// e-graph found is encodable in the layout, and expressing the gene it
    /// writes back to gives the same tree again.
    ///
    /// This is the property that makes an e-graph candidate *heritable*: the
    /// evolutionary loop needs a Karva gene, and the gene must denote exactly
    /// the program the e-graph found.
    #[test]
    fn a_candidate_writes_back_to_a_gene_and_comes_back() {
        use crate::futhark_shapes_egraph::{saturate_tree, Bridge};
        use crate::typed_gep::{fits, write_back};

        let k = kingdom();
        let bridge = Bridge::load();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let mut rng: u64 = 0x5EED_1234_ABCD_0005;
        let mut gene = vec![0u32; layout.width];
        for (pos, cell) in gene.iter_mut().enumerate() {
            let alphabet = layout.alphabet(&contract, pos, &const_len);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
        }
        let tree = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal").tree();

        // The gene and the tree are the same program, both ways round.
        fits(&contract, &layout, &tree).expect("the tree fits the layout");
        let written = write_back(&contract, &layout, &gene, &tree).expect("the write-back");
        assert_eq!(written, gene, "the write-back changed the gene");
        let again = express(&contract, &layout, &written, &const_len).expect("legal").tree();
        assert_eq!(again, tree, "the round trip is not the identity");

        // And every candidate the e-graph found survives it too.
        let sat = saturate_tree(&tree, &k, &bridge, 20, 3).expect("the saturation runs");
        for c in &sat.candidates {
            let t = c.gene.as_ref().expect("the candidate is spellable");
            let g = write_back(&contract, &layout, &gene, t).expect("the candidate fits the layout");
            let back = express(&contract, &layout, &g, &const_len).expect("legal").tree();
            assert_eq!(&back, t, "the candidate did not survive the round trip");
        }
    }

    /// The whole loop, closed: an e-graph candidate is written back to a
    /// Karva gene, expressed, rendered, **compiled by Futhark**, run, and
    /// compared with the interpreter.
    ///
    /// This is what the e-graph is *for*: not that it finds candidates, but
    /// that a candidate becomes a program Futhark accepts and that computes
    /// the same values as the program it came from.
    #[test]
    fn an_egraph_candidate_compiles_and_agrees() {
        use crate::futhark_shapes_egraph::{saturate_tree, Bridge};
        use crate::typed_gep::{express, write_back};

        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let bridge = Bridge::load();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let sizes = sizes();
        let (inputs, text) = inputs(&k, &sizes);
        let dir = std::env::temp_dir().join("futhark_shapes_egraph").join(format!("t{}", std::process::id()));
        let mut rng: u64 = 0x5EED_1234_ABCD_0006;
        let mut gene = vec![0u32; layout.width];
        for (pos, cell) in gene.iter_mut().enumerate() {
            let alphabet = layout.alphabet(&contract, pos, &const_len);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
        }
        let tree = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal").tree();
        let sat = saturate_tree(&tree, &k, &bridge, 12, 3).expect("the saturation runs");
        let mut checked = 0usize;
        for c in &sat.candidates {
            let t = c.gene.as_ref().expect("the candidate is spellable");
            let g = write_back(&contract, &layout, &gene, t).expect("the candidate fits the layout");
            let back = express(&contract, &layout, &g, &const_len).expect("legal").tree();
            let source = program(&k, root, &back, &sizes).expect("the program renders");
            let out = run_text_program(&source, &text, &dir, Duration::from_secs(120))
                .unwrap_or_else(|e| panic!("the candidate did not run: {e}\n{source}"));
            let got = parse_matrix(&out).unwrap_or_else(|e| panic!("the output is not a matrix: {e}\n{out}"));
            let want = match k.eval_matrices(&back, &inputs).expect("the interpreter evaluates") {
                MValue::Mat(m) => m,
                other => panic!("the candidate's value is not a matrix: {other:?}"),
            };
            assert_matrices_agree(&got, &want, &source);
            checked += 1;
        }
        assert!(checked > 0, "the e-graph found no candidate to check");
    }

    /// The symbol id of a row, by name.
    fn symbol_of(k: &FutharkKingdom, name: &str) -> u32 {
        k.rows
            .iter()
            .position(|r| r.name == name)
            .unwrap_or_else(|| panic!("the kingdom has no row {name}")) as u32
    }

    /// A tree of one row over the given rows.
    fn tree_of(k: &FutharkKingdom, name: &str, kids: &[&str]) -> Tree {
        tree_with(k, name, kids.iter().map(|c| tree_of(k, c, &[])).collect())
    }

    /// A tree of one row over the given trees.
    fn tree_with(k: &FutharkKingdom, name: &str, kids: Vec<Tree>) -> Tree {
        Tree { symbol: symbol_of(k, name), const_index: None, kids }
    }

    /// Every row's SPELLING means what the interpreter says it means.
    ///
    /// The random differential test above only exercises the rows a random
    /// gene happens to draw, so a row whose spelling had drifted from its
    /// meaning could survive it: the interpreter dispatches on the row's
    /// NAME, and the renderer uses the row's SPELLING, and nothing tied the
    /// two together. This test is deterministic and covers **every** row of
    /// the table that carries a meaning: each one gets a program, which
    /// Futhark compiles and runs, and the interpreter's value is compared
    /// with the program's.
    ///
    /// Two arguments of the **same** class get **different** values — the
    /// first the class's input, the second a scaled copy of it. Without
    /// that, a `map2` whose operands were swapped would pass: `id` sees the
    /// same value on both sides, `add` is commutative, and the other row
    /// functions ignore their second operand.
    #[test]
    fn every_rows_spelling_means_what_the_interpreter_says() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let sizes = sizes();
        let (inputs, text) = inputs(&k, &sizes);
        let dir = std::env::temp_dir().join("futhark_shapes_rows").join(format!("t{}", std::process::id()));

        // Every row with a meaning: the operators, and the function classes'
        // own rows (which are reached through the operators that take them).
        //
        // The two function classes are told apart by their class NAME, which
        // is the table's own naming: `fn.arr_f32_arr_f32` takes a
        // one-argument row function, `fn.arr2_f32_arr_f32_arr_f32` takes a
        // two-argument one, because Futhark's `map2` is curried.
        const FN: &str = "fn.arr_f32_arr_f32";
        const FN2: &str = "fn.arr2_f32_arr_f32_arr_f32";

        // The rows a function class can take: its elementary row functions
        // and its constant, which the table holds as a terminal.
        let function_rows_of = |k: &FutharkKingdom, class: &str| -> Vec<String> {
            k.rows
                .iter()
                .filter(|r| {
                    r.output == k.ty(class).expect("the class has a type")
                        && !r.name.contains('@')
                        && !r.name.ends_with(".input")
                })
                .map(|r| r.name.clone())
                .collect()
        };

        let mut cases: Vec<(String, String, Tree)> = Vec::new();
        for (i, row) in k.rows.iter().enumerate() {
            if row.kind != FutharkKind::Function || !row.name.contains('@') {
                continue;
            }
            let Some(out) = k.dual(row.output).map(|d| d.dual.clone()) else {
                continue;
            };
            // The arguments, with the same-class ones made different from
            // each other.
            let mut taken: BTreeMap<String, usize> = BTreeMap::new();
            let mut kids = Vec::new();
            for arg in &row.inputs {
                let Some(class) = k.dual(*arg).map(|d| d.dual.clone()) else {
                    continue;
                };
                let n = taken.entry(class.clone()).or_insert(0);
                let tree = if class == FN || class == FN2 {
                    // A function argument: one of that class's own rows, and
                    // a different one each time round.
                    let names = function_rows_of(&k, &class);
                    if names.is_empty() {
                        continue;
                    }
                    tree_of(&k, &names[i % names.len()], &[])
                } else if *n == 0 {
                    tree_of(&k, &format!("{class}.input"), &[])
                } else {
                    scaled(&k, &class)
                };
                *n += 1;
                kids.push(tree);
            }
            cases.push((row.name.clone(), out, tree_with(&k, &row.name, kids)));
        }

        // And every row of the two function classes, used through the
        // operator that takes it.
        for class in [FN, FN2] {
            let op = if class == FN { "map_rows" } else { "map2_rows" };
            let cls = "mat.n.m";
            for name in function_rows_of(&k, class) {
                let op_row = k
                    .rows
                    .iter()
                    .find(|r| r.name == format!("{op}@{cls}"))
                    .expect("the operator exists");
                let op_out = k.dual(op_row.output).expect("the operator has a class").dual.clone();
                let kids = if op == "map2_rows" {
                    vec![
                        tree_of(&k, &name, &[]),
                        tree_of(&k, &format!("{cls}.input"), &[]),
                        scaled(&k, cls),
                    ]
                } else {
                    vec![tree_of(&k, &name, &[]), tree_of(&k, &format!("{cls}.input"), &[])]
                };
                cases.push((name, op_out, tree_with(&k, &op_row.name.clone(), kids)));
            }
        }

        assert!(cases.len() > 50, "every row of the table is covered: {}", cases.len());
        let mut checked = 0usize;
        for (row, root, tree) in &cases {
            let source = program(&k, k.ty(root).expect("the root class"), tree, &sizes)
                .unwrap_or_else(|e| panic!("the program for {row} does not render: {e}"));
            let out = run_text_program(&source, &text, &dir, Duration::from_secs(120))
                .unwrap_or_else(|e| panic!("the program for {row} did not run: {e}\n{source}"));
            let got = parse_matrix(&out)
                .unwrap_or_else(|e| panic!("the output for {row} is not a matrix: {e}\n{out}"));
            let want = match k.eval_matrices(tree, &inputs).expect("the interpreter evaluates") {
                MValue::Mat(m) => m,
                other => panic!("the value for {row} is not a matrix: {other:?}"),
            };
            // A reduction is compared on its VALUES, and everything else BIT
            // FOR BIT: the elementwise paths have an exactly specified
            // arithmetic, so the compiled program must agree down to the last
            // bit — and that is what keeps a row that differs from another
            // only in the sign of a zero from slipping through.
            if row.starts_with("matmul@") {
                assert_matrices_agree(&got, &want, &format!("the row {row}\n{source}"));
            } else {
                assert_matrices_identical(&got, &want, &format!("the row {row}\n{source}"));
            }
            checked += 1;
        }
        assert_eq!(checked, cases.len(), "every row's spelling was checked");
    }

    /// A matrix of the same class as `cls`, but a DIFFERENT value: the row
    /// function `scale2` applied to the class's input.
    fn scaled(k: &FutharkKingdom, cls: &str) -> Tree {
        tree_with(
            k,
            &format!("map_rows@{cls}"),
            vec![
                tree_of(k, "fn.arr_f32_arr_f32.scale2", &[]),
                tree_of(k, &format!("{cls}.input"), &[]),
            ],
        )
    }

    /// A term of the e-graph's syntax: a constructor with its children, or a
    /// variable.
    #[derive(Debug, Clone)]
    enum Term {
        Ctor(String, Vec<Term>),
        Var(String),
    }

    /// Split an s-expression's body into its top-level terms.
    fn split_terms(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut depth = 0usize;
        let mut start = 0usize;
        for (i, c) in text.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                c if c.is_whitespace() && depth == 0 => {
                    if i > start {
                        out.push(text[start..i].to_string());
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }
        if start < text.len() {
            out.push(text[start..].to_string());
        }
        out
    }

    /// Parse one term of the e-graph's syntax.
    fn parse_term(text: &str) -> Result<Term, String> {
        let t = text.trim();
        if let Some(rest) = t.strip_prefix('(') {
            let inner = rest.strip_suffix(')').ok_or_else(|| format!("{text:?} is not closed"))?;
            let mut parts = split_terms(inner);
            if parts.is_empty() {
                return Err(format!("{text:?} names no constructor"));
            }
            let head = parts.remove(0);
            let kids = parts.iter().map(|p| parse_term(p)).collect::<Result<Vec<_>, _>>()?;
            Ok(Term::Ctor(head, kids))
        } else {
            Ok(Term::Var(t.to_string()))
        }
    }

    /// A rule's two sides, as the e-graph's text.
    fn rules_of(ruleset: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for line in ruleset.lines() {
            let s = line.trim();
            if !s.starts_with("(rewrite ") {
                continue;
            }
            let pattern = s
                .strip_prefix("(rewrite ")
                .and_then(|p| p.split(" :ruleset").next())
                .unwrap_or_default();
            let terms = split_terms(pattern);
            assert_eq!(terms.len(), 2, "a rewrite has two sides: {s}");
            out.push((terms[0].clone(), terms[1].clone()));
        }
        out
    }

    /// A term as a kingdom tree, with every variable replaced by a value of
    /// the class that variable has.
    ///
    /// **Each variable NAME is bound once**, and every occurrence of that name
    /// gets the same value. Two *different* names of the same class get
    /// *different* values — the first gets the class's input, the second a
    /// scaled copy of it. That is what makes the test able to tell a rule
    /// that renames a variable (`… x … => y`) from a rule that keeps it: if
    /// `x` and `y` were both the class's input, the renamed rule would
    /// compute the same thing on both sides and pass.
    fn instantiate(
        term: &Term,
        var_class: Option<&str>,
        bindings: &mut BTreeMap<String, (String, usize)>,
        k: &FutharkKingdom,
        bridge: &Bridge,
    ) -> Result<Tree, String> {
        match term {
            Term::Var(v) => {
                let class = var_class.ok_or_else(|| format!("the variable {v} has no class"))?;
                if !bindings.contains_key(v) {
                    let taken = bindings.values().filter(|(cls, _)| cls.as_str() == class).count();
                    bindings.insert(v.clone(), (class.to_string(), taken));
                }
                let (_, index) = &bindings[v];
                Ok(value_of_class(k, class, *index))
            }
            Term::Ctor(ctor, kids) => {
                let row_name = bridge
                    .row_of(ctor)
                    .ok_or_else(|| format!("the constructor {ctor} has no kingdom row"))?
                    .to_string();
                let row = k
                    .rows
                    .iter()
                    .find(|r| r.name == row_name)
                    .ok_or_else(|| format!("the kingdom has no row {row_name}"))?;
                if kids.len() != row.inputs.len() {
                    return Err(format!(
                        "the constructor {ctor} takes {} arguments, and the term has {}",
                        row.inputs.len(),
                        kids.len()
                    ));
                }
                let mut child_trees = Vec::new();
                for (i, kid) in kids.iter().enumerate() {
                    let class = k
                        .dual(row.inputs[i])
                        .ok_or_else(|| format!("the argument {i} of {ctor} has no class"))?;
                    let kid_class = match kid {
                        Term::Var(_) => Some(class.dual.as_str()),
                        Term::Ctor(..) => None,
                    };
                    child_trees.push(instantiate(kid, kid_class, bindings, k, bridge)?);
                }
                Ok(Tree { symbol: symbol_of(k, &row_name), const_index: None, kids: child_trees })
            }
        }
    }

    /// A value of a class: the class's own input, or — for the second and
    /// later values of the same class — a scaled copy of it, so that two
    /// values of one class are never the same value.
    fn value_of_class(k: &FutharkKingdom, class: &str, index: usize) -> Tree {
        let input = format!("{class}.input");
        if index == 0 {
            return tree_of(k, &input, &[]);
        }
        // A scaled copy: `map` the identity's opposite over the input. The
        // row exists for every matrix class; a class without one falls back
        // to the input, which is all a class of scalars can do.
        let map_row = format!("map_rows@{class}");
        let scale = "fn.arr_f32_arr_f32.scale2";
        if k.rows.iter().any(|r| r.name == map_row) && k.rows.iter().any(|r| r.name == scale) {
            tree_with(
                k,
                &map_row,
                vec![tree_of(k, scale, &[]), tree_of(k, &input, &[])],
            )
        } else {
            tree_of(k, &input, &[])
        }
    }

    /// Every e-graph rule means what its comment says.
    ///
    /// The rules are what let the e-graph offer a candidate the engine can
    /// inherit, so a rule that is not an equivalence would put a wrong
    /// program into the evolutionary loop. The generator's witness test
    /// checks each rule's TEXT against a hand-written schema; this test
    /// checks each rule's MEANING, by taking the rule as it is written,
    /// replacing the pattern's variables with the kingdom's own terminals,
    /// rendering both sides into real Futhark programs, and comparing what
    /// they compute.
    #[test]
    fn every_egraph_rule_is_an_equivalence() {
        use crate::futhark_shapes_egraph::SHAPE_RULESET;

        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let bridge = Bridge::load();
        let sizes = sizes();
        // The interpreter is not needed here: the two sides of a rule are
        // both compiled, and they are compared with each other.
        let (_, text) = inputs(&k, &sizes);
        let dir = std::env::temp_dir().join("futhark_shapes_rules").join(format!("t{}", std::process::id()));

        let rules = rules_of(SHAPE_RULESET);
        assert_eq!(rules.len(), 54, "three rules for each of the nine classes, and the transposed product for each of the twenty-seven products");
        let mut checked = 0usize;
        for (lhs, rhs) in &rules {
            let left = parse_term(lhs).expect("the left side parses");
            let right = parse_term(rhs).expect("the right side parses");
            // The variable bindings are per rule: each variable NAME is bound
            // once, and two names of the same class get different values.
            let mut bindings: BTreeMap<String, (String, usize)> = BTreeMap::new();

            // The left side's own class: the row its root constructor is.
            let root_ctor = match &left {
                Term::Ctor(c, _) => c.clone(),
                Term::Var(v) => panic!("the left side of a rule is not a constructor: {v}"),
            };
            let root_row_name = bridge.row_of(&root_ctor).expect("the root constructor has a row");
            let root_row = k.rows.iter().find(|r| r.name == root_row_name).expect("the row");
            let root_class = k.dual(root_row.output).expect("the result has a class");

            let left_tree = instantiate(&left, None, &mut bindings, &k, &bridge)
                .expect("the left side instantiates");
            // The right side's variable has the left side's class.
            let right_tree = instantiate(&right, Some(root_class.dual.as_str()), &mut bindings, &k, &bridge)
                .expect("the right side instantiates");

            let mut outputs = Vec::new();
            for tree in [&left_tree, &right_tree] {
                let source = program(&k, root_row.output, tree, &sizes)
                    .unwrap_or_else(|e| panic!("a side of the rule does not render: {e}"));
                let out = run_text_program(&source, &text, &dir, Duration::from_secs(120))
                    .unwrap_or_else(|e| panic!("a side of the rule did not run: {e}\n{source}"));
                outputs.push((source, out));
            }
            let (left_source, left_out) = &outputs[0];
            let (right_source, right_out) = &outputs[1];
            let left_value = parse_matrix(left_out)
                .unwrap_or_else(|e| panic!("the left side's output is not a matrix: {e}\n{left_out}"));
            let right_value = parse_matrix(right_out)
                .unwrap_or_else(|e| panic!("the right side's output is not a matrix: {e}\n{right_out}"));
            // Both sides are compiled by the same compiler, and the rules
            // are elementwise, so they must agree BIT FOR BIT.
            assert_matrices_identical(&left_value, &right_value, &format!("{left_source}\n-- against --\n{right_source}"));
            checked += 1;
        }
        assert_eq!(checked, rules.len(), "every rule was checked");
    }

    /// The committed artefacts are what the engine loads — the tables are
    /// `include_str!`'d — and the generator is what says what they should
    /// be. So the suite runs the generator's own check: a hand edit to a
    /// committed table is a failure of the test suite, not something a
    /// developer has to remember to look for.
    ///
    /// A missing `python3` is a FAILURE, not a skip: the check is part of
    /// the contract, and a machine that cannot run it cannot keep the
    /// contract.
    #[test]
    fn the_committed_artefacts_are_what_the_generator_produces() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = root.join("kingdoms/futhark_shapes/gen_tables.py");
        assert!(script.exists(), "the generator is at {script:?}");
        let out = Command::new("python3")
            .arg(&script)
            .arg("--check")
            .current_dir(root)
            .output()
            .expect("python3 runs the generator's check");
        assert!(
            out.status.success(),
            "the generator's check failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// The strong check: the compiled program and the interpreter agree, element
    /// by element, on a random gene — up to the numeric policy above, which
    /// is numeric rather than bitwise.
    ///
    /// The 1-D kingdom has this check, and it is what makes the kingdom's
    /// claim mean something: a program that compiles but computes the wrong
    /// thing would pass a compile-only test. The shape kingdom's programs are
    /// two-dimensional, so they run through Futhark's *textual* format
    /// (`[[1.0,2.0],[3.0,4.0]]`) rather than the C library API: no C driver
    /// is needed for a matrix.
    #[test]
    fn the_rendered_programs_agree_with_the_interpreter() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let sizes = sizes();
        let (inputs, text) = inputs(&k, &sizes);
        let dir = std::env::temp_dir().join("futhark_shapes_diff").join(format!("t{}", std::process::id()));
        let mut checked = 0usize;
        for seed in 0..3u64 {
            let mut rng: u64 = 0x5EED_1234_ABCD_0004 ^ seed;
            let mut gene = vec![0u32; layout.width];
            for (pos, cell) in gene.iter_mut().enumerate() {
                let alphabet = layout.alphabet(&contract, pos, &const_len);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
            }
            let e = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal");
            let tree = e.tree();
            let source = program(&k, root, &tree, &sizes).expect("the program renders");
            let out = run_text_program(&source, &text, &dir, Duration::from_secs(120))
                .unwrap_or_else(|e| panic!("the program did not run: {e}\n{source}"));
            let got = parse_matrix(&out).unwrap_or_else(|e| panic!("the output is not a matrix: {e}\n{out}"));
            let want = match k.eval_matrices(&tree, &inputs).expect("the interpreter evaluates") {
                MValue::Mat(m) => m,
                other => panic!("the program's value is not a matrix: {other:?}"),
            };
            assert_matrices_agree(&got, &want, &source);
            checked += 1;
        }
        assert_eq!(checked, 3, "every program was checked against the interpreter");
    }
}
