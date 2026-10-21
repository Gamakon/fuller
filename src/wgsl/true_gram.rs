//! The shipped Gram kernel (`kingdoms/wgsl/gram/shipped.wgsl`) written directly as a tree of `kingdoms/wgsl_true` rows.
//!
//! The shipped text: one invocation per output cell `(lr, c)`, `acc = 0; for t in 0..m { acc += phi[t*d + gr] * phi[t*d + c] }`,
//! `h[lr*d + c] = acc` (the accumulate into `h` reads zero on the first band), the workgroup 16 x 16. Each construct is
//! the row that spells it; a construct with no row is a table gap, returned as an error naming it, never substituted for.

use crate::typed_gep::Tree;
use crate::wgsl::true_kingdom::TrueKingdom;

/// A node by row name; `value` is the text of the `C_T` entry a constant placeholder reads.
fn node(k: &TrueKingdom, name: &str, value: Option<&str>, kids: Vec<Tree>) -> Result<Tree, String> {
    let i = k.rows.iter().position(|r| r.name == name).ok_or_else(|| format!("table gap: no row {name}"))?;
    let const_index = match value {
        None => None,
        Some(v) => Some(
            k.constants
                .get(&k.rows[i].out)
                .and_then(|c| c.iter().position(|t| t == v))
                .ok_or_else(|| format!("table gap: {name} has no constant {v:?} in C_T of {}", k.type_name(k.rows[i].out)))? as u32,
        ),
    };
    Ok(Tree { symbol: i as u32, const_index, kids })
}

/// `phi[t*d + x]` with `t` the loop counter and `x` the thread's row or column.
fn phi_at(k: &TrueKingdom, axis: &str) -> Result<Tree, String> {
    let (as_index, own) = if axis == "row" { ("safe.row_as_index_h", "own.u32_threadrow_h") } else { ("safe.col_as_index_h", "own.u32_threadcol_h") };
    let t_times_d = node(k, "index.mul", None, vec![node(k, "index.of_tcounter1", None, vec![node(k, "safe.tile_counter1", None, vec![])?])?, node(k, "index.of_count", None, vec![node(k, "load.d", None, vec![])?])?])?;
    let idx = node(k, "index.add", None, vec![t_times_d, node(k, as_index, None, vec![node(k, own, None, vec![])?])?])?;
    node(k, "load.phi", None, vec![idx])
}

/// The shipped kernel with the workgroup shape `shape` (the text of a `grid.shape` constant, e.g. "16 16").
pub fn shipped_gram_tree(k: &TrueKingdom, shape: &str) -> Result<Tree, String> {
    let grid = node(k, "grid.of", None, vec![node(k, "literal.shape", Some(shape), vec![])?, node(k, "literal.blk_col", Some("1"), vec![])?, node(k, "literal.blk_row", Some("1"), vec![])?, node(k, "own.u32_ktile", None, vec![])?])?;
    let acc = || node(k, "reg.rf0", None, vec![]);
    let product = node(k, "arith.mul", None, vec![phi_at(k, "row")?, phi_at(k, "col")?])?;
    let update = node(k, "reg.set_f_L1_f0", None, vec![acc()?, node(k, "arith.add", None, vec![node(k, "reg.get_f", None, vec![acc()?])?, product])?])?;
    let bound = node(k, "bound.of_uniform_6", None, vec![node(k, "uniform.m", None, vec![])?])?;
    let looped = node(k, "loop.tileu_6_L0_f0", None, vec![bound, node(k, "block.one_L1_f0", None, vec![update])?])?;
    let store = node(k, "store.h_O0_f1", None, vec![node(k, "own.u32_threadrow_h", None, vec![])?, node(k, "own.u32_threadcol_h", None, vec![])?, node(k, "reg.get_f", None, vec![acc()?])?])?;
    node(k, "kernel.entry_safe", None, vec![grid, node(k, "block.one_L0_f1", None, vec![looped])?, node(k, "block.one_O0_f1", None, vec![store])?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::{express, write_back, GeneLayout};
    use crate::wgsl::true_kingdom::{default_dir, random_gene, Rng, Share};

    /// The tree is built from rows, the write-back puts it in a gene of the N = 8 layout and the walk reads the same tree
    /// back; "16 16" is a table gap (not in the grid shapes) and is reported as one.
    #[test]
    fn the_shipped_gram_kernel_is_a_tree_of_rows_that_round_trips_through_a_gene() {
        let k = TrueKingdom::load(&default_dir()).expect("table");
        let c = k.relations().check().expect("check");
        let layout = GeneLayout::from_program_size(&c, k.root, 8).expect("layout");
        let const_len = k.const_lens();
        let err = shipped_gram_tree(&k, "16 16").expect_err("16 16 is not a grid shape of the table");
        assert!(err.starts_with("table gap"), "{err}");
        let tree = shipped_gram_tree(&k, "8 8").expect("tree");
        let previous = random_gene(&c, &layout, &const_len, &mut Rng(9), Share::Half, true);
        let gene = write_back(&c, &layout, &previous, &tree).expect("write-back");
        assert_eq!(express(&c, &layout, &gene, &const_len).expect("express").tree(), tree);
        assert!(k.render(&tree).expect("render").contains("loop"));
    }
}
