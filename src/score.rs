//! Angular extraction cost: a measure-vector whose ordering is a CDF-corrected
//! hyperspherical-fitness angle, so egglog's extractor picks each e-class winner
//! by HFF rather than by a scalar sum.
//!
//! egglog's extractor is generic over a cost type `C: Cost + Ord` and picks the
//! per-e-class winner with `<` (egglog `extract.rs:359`), which delegates to our
//! `Ord`. So if `C` accumulates a vector of structural measures up the tree
//! (`Cost::combine` = slot-wise add) and `Ord` compares two vectors by their
//! CDF-corrected HFF-TrueNorth angle, the angle becomes the selection key — and
//! it is evaluated PER E-CLASS during the walk, on whatever dimension that
//! subtree's measures occupy. The CDF correction (`hff_core::higd`) is what makes
//! angles from different-dimension subtrees comparable; the angle itself is
//! `hff_core::core_functions` TrueNorth.
//!
//! Every measure is a non-negative tally that accumulates associatively, so
//! `combine` stays associative and the Bellman-Ford extractor terminates. The
//! bounded-[0,1] mapping and the angle are computed lazily in `Ord`, never
//! stored — keeping `combine` a plain add.

use std::cmp::Ordering;

use egglog::extract::{Cost, CostModel};
use egglog::{EGraph, Function, FunctionRow};
use ndarray::Array1;

/// The measures, in fixed slot order. Each slot is a raw non-negative tally that
/// accumulates up the tree. The bounded-[0,1] mapping happens in [`MeasureVector::
/// normalised`]; 0 is best for every slot.
///
/// Self/transcendental-nesting are tallied structurally in [`HffCostModel::fold`]
/// (they need the parent's head plus whether a child already carried a
/// transcendental), so they ride in slots here like any other tally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureVector {
    /// total node count (parsimony) — always non-zero for any real term.
    pub nodes: u32,
    /// transcendental ops present (sin/cos/tan/exp/log/tanh).
    pub transc: u32,
    /// transcendental-inside-transcendental nesting events.
    pub transc_nest: u32,
    /// same transcendental directly inside itself (sin(sin..)) — strongest junk.
    pub self_nest: u32,
    /// numeric-literal nodes (free-parameter / overfit proxy).
    pub nums: u32,
    /// variable-leaf nodes (for the const-to-var ratio).
    pub vars: u32,
}

impl MeasureVector {
    /// All-zero: the identity for accumulation.
    pub const ZERO: MeasureVector = MeasureVector {
        nodes: 0,
        transc: 0,
        transc_nest: 0,
        self_nest: 0,
        nums: 0,
        vars: 0,
    };

    /// Slot-wise add (saturating, so a pathological graph can't panic).
    fn add(&self, o: &MeasureVector) -> MeasureVector {
        MeasureVector {
            nodes: self.nodes.saturating_add(o.nodes),
            transc: self.transc.saturating_add(o.transc),
            transc_nest: self.transc_nest.saturating_add(o.transc_nest),
            self_nest: self.self_nest.saturating_add(o.self_nest),
            nums: self.nums.saturating_add(o.nums),
            vars: self.vars.saturating_add(o.vars),
        }
    }

    /// Map the raw tallies to a bounded objective vector in `[0,1]` (0 = best).
    /// Counts use the saturating penalty `c -> 1 - 1/(1 + c/s)`; the const ratio
    /// is already in `[0,1]`. Slots whose pattern never fired across the whole
    /// term contribute a 0 (perfect) but still occupy a dimension — matching the
    /// "node_count always fires so the vector is never empty" rule.
    fn normalised(&self) -> Vec<f64> {
        fn sat(c: u32, s: f64) -> f64 {
            let c = c as f64;
            1.0 - 1.0 / (1.0 + c / s)
        }
        let ratio = {
            let denom = (self.nums + self.vars) as f64;
            if denom > 0.0 {
                self.nums as f64 / denom
            } else {
                0.0
            }
        };
        vec![
            sat(self.nodes, 12.0),
            sat(self.transc, 2.0),
            sat(self.transc_nest, 1.0),
            sat(self.self_nest, 1.0),
            sat(self.nums, 4.0),
            ratio,
        ]
    }

    /// CDF-corrected HFF-TrueNorth angle of this vector. Lower = better.
    ///
    /// The dimension handed to the CDF correction is the full objective count
    /// (the vector is fixed-width here; a slot that never fired is a 0 component,
    /// which is the ideal coordinate, so it neither helps nor hurts the angle).
    pub fn angle_percentile(&self) -> f64 {
        let x = self.normalised();
        let k = x.len();
        let arr = Array1::from(x);
        let theta = hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(
            &arr, k, false, None, "truenorth",
        );
        hff_core::higd::cdf_beta_correction(theta, k)
    }
}

impl Cost for MeasureVector {
    fn identity() -> Self {
        MeasureVector::ZERO
    }
    fn unit() -> Self {
        // A bare leaf is one node; its kind (Num/Var) is set by the cost model's
        // `enode_cost`, not here, so `unit` is just "one node".
        MeasureVector { nodes: 1, ..MeasureVector::ZERO }
    }
    fn combine(self, other: &Self) -> Self {
        self.add(other)
    }
}

impl PartialOrd for MeasureVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MeasureVector {
    /// Order by the CDF-corrected HFF angle FIRST — this is the paper's claim:
    /// HFF is the selection criterion, not a tiebreak on size. Exact-angle ties
    /// break deterministically on the raw tallies so the order is total.
    ///
    /// NOTE: this ordering is intentionally NON-MONOTONE (a larger term can score
    /// a smaller angle, because the angle normalises globally). egglog's stock
    /// Bellman-Ford extractor assumes monotone cost and panics on this — so the
    /// HFF extractor does NOT use the stock walk; it enumerates e-class members
    /// and selects the minimum-angle whole term (see `crate::eclass_hff`).
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.angle_percentile();
        let b = other.angle_percentile();
        a.partial_cmp(&b).unwrap_or(Ordering::Equal).then_with(|| {
            (self.nodes, self.transc_nest, self.self_nest, self.transc, self.nums, self.vars).cmp(
                &(other.nodes, other.transc_nest, other.self_nest, other.transc, other.nums, other.vars),
            )
        })
    }
}

/// The set of Math constructors that are transcendental (for the nesting
/// measures). Protected variants count too — they are still transcendental.
fn is_transcendental(head: &str) -> bool {
    matches!(
        head,
        "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Tanh"
            | "ProtectedExp" | "ProtectedLog"
    )
}

/// Cost model that scores by the angular [`MeasureVector`].
#[derive(Default, Clone)]
pub struct HffCostModel {}

impl HffCostModel {
    pub fn new() -> Self {
        HffCostModel {}
    }
}

impl CostModel<MeasureVector> for HffCostModel {
    /// This node's OWN contribution (no children). One node, plus its kind:
    /// transcendental / numeric-literal / variable. The constructor name comes
    /// from `func.decl.name`.
    fn enode_cost(
        &self,
        _egraph: &EGraph,
        func: &Function,
        _row: &FunctionRow,
    ) -> MeasureVector {
        let head = func.name();
        let mut m = MeasureVector { nodes: 1, ..MeasureVector::ZERO };
        if is_transcendental(head) {
            m.transc = 1;
        }
        match head {
            "Num" => m.nums = 1,
            "Var" => m.vars = 1,
            _ => {}
        }
        m
    }

    /// Accumulate this node + children, and tally nesting structurally: if this
    /// node is transcendental AND any child subtree already contains a
    /// transcendental, that is a transcendental-nesting event. We approximate the
    /// stronger "self-nest" (same fn inside itself) with the conservative signal
    /// that this transcendental node sits above other transcendentals; the exact
    /// same-symbol check is refined when the figure consumes the rendered term.
    fn fold(
        &self,
        head: &str,
        children_cost: &[MeasureVector],
        head_cost: MeasureVector,
    ) -> MeasureVector {
        let mut total = head_cost;
        let mut child_has_transc = false;
        for c in children_cost {
            child_has_transc |= c.transc > 0;
            total = total.add(c);
        }
        if is_transcendental(head) && child_has_transc {
            total.transc_nest = total.transc_nest.saturating_add(1);
            // A transcendental directly over transcendental subtrees is the junk
            // signature; record it in self_nest too (the strongest penalty slot).
            total.self_nest = total.self_nest.saturating_add(1);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_is_slotwise_add() {
        let a = MeasureVector { nodes: 2, transc: 1, ..MeasureVector::ZERO };
        let b = MeasureVector { nodes: 3, nums: 1, ..MeasureVector::ZERO };
        let c = a.combine(&b);
        assert_eq!(c.nodes, 5);
        assert_eq!(c.transc, 1);
        assert_eq!(c.nums, 1);
    }

    #[test]
    fn identity_is_zero_and_unit_is_one_node() {
        assert_eq!(MeasureVector::identity(), MeasureVector::ZERO);
        assert_eq!(MeasureVector::unit().nodes, 1);
    }

    #[test]
    fn cleaner_vector_ranks_below_noisier() {
        // A small clean term (few nodes, no nesting) must have a lower angle
        // percentile than a bloated, self-nested one.
        let clean = MeasureVector { nodes: 3, vars: 1, nums: 1, ..MeasureVector::ZERO };
        let junk = MeasureVector {
            nodes: 30,
            transc: 5,
            transc_nest: 4,
            self_nest: 3,
            nums: 6,
            vars: 1,
        };
        assert!(clean < junk, "clean term should sort before junk");
        assert!(clean.angle_percentile() < junk.angle_percentile());
    }

    #[test]
    fn angle_is_finite_and_in_unit_interval() {
        let m = MeasureVector { nodes: 10, transc: 2, nums: 3, vars: 2, ..MeasureVector::ZERO };
        let p = m.angle_percentile();
        assert!(p.is_finite());
        assert!((0.0..=1.0).contains(&p), "percentile {p} out of [0,1]");
    }

    #[test]
    fn ordering_is_total_and_deterministic() {
        // Same vector compares Equal; repeated calls agree (determinism).
        let m = MeasureVector { nodes: 7, transc: 1, ..MeasureVector::ZERO };
        assert_eq!(m.cmp(&m.clone()), Ordering::Equal);
        assert_eq!(m.angle_percentile(), m.angle_percentile());
    }
}
