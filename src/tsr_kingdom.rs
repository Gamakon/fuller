//! THE SYMBOLIC REGRESSION KINGDOMS AS ROWS of the gene frame's fixed schema (`docs/DESIGN_geneframe.md` 3.3, 8.1
//! and 8.3; `docs/PLAN_futhark_kingdom.md` 7.7 task K1 and section 8). TSR is Typed Symbolic Regression.
//!
//! Every variant is the SAME one-type kingdom, `F` with the properties `depth`, `order`, `fam`, `arm` and `ties`
//! derived by up Flow rows; the variants differ only by their `must` Guard rows (the kingdom `tsr.<variant>`, `sr` for depth, of the
//! master tables `kingdoms/master/`, written by `kingdoms/tsr/gen_tables.py`). [`rows::TsrRows`] loads one for a law
//! of n input columns.
//!
//! - `single`: no guard, classical GEP over the 18-operator alphabet.
//! - `depth`: THE SYMBOLIC REGRESSION KINGDOM of the plan's K1 row: `guard sin: in0.depth <= 1 must -> {in0}` on the
//!   ten raising unaries. It replaces the old depth ladder of types (`F`, `T1`, `T2`, deleted): a tower of three
//!   transcendentals stays representable in the gene and is INEXPRESSIBLE in the program, the third rendered as its
//!   input.
//! - `order`, `order_depth`, `order_depth_windows`: TSR's stream-order ceiling (`out.order <= 3` on the four
//!   binaries), with the depth guard, and with the argument windows (`in0.order <= 2` on seven unaries).
//!
//! The old row forms (one type per dual `shape.level` in `types.tsv`/`functions.tsv`, the `ROOT` twins) were
//! deleted with the move to the four kinds of row; programs worth keeping are kept as text (phylu's champion pool).

/// The variants as rows of the gene frame's fixed schema (one data type, properties and guards).
pub mod rows;

/// The kingdom's name in the symbol table.
pub const TSR: &str = "TSR";

/// One kingdom of the plan's arms: which limits the guards carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TsrVariant {
    /// Neither limit: one data type `F`, no guard.
    Single,
    /// Stream order (arm B).
    Order,
    /// Transcendental depth (arm E): the symbolic regression kingdom (K1).
    Depth,
    /// Both limits (arm C).
    OrderDepth,
    /// Both limits and the operator windows (arm F).
    OrderDepthWindows,
}

impl TsrVariant {
    pub const ALL: [TsrVariant; 5] =
        [TsrVariant::Single, TsrVariant::Order, TsrVariant::Depth, TsrVariant::OrderDepth, TsrVariant::OrderDepthWindows];

    /// The directory under `kingdoms/tsr`.
    pub fn dir(self) -> &'static str {
        match self {
            TsrVariant::Single => "single",
            TsrVariant::Order => "order",
            TsrVariant::Depth => "depth",
            TsrVariant::OrderDepth => "order_depth",
            TsrVariant::OrderDepthWindows => "order_depth_windows",
        }
    }
}
