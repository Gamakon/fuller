//! The WGSL kingdom: a compute kernel read through naga into the kingdom's
//! chromosome shape (`kingdoms/wgsl/README.md`, `docs/PLAN_wgsl_kernel_reader.md`).
//!
//! The genome is the pure expression DAG a kernel computes between its loads
//! and its stores; the statements around it (loops, branches, calls) are a
//! fixed scaffold. [`reader`] cuts a kernel at its effects: every store, every
//! branch condition, every call argument and every return value is a root,
//! and each root's expression tree is rendered as an s-expression over the
//! kingdom's `class.instance` names ([`table`]), the form a kingdom's
//! generic Karva pair (`karva::terms_to_karva_generic`) encodes.
//!
//! Compiled only under the `wgsl` feature (naga in and out).

pub mod canonical;
pub mod check;
pub mod closeness;
pub mod cpu;
pub mod aggregate;
pub mod chromosome;
pub mod generator;
pub mod flow;
/// The kernel texts the tests read, tracked as fixtures (T63).
#[cfg(test)]
pub(crate) mod fixture_texts;
/// Which tracked kernels the safe reader does not read, asserted against REFUSED.md (T43).
#[cfg(test)]
mod reader_census;
pub mod harness;
pub mod gram;
pub mod cuneus_originals;
pub mod image;
pub mod individual;
#[cfg(feature = "gpu")]
pub mod llm;
pub mod pairs;
pub mod probes;
pub mod infer;
pub mod interface;
pub mod interp;
pub mod knob;
pub mod legality;
pub mod loader;
pub mod lower;
pub mod mutate;
pub mod naga_names;
pub mod naga_shape;
pub mod naga_snapshot;
pub mod oracle;
pub mod params;
pub mod reader;
pub mod rewrites;
pub mod rules;
pub mod sections;
pub mod rules_more;
pub mod render;
pub mod safe_karva;
pub mod roundtrip;
pub mod safe_reader;
pub mod safety;
pub mod scaffold;
pub mod sketch;
pub mod table;
pub mod tree_diff;
pub mod karva_seed;
pub mod true_gram;
pub mod true_kingdom;
pub mod true_module;
pub mod true_reader;
pub mod true_share;
pub mod true_opt;
pub mod typed_gene;
pub mod target;
pub mod versions;
pub mod work_guard;

pub use chromosome::{chromosome, chromosome_typed, chromosome_with_roots, ChromosomeOptions, WgslChromosome};
pub use infer::{infer_function, FunctionTypes, NodeType, TypeStats, TypedTree};
pub use interp::{Interp, Invocation, Memory, Value};
pub use legality::{Decision, Placement, Reason};
pub use loader::{wgsl_duals, wgsl_table, WgslDual, WgslKingdom, WgslRow, WGSL};
pub use reader::{read, Kernel, KernelFunction, Node, Root, RootKind};
pub use scaffold::{rebuild, rebuild_folded, round_trip, Rebuilt};
pub use table::{FunctionTable, ScalarKind, Template};
pub use versions::split_version;
