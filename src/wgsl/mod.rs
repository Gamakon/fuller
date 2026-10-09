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

pub mod chromosome;
pub mod generator;
pub mod infer;
pub mod interp;
pub mod legality;
pub mod loader;
pub mod naga_names;
pub mod oracle;
pub mod reader;
pub mod scaffold;
pub mod table;
pub mod versions;

pub use chromosome::{chromosome, chromosome_typed, ChromosomeOptions, WgslChromosome};
pub use infer::{infer_function, FunctionTypes, NodeType, TypeStats, TypedTree};
pub use interp::{Interp, Invocation, Memory, Value};
pub use legality::{Decision, Placement, Reason};
pub use loader::{wgsl_duals, wgsl_fallback_table, wgsl_table, WgslDual, WgslKingdom, WgslRow, WGSL};
pub use reader::{read, Kernel, KernelFunction, Node, Root, RootKind};
pub use scaffold::{rebuild, round_trip, Rebuilt};
pub use table::{FunctionTable, ScalarKind, Template};
pub use versions::split_version;
