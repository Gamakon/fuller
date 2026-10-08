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

pub mod reader;
pub mod table;

pub use reader::{read, Kernel, KernelFunction, Node, Root, RootKind};
pub use table::{FunctionTable, ScalarKind, Template};
