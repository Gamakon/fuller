//! Aggregate values as duals (docs/PLAN_wgsl_terminals.md, "Aggregates"): a fixed array
//! `array<T, N>` or a struct `S` the module declares is a value the kernel passes to a helper,
//! indexes and reads members of. Its dual is `<type text>.value` (lanes 0), appended to the
//! shipped duals per module by [`WgslKingdom::extended`], with the rows over its shape:
//! `agg.zero` (the terminal), `agg.at` (an array's element), `agg.member_<m>` (a struct's member).
//! The codes are `Ty::Wgsl(duals.len() + k)` with the types ordered by text, so one module
//! always gets the same codes.

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::Ty;

use super::interface::type_text;
use super::loader::{Shape, WgslDual, WgslKingdom, WgslRow};

/// The default form of a value slot (`f32` is `real`): the dual an aggregate's element or
/// member takes before inference retypes it.
fn default_form(slot: &str) -> &'static str {
    let scalar = match slot.find('<') {
        Some(i) if slot.starts_with("vec") || slot.starts_with("mat") => &slot[i + 1..slot.len() - 1],
        _ => slot,
    };
    match scalar {
        "f32" | "f16" => "real",
        "i32" => "int",
        "u32" => "index",
        "bool" => "flag",
        _ => "value",
    }
}

/// Whether a type is copied as a value: no runtime-sized array, no atomic, in it.
fn is_value(module: &naga::Module, ty: naga::Handle<naga::Type>, depth: usize) -> bool {
    if depth > 8 {
        return false;
    }
    match &module.types[ty].inner {
        naga::TypeInner::Scalar(_) | naga::TypeInner::Vector { .. } | naga::TypeInner::Matrix { .. } => true,
        naga::TypeInner::Array { base, size: naga::ArraySize::Constant(_), .. } => is_value(module, *base, depth + 1),
        naga::TypeInner::Struct { members, .. } => members.iter().all(|m| is_value(module, m.ty, depth + 1)),
        _ => false,
    }
}

impl WgslKingdom {
    /// This kingdom with the aggregate value duals of `module`'s types appended, and their rows.
    pub fn extended(&self, module: &naga::Module) -> WgslKingdom {
        let mut found: BTreeMap<String, Shape> = BTreeMap::new();
        for (h, t) in module.types.iter() {
            let shape = match &t.inner {
                naga::TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } if is_value(module, h, 0) => Shape::Array { elem: type_text(module, *base), len: n.get() },
                naga::TypeInner::Struct { members, .. } if is_value(module, h, 0) && t.name.is_some() => Shape::Struct { members: members.iter().map(|m| (m.name.clone().unwrap_or_default(), type_text(module, m.ty))).collect() },
                _ => continue,
            };
            found.insert(type_text(module, h), shape);
        }
        let known: BTreeSet<&str> = self.duals.iter().map(|d| d.slot.as_str()).collect();
        found.retain(|slot, _| !known.contains(slot.as_str()));
        if found.is_empty() {
            return self.clone();
        }
        let mut k = self.clone();
        let base = k.duals.len();
        for (i, (slot, shape)) in found.iter().enumerate() {
            let index = u16::try_from(base + i).expect("fewer than 65,536 duals");
            k.duals.push(WgslDual { index, reference: format!("A{:03}", i + 1), dual: format!("{slot}.value"), slot: slot.clone(), form: "value".into(), lanes: 0, fallback_bits: 0, fallback_noop: false, fallback_elem: false, literal: None });
            k.shapes.insert(index, shape.clone());
        }
        let by_dual: BTreeMap<String, Ty> = k.duals.iter().map(|d| (d.dual.clone(), d.ty())).collect();
        let dual_of = |slot: &str| by_dual.get(&format!("{slot}.{}", default_form(slot))).copied();
        let index_ty = by_dual.get("u32.index").copied();
        let mut rows: Vec<WgslRow> = Vec::new();
        for (i, shape) in found.values().enumerate() {
            let a = Ty::Wgsl((base + i) as u16);
            rows.push(WgslRow { name: "agg.zero".into(), template: "A-zero".into(), inputs: Vec::new(), output: a, terminal: true });
            match shape {
                Shape::Array { elem, .. } => {
                    if let (Some(e), Some(ix)) = (dual_of(elem), index_ty) {
                        rows.push(WgslRow { name: "agg.at".into(), template: "A-at".into(), inputs: vec![a, ix], output: e, terminal: false });
                    }
                }
                Shape::Struct { members } => {
                    for (name, mslot) in members {
                        if let Some(m) = dual_of(mslot) {
                            rows.push(WgslRow { name: format!("agg.member_{name}"), template: "A-member".into(), inputs: vec![a], output: m, terminal: false });
                        }
                    }
                }
            }
        }
        k.rows.extend(rows);
        k
    }
}
