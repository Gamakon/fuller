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

/// Whether the kingdom's private arrays (`parr`) hold arrays of this element type.
pub(crate) fn has_private_array(elem_slot: &str) -> bool {
    matches!(elem_slot, "f32" | "vec4<f32>" | "u32" | "bool" | "vec2<f32>")
}

/// Whether a type is copied as a value: no runtime-sized array, no atomic, in it.
pub(crate) fn is_value(module: &naga::Module, ty: naga::Handle<naga::Type>, depth: usize) -> bool {
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
        let mut found: BTreeMap<String, (Shape, &'static str)> = BTreeMap::new();
        for (h, t) in module.types.iter() {
            let shape = match &t.inner {
                naga::TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } if is_value(module, h, 0) => (Shape::Array { elem: type_text(module, *base), len: n.get() }, "value"),
                naga::TypeInner::Struct { members, .. } if is_value(module, h, 0) && t.name.is_some() => (Shape::Struct { members: members.iter().map(|m| (m.name.clone().unwrap_or_default(), type_text(module, m.ty))).collect() }, "value"),
                naga::TypeInner::Matrix { columns, rows, scalar } if scalar.kind == naga::ScalarKind::Float && matches!(scalar.width, 2 | 4) => (Shape::Matrix { cols: *columns as u32, rows: *rows as u32, scalar: if scalar.width == 2 { "f16" } else { "f32" }.into() }, "real"),
                _ => continue,
            };
            found.insert(type_text(module, h), shape);
        }
        let known: BTreeSet<&str> = self.duals.iter().map(|d| d.slot.as_str()).collect();
        // (The shapes of the types the shipped duals already hold are kept for the rows that write a part of one.)
        let mut all_shapes: BTreeMap<String, Shape> = found.iter().map(|(slot, (shape, _))| (slot.clone(), shape.clone())).collect();
        for (h, t) in module.types.iter() {
            if let naga::TypeInner::Vector { size, scalar } = &t.inner {
                let text = match (scalar.kind, scalar.width) {
                    (naga::ScalarKind::Float, 2) => "f16",
                    (naga::ScalarKind::Float, _) => "f32",
                    (naga::ScalarKind::Sint, _) => "i32",
                    (naga::ScalarKind::Uint, _) => "u32",
                    _ => "bool",
                };
                all_shapes.insert(type_text(module, h), Shape::Vector { lanes: *size as u32, scalar: text.into() });
            }
        }
        // (A matrix's column is a vector the module may not declare.)
        let columns: Vec<(String, Shape)> = all_shapes.values().filter_map(|shape| match shape {
            Shape::Matrix { rows, scalar, .. } => Some((format!("vec{rows}<{scalar}>"), Shape::Vector { lanes: *rows, scalar: scalar.clone() })),
            _ => None,
        }).collect();
        all_shapes.extend(columns);
        found.retain(|slot, _| !known.contains(slot.as_str()));
        // The types the module holds in a local, a private or a returned value that no shipped register kind holds.
        let mut held: BTreeSet<String> = BTreeSet::new();
        for (h, t) in module.types.iter() {
            let value = matches!(&t.inner, naga::TypeInner::Scalar(_) | naga::TypeInner::Vector { .. } | naga::TypeInner::Matrix { .. }) || (matches!(&t.inner, naga::TypeInner::Array { size: naga::ArraySize::Constant(_), .. } | naga::TypeInner::Struct { .. }) && is_value(module, h, 0));
            let slot = type_text(module, h);
            // (An array of an element the private arrays hold stays a private array: its reads and writes are the rows of
            // `parr`, not a register.)
            let private_array = matches!(&t.inner, naga::TypeInner::Array { base, .. } if has_private_array(&type_text(module, *base)));
            let unnamed_struct = matches!(&t.inner, naga::TypeInner::Struct { .. }) && t.name.is_none();
            if value && !private_array && !unnamed_struct && !slot.is_empty() && !crate::wgsl::render::has_register_kind(&slot) {
                held.insert(slot);
            }
        }
        if found.is_empty() && held.is_empty() {
            return self.clone();
        }
        let mut k = self.clone();
        let base = k.duals.len();
        for (i, (slot, (shape, form))) in found.iter().enumerate() {
            let index = u16::try_from(base + i).expect("fewer than 65,536 duals");
            let lanes = if let Shape::Matrix { cols, rows, .. } = shape { cols * rows } else { 0 };
            k.duals.push(WgslDual { index, reference: format!("A{:03}", i + 1), dual: format!("{slot}.{form}"), slot: slot.clone(), form: (*form).into(), lanes, fallback_bits: 0, fallback_noop: false, fallback_elem: false, literal: None });
            k.shapes.insert(index, shape.clone());
        }
        let by_dual: BTreeMap<String, Ty> = k.duals.iter().map(|d| (d.dual.clone(), d.ty())).collect();
        let dual_of = |slot: &str| by_dual.get(&format!("{slot}.{}", default_form(slot))).copied();
        let index_ty = by_dual.get("u32.index").copied();
        let mut rows: Vec<WgslRow> = Vec::new();
        for (i, (shape, _)) in found.values().enumerate() {
            let a = Ty::Wgsl((base + i) as u16);
            rows.push(WgslRow { name: "agg.zero".into(), template: "A-zero".into(), inputs: Vec::new(), output: a, terminal: true });
            match shape {
                Shape::Array { elem, .. } => {
                    if let (Some(e), Some(ix)) = (dual_of(elem), index_ty) {
                        rows.push(WgslRow { name: "agg.at".into(), template: "A-at".into(), inputs: vec![a, ix], output: e, terminal: false });
                    }
                }
                Shape::Matrix { cols, rows: nrows, scalar } => {
                    if let Some(col) = by_dual.get(&format!("vec{nrows}<{scalar}>.real")) {
                        for c in 0..*cols {
                            rows.push(WgslRow { name: format!("shape.col_{c}"), template: "A-col".into(), inputs: vec![a], output: *col, terminal: false });
                        }
                    }
                }
                Shape::Struct { members } => {
                    for (name, mslot) in members {
                        if let Some(m) = dual_of(mslot) {
                            rows.push(WgslRow { name: format!("agg.member_{name}"), template: "A-member".into(), inputs: vec![a], output: m, terminal: false });
                        }
                    }
                }
                // (A vector is not an added dual: only the rows that write a lane of one are.)
                Shape::Vector { .. } => {}
            }
        }
        k.rows.extend(rows);
        k.add_registers(&held);
        k.add_with_rows(all_shapes.iter().map(|(slot, shape)| (slot.as_str(), shape)));
        k
    }

    /// The rows that write a part of an aggregate value, one for each part: `agg.with_at` (an array's or a matrix's
    /// element at an index) and `agg.with_<m>` (a struct's member), the value with that part replaced. A write through a
    /// place of any depth is a chain of them over the reads of the parts on the way.
    fn add_with_rows<'a>(&mut self, shapes: impl Iterator<Item = (&'a str, &'a Shape)>) {
        let by_dual: BTreeMap<String, Ty> = self.duals.iter().map(|d| (d.dual.clone(), d.ty())).collect();
        let Some(index) = by_dual.get("u32.index").copied() else { return };
        let mut rows = Vec::new();
        for (slot, shape) in shapes {
            let Some(&whole) = by_dual.get(&format!("{slot}.{}", default_form(slot))) else { continue };
            let part = |text: &str| by_dual.get(&format!("{text}.{}", default_form(text))).copied();
            match shape {
                Shape::Array { elem, .. } => {
                    if let Some(e) = part(elem) {
                        rows.push(WgslRow { name: "agg.with_at".into(), template: "A-with-at".into(), inputs: vec![whole, index, e], output: whole, terminal: false });
                    }
                }
                Shape::Matrix { rows: nrows, scalar, .. } => {
                    if let Some(col) = part(&format!("vec{nrows}<{scalar}>")) {
                        rows.push(WgslRow { name: "agg.with_at".into(), template: "A-with-at".into(), inputs: vec![whole, index, col], output: whole, terminal: false });
                    }
                }
                Shape::Vector { scalar, .. } => {
                    if let Some(lane) = part(scalar) {
                        rows.push(WgslRow { name: "agg.with_at".into(), template: "A-with-at".into(), inputs: vec![whole, index, lane], output: whole, terminal: false });
                    }
                }
                Shape::Struct { members } => {
                    for (name, mslot) in members {
                        if let Some(m) = part(mslot) {
                            rows.push(WgslRow { name: format!("agg.with_{name}"), template: "A-with-member".into(), inputs: vec![whole, m], output: whole, terminal: false });
                        }
                    }
                }
            }
        }
        self.rows.extend(rows);
    }

    /// A register kind for each of `slots` that has a value dual here (the aggregates', and the shipped matrices and vectors
    /// no register kind holds): the dual `reg<slot>.<form>`, its terminals `reg.<L>0..3`, the read `reg.get_<L>` and a write
    /// at every rung `reg.set_<L>_<rung>`, with a capital letter `L` in order of the slot's text (the render, check and read
    /// paths treat a capital letter as a register kind, as for [`crate::wgsl::render::EXTRA_REGISTERS`]).
    fn add_registers(&mut self, slots: &BTreeSet<String>) {
        const LETTERS: &str = "ABMNOPQRSTUVWXYZ";
        const RUNGS: [&str; 12] = ["L0", "L1", "L2", "L3", "L4", "C0", "C1", "C2", "C3", "C4", "L5", "C5"];
        let mut letters = LETTERS.chars();
        let mut rows: Vec<WgslRow> = Vec::new();
        for slot in slots {
            let want = default_form(slot);
            let Some(value) = self.duals.iter().find(|d| d.slot == *slot && d.form == want).cloned() else { continue };
            let stmts: Option<Vec<Ty>> = RUNGS.iter().map(|r| self.duals.iter().find(|d| d.dual == format!("stmt.{r}")).map(WgslDual::ty)).collect();
            let Some(stmts) = stmts else { continue };
            let Some(letter) = letters.next() else { break };
            let index = u16::try_from(self.duals.len()).expect("fewer than 65,536 duals");
            self.duals.push(WgslDual { index, reference: format!("R{index:03}"), dual: format!("reg<{slot}>.{want}"), slot: format!("reg<{slot}>"), form: want.into(), lanes: value.lanes, fallback_bits: 0, fallback_noop: false, fallback_elem: false, literal: None });
            let reg = Ty::Wgsl(index);
            for i in 0..REGISTER_COUNT {
                rows.push(WgslRow { name: format!("reg.{letter}{i}"), template: "R-reg".into(), inputs: Vec::new(), output: reg, terminal: true });
            }
            rows.push(WgslRow { name: format!("reg.get_{letter}"), template: "R-get".into(), inputs: vec![reg], output: value.ty(), terminal: false });
            for (r, st) in RUNGS.iter().zip(stmts) {
                rows.push(WgslRow { name: format!("reg.set_{letter}_{r}"), template: "R-set".into(), inputs: vec![reg, value.ty()], output: st, terminal: false });
            }
        }
        self.rows.extend(rows);
    }
}

/// How many registers a kernel may name of each kind [`WgslKingdom::extended`] adds.
pub const REGISTER_COUNT: usize = 4;

/// The register kind a module's extended kingdom adds for a value dual: its letter and how many registers there are.
pub fn module_register_of_value(kingdom: &WgslKingdom, dual: &str) -> Option<(char, usize)> {
    let slot = dual.rsplit_once('.').map_or(dual, |(s, _)| s);
    kingdom.rows.iter().find_map(|r| {
        let letter = r.name.strip_prefix("reg.get_").filter(|l| l.len() == 1)?.chars().next().filter(char::is_ascii_uppercase)?;
        let d = kingdom.dual(r.output)?;
        (d.slot == slot && kingdom.dual(*r.inputs.first()?).is_some_and(|i| i.slot.starts_with("reg<")) && !crate::wgsl::render::EXTRA_REGISTERS.iter().any(|x| x.letter == letter)).then_some((letter, REGISTER_COUNT))
    })
}
