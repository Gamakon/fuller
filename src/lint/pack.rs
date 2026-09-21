//! Rule rows in the fixed-width form the device reads (spec §4).
//!
//! A pattern and a template are each a 15-slot binary heap (children of slot
//! `i` at `2i+1`, `2i+2`; a unary op uses `2i+1`), which holds any tree of
//! depth <= 3. The kernel compares integers only: an exact literal in a
//! pattern is a LITERAL CLASS id, and a predicate on a bound literal is a
//! PREDICATE BIT — both computed on the host, in f64, per node.
//!
//! The binary heap assumes arity <= 2 (`HEAP_ARITY`). That is a Symbolic
//! Regression seam, named here; the CPU `Pat` has no such limit.

use super::tables::{Cmp, Exactness, GuardRule, NumExpr, Order, Pat, Rule, Tmpl};
use crate::gpu_eval::Op;

pub const HEAP_ARITY: usize = 2;
pub const HEAP_SLOTS: usize = 15;
pub const MAX_METAVARS: usize = 8;

/// Slot kinds. Shared by patterns and templates.
pub const KIND_UNUSED: u32 = 0;
pub const KIND_OP: u32 = 1;
pub const KIND_MV: u32 = 2;
pub const KIND_NUM: u32 = 3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Slot {
    pub kind: u32,
    /// op: the opcode. mv: the metavariable index. num (pattern): the numeric
    /// metavariable index + 1, or 0 for an exact literal. num (template): an
    /// index into the literal pool.
    pub id: u32,
    /// mv (pattern): fact bits the binding must carry. num (pattern):
    /// predicate bits the literal must carry.
    pub req: u32,
    /// num (pattern, exact): the literal class id the node must carry.
    pub lit: u32,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct PackedRule {
    pub rule_id: u32,
    pub root_kind: u32,
    pub root_op: u32,
    pub n_mv: u32,
    /// 0 = A (shrinks), 1 = B (enables).
    pub order: u32,
    /// 0 = bit, 1 = rounding, 2 = finite.
    pub exactness: u32,
    pub pat: [Slot; HEAP_SLOTS],
    pub tmpl: [Slot; HEAP_SLOTS],
}

/// What the host must compute for every literal node, so the kernel never
/// compares floats.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LiteralCodes {
    /// Exact literals that occur in patterns; a node's class id is its index
    /// here + 1, or 0 for "none of them".
    pub classes: Vec<f64>,
    /// Predicates that occur on bound literals; bit `i` of a node's predicate
    /// word is `preds[i]` holding for its value.
    pub preds: Vec<(Cmp, f64)>,
    /// Literals that templates write.
    pub pool: Vec<f64>,
}

impl LiteralCodes {
    pub fn class_of(&self, v: f64) -> u32 {
        self.classes.iter().position(|c| *c == v).map_or(0, |i| i as u32 + 1)
    }

    pub fn pred_bits(&self, v: f64) -> u32 {
        self.preds
            .iter()
            .enumerate()
            .filter(|(_, (cmp, k))| cmp.holds(v, *k))
            .fold(0u32, |bits, (i, _)| bits | (1 << i))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Packed {
    pub rules: Vec<PackedRule>,
    pub codes: LiteralCodes,
    /// Rules the device cannot take, with the reason. Never dropped silently.
    pub excluded: Vec<(usize, String)>,
}

fn intern_f64(list: &mut Vec<f64>, v: f64) -> u32 {
    match list.iter().position(|x| x.to_bits() == v.to_bits()) {
        Some(i) => i as u32,
        None => {
            list.push(v);
            (list.len() - 1) as u32
        }
    }
}

fn intern_pred(list: &mut Vec<(Cmp, f64)>, cmp: Cmp, k: f64) -> Result<u32, String> {
    let i = match list.iter().position(|(c, x)| *c == cmp && x.to_bits() == k.to_bits()) {
        Some(i) => i,
        None => {
            list.push((cmp, k));
            list.len() - 1
        }
    };
    if i >= 24 {
        return Err("more than 24 distinct literal predicates: the kernel's predicate word is 24 bits".to_string());
    }
    Ok(i as u32)
}

/// Compile the admitted rules for the device. A rule that does not fit is
/// listed in `excluded`; it still runs on the CPU engine.
pub fn pack(rules: &[&Rule]) -> Packed {
    let mut out = Packed::default();
    for rule in rules {
        match pack_one(rule, &mut out.codes) {
            Ok(p) => out.rules.push(p),
            Err(why) => out.excluded.push((rule.id, why)),
        }
    }
    out
}

fn pack_one(rule: &Rule, codes: &mut LiteralCodes) -> Result<PackedRule, String> {
    if rule.computes_literal() {
        return Err("template computes a literal: its class would have to be decided in f32".to_string());
    }
    if rule.n_mv as usize > MAX_METAVARS {
        return Err(format!("more than {MAX_METAVARS} metavariables"));
    }
    let mut pat = [Slot::default(); HEAP_SLOTS];
    let mut tmpl = [Slot::default(); HEAP_SLOTS];
    let mut seen_num = Vec::new();
    place_pat(&rule.lhs, 0, rule, codes, &mut seen_num, &mut pat)?;
    place_tmpl(&rule.rhs, 0, codes, &mut tmpl)?;
    Ok(PackedRule {
        rule_id: rule.id as u32,
        root_kind: pat[0].kind,
        root_op: pat[0].id,
        n_mv: rule.n_mv as u32,
        order: u32::from(rule.order == Order::B),
        exactness: match rule.exactness {
            Exactness::Bit => 0,
            Exactness::Rounding => 1,
            Exactness::Finite => 2,
        },
        pat,
        tmpl,
    })
}

fn child_slot(slot: usize, child: usize) -> Result<usize, String> {
    let at = HEAP_ARITY * slot + 1 + child;
    if child >= HEAP_ARITY || at >= HEAP_SLOTS {
        return Err("pattern or template does not fit the 15-slot heap".to_string());
    }
    Ok(at)
}

fn place_pat(
    p: &Pat,
    slot: usize,
    rule: &Rule,
    codes: &mut LiteralCodes,
    seen_num: &mut Vec<u8>,
    heap: &mut [Slot; HEAP_SLOTS],
) -> Result<(), String> {
    match p {
        Pat::Op(op, kids) => {
            heap[slot] = Slot { kind: KIND_OP, id: *op as u32, req: 0, lit: 0 };
            for (i, k) in kids.iter().enumerate() {
                place_pat(k, child_slot(slot, i)?, rule, codes, seen_num, heap)?;
            }
        }
        Pat::Mv(mv) => {
            let req = rule
                .when
                .iter()
                .filter(|(m, _)| m == mv)
                .fold(0u32, |bits, (_, f)| bits | u32::from(f.0));
            heap[slot] = Slot { kind: KIND_MV, id: u32::from(*mv), req, lit: 0 };
        }
        Pat::NumExact(v) => {
            let class = intern_f64(&mut codes.classes, *v) + 1;
            heap[slot] = Slot { kind: KIND_NUM, id: 0, req: 0, lit: class };
        }
        Pat::NumMv(num) => {
            if seen_num.contains(num) {
                return Err("a bound literal appears twice in the pattern".to_string());
            }
            seen_num.push(*num);
            let mut req = 0u32;
            for p in rule.preds.iter().filter(|p| p.num == *num) {
                req |= 1 << intern_pred(&mut codes.preds, p.cmp, p.k)?;
            }
            heap[slot] = Slot { kind: KIND_NUM, id: u32::from(*num) + 1, req, lit: 0 };
        }
    }
    Ok(())
}

fn place_tmpl(t: &Tmpl, slot: usize, codes: &mut LiteralCodes, heap: &mut [Slot; HEAP_SLOTS]) -> Result<(), String> {
    match t {
        Tmpl::Op(op, kids) => {
            heap[slot] = Slot { kind: KIND_OP, id: *op as u32, req: 0, lit: 0 };
            for (i, k) in kids.iter().enumerate() {
                place_tmpl(k, child_slot(slot, i)?, codes, heap)?;
            }
        }
        Tmpl::Mv(mv) => heap[slot] = Slot { kind: KIND_MV, id: u32::from(*mv), req: 0, lit: 0 },
        Tmpl::Num(NumExpr::Const(v)) => {
            heap[slot] = Slot { kind: KIND_NUM, id: intern_f64(&mut codes.pool, *v), req: 0, lit: 0 };
        }
        Tmpl::Num(_) => return Err("template computes a literal".to_string()),
    }
    Ok(())
}

pub const GUARD_STRIDE: usize = 8;
pub const RULE_STRIDE: usize = 6 + 2 * HEAP_SLOTS * 4;
const NO_OP: u32 = u32::MAX;

impl PackedRule {
    /// The row as the kernel reads it: 6 header words, then the pattern heap,
    /// then the template heap, 4 words a slot.
    pub fn words(&self) -> Vec<u32> {
        let mut w = vec![self.rule_id, self.root_kind, self.root_op, self.n_mv, self.order, self.exactness];
        for s in self.pat.iter().chain(self.tmpl.iter()) {
            w.extend_from_slice(&[s.kind, s.id, s.req, s.lit]);
        }
        w
    }

    /// The opcode bucket a rule is filed under: its pattern's root.
    pub fn bucket(&self) -> u32 {
        if self.root_kind == KIND_NUM {
            Op::Num as u32
        } else {
            self.root_op
        }
    }
}

/// Guard rows for the kernel: `[op | NONE, child0 facts, child1 facts, self
/// facts, gives, range_only, literal predicate bits, 0]`. Literal-seed
/// predicates are interned into `codes`, so this must run BEFORE any node's
/// predicate word is computed.
pub fn pack_guards(guards: &[GuardRule], codes: &mut LiteralCodes) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(guards.len() * GUARD_STRIDE);
    for g in guards {
        let mut pred_bits = 0u32;
        for (cmp, k) in &g.num_preds {
            pred_bits |= 1 << intern_pred(&mut codes.preds, *cmp, *k)?;
        }
        let req = |i: usize| g.child_req.get(i).map_or(0, |f| u32::from(f.0));
        out.extend_from_slice(&[
            g.op.map_or(NO_OP, |o| o as u32),
            req(0),
            req(1),
            u32::from(g.self_req.0),
            u32::from(g.gives.0),
            u32::from(g.range_only),
            pred_bits,
            0,
        ]);
    }
    Ok(out)
}

/// The arity the kernel uses for an opcode. `Var` and `Num` are leaves.
pub fn arity(op: u32) -> usize {
    [
        Op::Var, Op::Num, Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Neg, Op::Abs, Op::Sqrt, Op::Log,
        Op::Exp, Op::Sin, Op::Cos, Op::Tan, Op::Tanh, Op::Pow, Op::Pow2, Op::Pow3, Op::Inv,
        Op::ProtectedDiv, Op::ProtectedSqrt, Op::ProtectedLog, Op::ProtectedExp, Op::ProtectedInv,
        Op::Asin, Op::Acos, Op::ProtectedAsin, Op::ProtectedAcos,
    ]
    .get(op as usize)
    .map_or(0, |o| o.arity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::tables::Tables;

    #[test]
    fn every_admitted_rule_is_packed_or_excluded_with_a_reason() {
        let tables = Tables::standard().unwrap();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let packed = pack(&rules);
        assert_eq!(packed.rules.len() + packed.excluded.len(), rules.len());
        assert!(!packed.rules.is_empty());
        for (id, why) in &packed.excluded {
            assert!(!why.is_empty(), "rule {id} excluded without a reason");
            // Today's reasons: the template computes a literal, or the pattern
            // names one bound literal twice (`c * (x / c)`).
            assert!(
                tables.rules[*id].computes_literal() || why.contains("appears twice"),
                "rule {id}: {why}"
            );
        }
    }

    #[test]
    fn literal_classes_and_predicates_are_integers_for_the_kernel() {
        let tables = Tables::standard().unwrap();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let codes = pack(&rules).codes;
        assert_eq!(codes.class_of(1.0), codes.class_of(1.0));
        assert_ne!(codes.class_of(1.0), 0, "1.0 occurs in patterns");
        assert_eq!(codes.class_of(1.0000001), 0, "close to 1 is not 1");
        // |k| < 1e-6 is two predicates; 1e-7 satisfies both, 1e-5 only one.
        assert_ne!(codes.pred_bits(1e-7), codes.pred_bits(1e-5));
    }

    #[test]
    fn a_nonlinear_pattern_repeats_the_metavariable_index() {
        let tables = Tables::standard().unwrap();
        let rule = tables.rules.iter().find(|r| r.text.starts_with("(rewrite (Sub x x)")).unwrap();
        let p = pack(&[rule]).rules.remove(0);
        assert_eq!((p.pat[1].kind, p.pat[2].kind), (KIND_MV, KIND_MV));
        assert_eq!(p.pat[1].id, p.pat[2].id);
    }
}
