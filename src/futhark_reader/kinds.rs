//! THE INTEGERS HELD IN F32 (`docs/kmeans_seed.md`, "What is not one-to-one", item 4): a view whose one element is
//! `f32` holds a program's integer values (a count, an assignment, an index as a value) as f32 numbers, exact below
//! 2^24. A program a model writes, or the published k-means step, computes with `i32` values where Futhark's own type
//! system says so (`i32.i64 i`, `hist (+) 0 k ... (replicate n 1)`, `if n == 0 then 1 else n`, `r32 n`), and this
//! module reads those constructs into the f32 kingdom by the exact rewrites of the integers:
//!
//! * `i32.i64 i` is `f32.i64 i`; `f32.i32 x` (`r32 x`) is `x`; `i32.abs`, `i32.min`, `i32.max`, `i32.sum`,
//!   `i32.minimum`, `i32.maximum` are their f32 forms (a sum of integers is exact while it stays below 2^24);
//! * a comparison chooses through the kingdom's one comparison, `e < 0`: `a < b` is `a - b < 0` and `a > b` is
//!   `b - a < 0` for EVERY pair of f32 values (a difference of two distinct floats is never zero, by gradual
//!   underflow; an infinity or a NaN compares false both ways, as `<` does); on INTEGER values only, `a == b` is
//!   `|a - b| - 1 < 0`, `a != b` the same with the branches swapped, `a >= b` is `if a - b < 0` with the branches
//!   swapped and `a <= b` is `if b - a < 0` with the branches swapped (a NaN makes the swapped forms wrong on f32, so
//!   on f32 those four comparisons stay as they are and the reader reports the gap); an `i64` operand (an index
//!   compared as a value) is converted by `f32.i64` first;
//! * the index map of a histogram, `map i64.i32 m` over integers held in f32, is the f32 rows' guarded map
//!   (`\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))`): on an integer in
//!   `[0, k)` both give the integer, on one outside both give an index the histogram ignores.
//!
//! Which values are integers is Futhark's typing, recovered here by a small unification-based kind inference over
//! the normalised expression (`i32`, `i64`, `f32`, `bool`, arrays, tuples, functions; the prelude's functions have
//! their signatures below; an integer literal is a kind variable that Futhark's own default, `i32`, settles when no
//! use does). A construct whose kind the inference cannot settle is left as it is: the reader then reads it as f32 or
//! reports the gap. Nothing here is a substitution: every rewrite is an identity on the values the program computes,
//! under the one stated condition that the program's integers are below 2^24 (16,777,216 points or centres: the
//! bound of the representation decision, `docs/kmeans_seed.md`).

use std::collections::HashMap;

use super::{app, b, var, E};

/// The kind of a value: Futhark's type with the sizes left out.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    I32,
    I64,
    F32,
    Bool,
    Arr(Box<Kind>),
    Tuple(Vec<Kind>),
    Fn(Vec<Kind>, Box<Kind>),
    Var(usize),
}

/// The inference: a union-find of kind variables, and the kind of every node of the expression in pre-order (the
/// rewrite walks the same tree in the same order and reads them back).
struct Infer {
    subst: Vec<Option<Kind>>,
    literals: Vec<usize>,
    trace: Vec<Kind>,
}

impl Infer {
    fn fresh(&mut self) -> Kind {
        self.subst.push(None);
        Kind::Var(self.subst.len() - 1)
    }

    fn find(&self, k: &Kind) -> Kind {
        match k {
            Kind::Var(i) => match &self.subst[*i] {
                Some(k2) => self.find(k2),
                None => k.clone(),
            },
            _ => k.clone(),
        }
    }

    /// The kind with every solved variable replaced, through the structure.
    fn resolve(&self, k: &Kind) -> Kind {
        match self.find(k) {
            Kind::Arr(x) => Kind::Arr(Box::new(self.resolve(&x))),
            Kind::Tuple(xs) => Kind::Tuple(xs.iter().map(|x| self.resolve(x)).collect()),
            Kind::Fn(ps, r) => Kind::Fn(ps.iter().map(|x| self.resolve(x)).collect(), Box::new(self.resolve(&r))),
            other => other,
        }
    }

    fn occurs(&self, i: usize, k: &Kind) -> bool {
        match self.find(k) {
            Kind::Var(j) => i == j,
            Kind::Arr(x) => self.occurs(i, &x),
            Kind::Tuple(xs) => xs.iter().any(|x| self.occurs(i, x)),
            Kind::Fn(ps, r) => ps.iter().any(|x| self.occurs(i, x)) || self.occurs(i, &r),
            _ => false,
        }
    }

    /// Make two kinds one; false when they cannot be (the information is then dropped, never forced).
    fn unify(&mut self, a: &Kind, b: &Kind) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        match (&a, &b) {
            (Kind::Var(i), Kind::Var(j)) if i == j => true,
            (Kind::Var(i), _) => {
                if self.occurs(*i, &b) {
                    return false;
                }
                self.subst[*i] = Some(b);
                true
            }
            (_, Kind::Var(j)) => {
                if self.occurs(*j, &a) {
                    return false;
                }
                self.subst[*j] = Some(a);
                true
            }
            (Kind::Arr(x), Kind::Arr(y)) => self.unify(x, y),
            (Kind::Tuple(xs), Kind::Tuple(ys)) if xs.len() == ys.len() => xs.iter().zip(ys).all(|(x, y)| self.unify(x, y)),
            (Kind::Fn(ps, r), Kind::Fn(qs, s)) if ps.len() == qs.len() => ps.iter().zip(qs).all(|(x, y)| self.unify(x, y)) && self.unify(r, s),
            _ => a == b,
        }
    }

    /// The signature of a prelude function, polymorphic in fresh variables for this use.
    fn prelude(&mut self, name: &str) -> Option<Kind> {
        let f = |ps: Vec<Kind>, r: Kind| Kind::Fn(ps, Box::new(r));
        let arr = |k: Kind| Kind::Arr(Box::new(k));
        let (a, bk, c) = (self.fresh(), self.fresh(), self.fresh());
        Some(match name {
            "map" => f(vec![f(vec![a.clone()], bk.clone()), arr(a)], arr(bk)),
            "map2" => f(vec![f(vec![a.clone(), bk.clone()], c.clone()), arr(a), arr(bk)], arr(c)),
            "map3" => {
                let d = self.fresh();
                f(vec![f(vec![a.clone(), bk.clone(), c.clone()], d.clone()), arr(a), arr(bk), arr(c)], arr(d))
            }
            "tabulate" => f(vec![Kind::I64, f(vec![Kind::I64], a.clone())], arr(a)),
            "replicate" => f(vec![Kind::I64, a.clone()], arr(a)),
            "iota" => f(vec![Kind::I64], arr(Kind::I64)),
            "indices" => f(vec![arr(a)], arr(Kind::I64)),
            "length" => f(vec![arr(a)], Kind::I64),
            "zip" => f(vec![arr(a.clone()), arr(bk.clone())], arr(Kind::Tuple(vec![a, bk]))),
            "reduce" | "reduce_comm" => f(vec![f(vec![a.clone(), a.clone()], a.clone()), a.clone(), arr(a.clone())], a),
            "scan" => f(vec![f(vec![a.clone(), a.clone()], a.clone()), a.clone(), arr(a.clone())], arr(a)),
            "foldl" => f(vec![f(vec![a.clone(), bk.clone()], a.clone()), a.clone(), arr(bk)], a),
            "hist" => f(vec![f(vec![a.clone(), a.clone()], a.clone()), a.clone(), Kind::I64, arr(Kind::I64), arr(a.clone())], arr(a)),
            "reduce_by_index" => f(vec![arr(a.clone()), f(vec![a.clone(), a.clone()], a.clone()), a.clone(), arr(Kind::I64), arr(a.clone())], arr(a)),
            "transpose" => f(vec![arr(arr(a.clone()))], arr(arr(a))),
            "opaque" | "copy" => f(vec![a.clone()], a),
            "f32.sum" | "f32.minimum" | "f32.maximum" | "f32.product" => f(vec![arr(Kind::F32)], Kind::F32),
            "i32.sum" | "i32.minimum" | "i32.maximum" | "i32.product" => f(vec![arr(Kind::I32)], Kind::I32),
            "i64.sum" | "i64.minimum" | "i64.maximum" | "i64.product" => f(vec![arr(Kind::I64)], Kind::I64),
            "f32.sqrt" | "f32.abs" | "f32.neg" | "f32.exp" | "f32.log" | "f32.floor" | "f32.ceil" | "f32.round" | "f32.trunc" => f(vec![Kind::F32], Kind::F32),
            "f32.min" | "f32.max" => f(vec![Kind::F32, Kind::F32], Kind::F32),
            "i32.abs" | "i32.neg" => f(vec![Kind::I32], Kind::I32),
            "i32.min" | "i32.max" => f(vec![Kind::I32, Kind::I32], Kind::I32),
            "i64.abs" | "i64.neg" => f(vec![Kind::I64], Kind::I64),
            "i64.min" | "i64.max" => f(vec![Kind::I64, Kind::I64], Kind::I64),
            "f32.i64" => f(vec![Kind::I64], Kind::F32),
            "f32.i32" | "r32" => f(vec![Kind::I32], Kind::F32),
            "f32.f32" => f(vec![Kind::F32], Kind::F32),
            "i32.i64" => f(vec![Kind::I64], Kind::I32),
            "i32.i32" => f(vec![Kind::I32], Kind::I32),
            "i32.f32" | "t32" => f(vec![Kind::F32], Kind::I32),
            "i64.i32" => f(vec![Kind::I32], Kind::I64),
            "i64.i64" => f(vec![Kind::I64], Kind::I64),
            "i64.f32" => f(vec![Kind::F32], Kind::I64),
            "f32.isnan" | "f32.isinf" => f(vec![Kind::F32], Kind::Bool),
            "i32.highest" | "i32.lowest" => Kind::I32,
            "i64.highest" | "i64.lowest" => Kind::I64,
            _ => return None,
        })
    }

    /// Apply a function kind to arguments' kinds: the result, a partial application, or nothing known.
    fn apply(&mut self, f: Kind, args: &[Kind]) -> Kind {
        match self.find(&f) {
            Kind::Fn(ps, r) => {
                let n = ps.len().min(args.len());
                for (p, a) in ps.iter().zip(args) {
                    self.unify(p, a);
                }
                if args.len() < ps.len() {
                    Kind::Fn(ps[n..].to_vec(), r)
                } else if args.len() == ps.len() {
                    *r
                } else {
                    self.apply(*r, &args[n..])
                }
            }
            _ => self.fresh(),
        }
    }

    /// The kind of an expression; every node's kind is recorded in pre-order.
    fn infer(&mut self, e: &E, env: &HashMap<String, Kind>) -> Kind {
        let at = self.trace.len();
        self.trace.push(Kind::Bool);
        let k = match e {
            E::Int(_) => {
                let v = self.fresh();
                if let Kind::Var(i) = v {
                    self.literals.push(i);
                }
                v
            }
            E::F32(_) => Kind::F32,
            E::Hole(_) => self.fresh(),
            E::Var(x) => match env.get(x) {
                Some(k) => k.clone(),
                None => match self.prelude(x) {
                    Some(k) => k,
                    None => self.fresh(),
                },
            },
            E::App(f, args) => {
                let fk = self.infer(f, env);
                let aks: Vec<Kind> = args.iter().map(|a| self.infer(a, env)).collect();
                self.apply(fk, &aks)
            }
            E::Bin(op, l, r) => {
                let (lk, rk) = (self.infer(l, env), self.infer(r, env));
                self.unify(&lk, &rk);
                match op.as_str() {
                    "<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||" => Kind::Bool,
                    _ => lk,
                }
            }
            E::Neg(x) => self.infer(x, env),
            E::Index(a, i) => {
                let ak = self.infer(a, env);
                let ik = self.infer(i, env);
                self.unify(&ik, &Kind::I64);
                let elem = self.fresh();
                self.unify(&ak, &Kind::Arr(Box::new(elem.clone())));
                elem
            }
            E::If(c, t, f) => {
                let ck = self.infer(c, env);
                self.unify(&ck, &Kind::Bool);
                let tk = self.infer(t, env);
                let fk = self.infer(f, env);
                self.unify(&tk, &fk);
                tk
            }
            E::Lam(ps, body) => {
                let mut env2 = env.clone();
                let pks: Vec<Kind> = ps.iter().map(|p| {
                    let k = self.fresh();
                    env2.insert(p.clone(), k.clone());
                    k
                }).collect();
                let bk = self.infer(body, &env2);
                Kind::Fn(pks, Box::new(bk))
            }
            E::Let(x, v, body) => {
                let vk = self.infer(v, env);
                let mut env2 = env.clone();
                env2.insert(x.clone(), vk);
                self.infer(body, &env2)
            }
            E::Tuple(xs) => Kind::Tuple(xs.iter().map(|x| self.infer(x, env)).collect()),
            E::Proj(x, i) => {
                let xk = self.infer(x, env);
                match self.find(&xk) {
                    Kind::Tuple(ks) if *i < ks.len() => ks[*i].clone(),
                    _ => self.fresh(),
                }
            }
            E::Loop { states, init, var: v, lo, hi, body } => {
                let ik = self.infer(init, env);
                let lk = self.infer(lo, env);
                self.unify(&lk, &Kind::I64);
                let hk = self.infer(hi, env);
                self.unify(&hk, &Kind::I64);
                let mut env2 = env.clone();
                env2.insert(v.clone(), Kind::I64);
                if states.len() == 1 {
                    env2.insert(states[0].clone(), ik.clone());
                } else {
                    let parts: Vec<Kind> = states.iter().map(|_| self.fresh()).collect();
                    self.unify(&ik, &Kind::Tuple(parts.clone()));
                    for (s, k) in states.iter().zip(parts) {
                        env2.insert(s.clone(), k);
                    }
                }
                let bk = self.infer(body, &env2);
                self.unify(&ik, &bk);
                ik
            }
        };
        self.trace[at] = k.clone();
        k
    }
}

/// The rewrite of the integers into the f32 kingdom (the module's header), over an expression whose free names have
/// the kinds `free` gives (the inputs, the dimensions, the size inputs, the zero arrays).
pub fn hold_integers(e: &E, free: &HashMap<String, Kind>) -> E {
    let mut inf = Infer { subst: Vec::new(), literals: Vec::new(), trace: Vec::new() };
    inf.infer(e, free);
    // Futhark's default for an integer literal no use settles.
    for i in inf.literals.clone() {
        if matches!(inf.find(&Kind::Var(i)), Kind::Var(_)) {
            inf.unify(&Kind::Var(i), &Kind::I32);
        }
    }
    let mut pos = 0;
    rewrite(e, &inf, &mut pos)
}

fn is_int(k: &Kind) -> bool {
    matches!(k, Kind::I32 | Kind::I64)
}

fn is_zero(e: &E) -> bool {
    matches!(e, E::Int(0)) || matches!(e, E::F32(bits) if *bits == 0)
}

/// `a` as an f32 value: an `i64` is converted, an `i32` is held as it is.
fn as_f32(a: E, k: &Kind) -> E {
    if *k == Kind::I64 {
        app("f32.i64", vec![a])
    } else {
        a
    }
}

/// The guarded index map of the f32 histogram rows, for the dimension `d`.
fn guard(d: &E) -> E {
    let x = "%g".to_string();
    E::Lam(vec![x.clone()], b(E::If(b(app("f32.isnan", vec![var(&x)])), b(E::Int(-1)), b(app("i64.f32", vec![app("f32.max", vec![E::F32((-1f32).to_bits()), app("f32.min", vec![app("f32.i64", vec![d.clone()]), var(&x)])])])))))
}

/// `map i64.i32 m` (or `map (\x -> i64.i32 (g x)) m`) as the guarded map over the integers held in f32.
fn guarded_index_map(idx: E, d: &E) -> E {
    let E::App(f, args) = &idx else { return idx };
    if **f != var("map") || args.len() != 2 {
        return idx;
    }
    let m = args[1].clone();
    if args[0] == var("i64.i32") || args[0].eta() == Some("i64.i32") {
        return app("map", vec![guard(d), m]);
    }
    if let E::Lam(ps, body) = &args[0] {
        if let (1, E::App(g, ga)) = (ps.len(), body.as_ref()) {
            if **g == var("i64.i32") && ga.len() == 1 {
                return app("map", vec![guard(d), app("map", vec![E::Lam(ps.clone(), b(ga[0].clone())), m])]);
            }
        }
    }
    idx
}

fn rewrite(e: &E, inf: &Infer, pos: &mut usize) -> E {
    *pos += 1;
    let r = |x: &E, pos: &mut usize| rewrite(x, inf, pos);
    match e {
        E::Int(_) | E::F32(_) | E::Hole(_) | E::Var(_) => e.clone(),
        E::App(f, args) => {
            let f2 = r(f, pos);
            let a2: Vec<E> = args.iter().map(|a| r(a, pos)).collect();
            let name = match &f2 {
                E::Var(n) => n.as_str(),
                _ => "",
            };
            match (name, a2.as_slice()) {
                ("i32.i64", [x]) => app("f32.i64", vec![x.clone()]),
                ("f32.i32" | "r32", [x]) => x.clone(),
                ("i32.abs", [x]) => app("f32.abs", vec![x.clone()]),
                ("i32.min", [x, y]) => app("f32.min", vec![x.clone(), y.clone()]),
                ("i32.max", [x, y]) => app("f32.max", vec![x.clone(), y.clone()]),
                ("i32.sum", [x]) => app("f32.sum", vec![x.clone()]),
                ("i32.minimum", [x]) => app("f32.minimum", vec![x.clone()]),
                ("i32.maximum", [x]) => app("f32.maximum", vec![x.clone()]),
                // The index map is `i64.i32` of integers held in f32 exactly when Futhark typed them `i32`.
                ("hist", [op, ne, d, idx, vs]) => app("hist", vec![op.clone(), ne.clone(), d.clone(), guarded_index_map(idx.clone(), d), vs.clone()]),
                ("reduce_by_index", [dst, op, ne, idx, vs]) => {
                    let d = match dst {
                        E::App(g, ga) if **g == var("replicate") && ga.len() == 2 => ga[0].clone(),
                        _ => return E::App(b(f2.clone()), a2.clone()),
                    };
                    app("reduce_by_index", vec![dst.clone(), op.clone(), ne.clone(), guarded_index_map(idx.clone(), &d), vs.clone()])
                }
                _ => E::App(b(f2), a2),
            }
        }
        E::Bin(op, l, rr) => {
            let (l2, r2) = (r(l, pos), r(rr, pos));
            E::Bin(op.clone(), b(l2), b(r2))
        }
        E::Neg(x) => E::Neg(b(r(x, pos))),
        E::Index(a, i) => {
            let a2 = r(a, pos);
            E::Index(b(a2), b(r(i, pos)))
        }
        E::If(c, t, f) => {
            // The comparison's operands' kinds, read ahead (the condition's node, then its left operand's).
            let operand_kinds = match c.as_ref() {
                E::Bin(_, l, _) => {
                    let p = *pos + 1;
                    let lk = inf.resolve(&inf.trace[p]);
                    let rk = inf.resolve(&inf.trace[p + count(l)]);
                    Some((lk, rk))
                }
                _ => None,
            };
            let c2 = r(c, pos);
            let t2 = r(t, pos);
            let f2 = r(f, pos);
            let (E::Bin(op, a, bb), Some((ka, kb))) = (&c2, operand_kinds) else { return E::If(b(c2), b(t2), b(f2)) };
            let ints = is_int(&ka) || is_int(&kb);
            let (a, bb) = (as_f32(*a.clone(), &ka), as_f32(*bb.clone(), &kb));
            let lt0 = |x: E| E::Bin("<".into(), b(x), b(E::Int(0)));
            let sub = |x: E, y: E| E::Bin("-".into(), b(x), b(y));
            let abs_diff = |x: E, y: E| if is_zero(&y) { app("f32.abs", vec![x]) } else if is_zero(&x) { app("f32.abs", vec![y]) } else { app("f32.abs", vec![sub(x, y)]) };
            match op.as_str() {
                "<" if is_zero(&bb) => E::If(b(lt0(a)), b(t2), b(f2)),
                "<" => E::If(b(lt0(sub(a, bb))), b(t2), b(f2)),
                ">" if is_zero(&a) => E::If(b(lt0(bb)), b(t2), b(f2)),
                ">" => E::If(b(lt0(sub(bb, a))), b(t2), b(f2)),
                ">=" if ints => E::If(b(lt0(sub(a, bb))), b(f2), b(t2)),
                "<=" if ints => E::If(b(lt0(sub(bb, a))), b(f2), b(t2)),
                "==" if ints => E::If(b(lt0(sub(abs_diff(a, bb), E::Int(1)))), b(t2), b(f2)),
                "!=" if ints => E::If(b(lt0(sub(abs_diff(a, bb), E::Int(1)))), b(f2), b(t2)),
                _ => E::If(b(c2), b(t2), b(f2)),
            }
        }
        E::Lam(ps, body) => E::Lam(ps.clone(), b(r(body, pos))),
        E::Let(x, v, body) => {
            let v2 = r(v, pos);
            E::Let(x.clone(), b(v2), b(r(body, pos)))
        }
        E::Tuple(xs) => E::Tuple(xs.iter().map(|x| r(x, pos)).collect()),
        E::Proj(x, i) => E::Proj(b(r(x, pos)), *i),
        E::Loop { states, init, var: v, lo, hi, body } => {
            let init2 = r(init, pos);
            let lo2 = r(lo, pos);
            let hi2 = r(hi, pos);
            E::Loop { states: states.clone(), init: b(init2), var: v.clone(), lo: b(lo2), hi: b(hi2), body: b(r(body, pos)) }
        }
    }
}

/// The number of nodes of an expression in the inference's pre-order (one a node).
fn count(e: &E) -> usize {
    1 + match e {
        E::Int(_) | E::F32(_) | E::Hole(_) | E::Var(_) => 0,
        E::App(f, args) => count(f) + args.iter().map(count).sum::<usize>(),
        E::Bin(_, l, r) | E::Index(l, r) => count(l) + count(r),
        E::Neg(x) | E::Lam(_, x) | E::Proj(x, _) => count(x),
        E::If(c, t, f) => count(c) + count(t) + count(f),
        E::Let(_, v, body) => count(v) + count(body),
        E::Tuple(xs) => xs.iter().map(count).sum(),
        E::Loop { init, lo, hi, body, .. } => count(init) + count(lo) + count(hi) + count(body),
    }
}
