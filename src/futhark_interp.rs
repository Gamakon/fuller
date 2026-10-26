//! The Futhark-style kingdom's interpreter: what a gene MEANS, independent of
//! Futhark.
//!
//! The renderer ([`crate::futhark_kingdom::FutharkKingdom::render`]) turns an
//! expressed tree into Futhark source; Futhark compiles and runs that. This
//! module turns the same tree into a value. Putting the two side by side is
//! the check that the kingdom means what it says: the Futhark program and the
//! interpreter must agree, element for element, on a fixed input.
//!
//! It is also what a fast fitness screen would be built on: no compiler, no
//! process, no GPU — just the tree.
//!
//! The semantics are Futhark's:
//!
//! * `map f xs` — `f` applied to each element;
//! * `map2 f xs ys` — `f` applied to each pair, the shorter array deciding
//!   the length;
//! * `scan f ne xs` — the INCLUSIVE prefix scan: `out[0] = f ne xs[0]`,
//!   `out[i] = f out[i-1] xs[i]`. The kingdom's scan rows fix `ne` at the
//!   operator's identity, so `scan (+) 0.0` and `scan (*) 1.0`;
//! * `reduce f ne xs` — `ne` folded with `f` over the elements. Again `ne`
//!   is the operator's identity, so `reduce (+) 0.0` and `reduce (*) 1.0`;
//! * `reverse xs` — the elements in the other order.
//!
//! The identity neutrals are not a convenience. Futhark requires the neutral
//! element of a `reduce` or a `scan` to be a *true* neutral element for the
//! operator, and states that otherwise the result is undefined and the
//! compiler cannot detect it. Measured on Futhark 0.28.1, a free neutral
//! makes `reduce (+) 0.0 (scan (+) 0.5 xs)` return 27.0 where 22.0 is the
//! answer, and poisons the sibling `last (scan (+) 0.5 xs)` as well. With
//! the identities every size from 1 to 9 is exact.
//!
//! One primitive needs care: Futhark's `f32.round` rounds a half to the
//! nearest EVEN, like `rintf` and like WGSL's `round`, while Rust's
//! `f32::round` rounds a half away from zero. See [`Unary::Round`].

use crate::futhark_kingdom::FutharkKingdom;
use crate::typed_gep::Tree;

/// A value of the kingdom.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `f32.real`.
    Real(f32),
    /// `arr.f32` or `scn.f32`.
    Arr(Vec<f32>),
    /// `fn.f32_f32`.
    Fun(Unary),
}

/// An elementary function of a real to a real.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unary {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Sqr,
    Cube,
    Recip,
    Floor,
    Round,
    /// `\x -> x + c`
    AddC(f32),
    /// `\x -> x * c`
    MulC(f32),
    /// `\x -> x - c`
    SubC(f32),
    /// `\x -> 0.0`
    Const,
}

impl Unary {
    /// Apply the function.
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Unary::Neg => -x,
            Unary::Abs => x.abs(),
            Unary::Sqrt => x.sqrt(),
            Unary::Exp => x.exp(),
            Unary::Log => x.ln(),
            Unary::Sin => x.sin(),
            Unary::Cos => x.cos(),
            Unary::Tan => x.tan(),
            Unary::Sqr => x * x,
            Unary::Cube => x * x * x,
            Unary::Recip => 1.0 / x,
            Unary::Floor => x.floor(),
            // Futhark's `f32.round` rounds a half to the NEAREST EVEN
            // (it is `rintf`, and WGSL's `round` is ties-to-even too), while
            // Rust's `f32::round` rounds a half away from zero. On
            // `round(2.5)` the two differ by a whole unit, and the difference
            // propagates: `map2 (*) (map f32.round xs) xs` then `f32.tan`
            // turns a one-unit difference into -8.45 against 6.77.
            Unary::Round => x.round_ties_even(),
            Unary::AddC(c) => x + c,
            Unary::MulC(c) => x * c,
            Unary::SubC(c) => x - c,
            Unary::Const => 0.0,
        }
    }
}

impl FutharkKingdom {
    /// The value of an expressed tree over `input`.
    ///
    /// The tree's root decides the shape of the result: an array for the two
    /// array roots, a real for the reduction root.
    pub fn eval(&self, tree: &Tree, input: &[f32]) -> Result<Value, String> {
        self.eval_at(tree, input)
    }

    fn eval_at(&self, tree: &Tree, input: &[f32]) -> Result<Value, String> {
        let row = self
            .rows
            .get(tree.symbol as usize)
            .ok_or_else(|| format!("futhark_true: symbol {} is not a row of this kingdom", tree.symbol))?;
        let kids: Vec<Value> =
            tree.kids.iter().map(|k| self.eval_at(k, input)).collect::<Result<Vec<_>, _>>()?;
        let name = row.name.as_str();

        // The constant placeholder: its value is the literal its
        // constant-array index names.
        if row.kind == crate::futhark_kingdom::FutharkKind::Const {
            return Ok(Value::Real(self.literal(tree)?));
        }

        // The elementary functions: a value of type `fn.f32_f32`.
        if let Some(u) = self.unary(name, tree, input)? {
            return Ok(Value::Fun(u));
        }

        match name {
            // The input array.
            "arr.f32.input" => Ok(Value::Arr(input.to_vec())),
            // The identity scan of the input: `map (\x -> x) (scan (+) 0.0 xs)`.
            "scn.f32.identity" => {
                let mut out = Vec::with_capacity(input.len());
                let mut acc = 0.0f32;
                for x in input {
                    acc += x;
                    out.push(acc);
                }
                Ok(Value::Arr(out))
            }
            "map_a" | "map_s" => {
                let f = as_fun(&kids[0])?;
                let xs = as_arr(&kids[1])?;
                Ok(Value::Arr(xs.into_iter().map(|x| f.apply(x)).collect()))
            }
            "map2_add_a" | "map2_add_s" => zip(&kids[0], &kids[1], |a, b| a + b),
            "map2_mul_a" | "map2_mul_s" => zip(&kids[0], &kids[1], |a, b| a * b),
            "map2_max_a" | "map2_max_s" => zip(&kids[0], &kids[1], |a, b| a.max(b)),
            "map2_min_a" | "map2_min_s" => zip(&kids[0], &kids[1], |a, b| a.min(b)),
            "rev_a" | "rev_s" => {
                let mut xs = as_arr(&kids[0])?;
                xs.reverse();
                Ok(Value::Arr(xs))
            }
            "scan_add_s" => {
                let xs = as_arr(&kids[0])?;
                let mut out = Vec::with_capacity(xs.len());
                let mut acc = 0.0f32;
                for x in xs {
                    acc += x;
                    out.push(acc);
                }
                Ok(Value::Arr(out))
            }
            "scan_mul_s" => {
                let xs = as_arr(&kids[0])?;
                let mut out = Vec::with_capacity(xs.len());
                let mut acc = 1.0f32;
                for x in xs {
                    acc *= x;
                    out.push(acc);
                }
                Ok(Value::Arr(out))
            }
            "reduce_add_a" | "reduce_add_s" => {
                let xs = as_arr(&kids[0])?;
                Ok(Value::Real(xs.iter().fold(0.0f32, |a, x| a + x)))
            }
            "reduce_mul_a" | "reduce_mul_s" => {
                let xs = as_arr(&kids[0])?;
                Ok(Value::Real(xs.iter().fold(1.0f32, |a, x| a * x)))
            }
            other => Err(format!("futhark_true: the interpreter has no rule for {other}")),
        }
    }

    /// The elementary function a row names, `None` for a row that is not one.
    ///
    /// The captured literal of `add_c`/`mul_c`/`sub_c` is a value of type
    /// `f32.real`, and that value may be a REDUCTION over the input -- so the
    /// input is needed here, not just the tree.
    fn unary(&self, name: &str, tree: &Tree, input: &[f32]) -> Result<Option<Unary>, String> {
        let u = match name {
            "fn.f32_f32.neg" => Unary::Neg,
            "fn.f32_f32.abs" => Unary::Abs,
            "fn.f32_f32.sqrt" => Unary::Sqrt,
            "fn.f32_f32.exp" => Unary::Exp,
            "fn.f32_f32.log" => Unary::Log,
            "fn.f32_f32.sin" => Unary::Sin,
            "fn.f32_f32.cos" => Unary::Cos,
            "fn.f32_f32.tan" => Unary::Tan,
            "fn.f32_f32.sqr" => Unary::Sqr,
            "fn.f32_f32.cube" => Unary::Cube,
            "fn.f32_f32.recip" => Unary::Recip,
            "fn.f32_f32.floor" => Unary::Floor,
            "fn.f32_f32.round" => Unary::Round,
            "fn.f32_f32.const" => Unary::Const,
            "fn.f32_f32.add_c" | "fn.f32_f32.mul_c" | "fn.f32_f32.sub_c" => {
                // The captured literal is the row's child.
                let c = match tree.kids.first() {
                    Some(k) => match self.eval_at(k, input)? {
                        Value::Real(v) => v,
                        other => return Err(format!("futhark_true: {name} captured a {other:?}, not a real")),
                    },
                    None => return Err(format!("futhark_true: {name} has no captured literal")),
                };
                match name {
                    "fn.f32_f32.add_c" => Unary::AddC(c),
                    "fn.f32_f32.mul_c" => Unary::MulC(c),
                    _ => Unary::SubC(c),
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(u))
    }
}

/// The constant placeholder's literal: the value its constant-array index
/// names.
impl FutharkKingdom {
    /// The literal a `?_T` node reads.
    pub fn literal(&self, tree: &Tree) -> Result<f32, String> {
        let row = self
            .rows
            .get(tree.symbol as usize)
            .ok_or_else(|| format!("futhark_true: symbol {} is not a row of this kingdom", tree.symbol))?;
        let dual = self
            .dual(row.output)
            .ok_or_else(|| "futhark_true: a constant placeholder with no dual".to_string())?;
        let idx = tree
            .const_index
            .ok_or_else(|| format!("futhark_true: {} has no constant index", row.name))? as usize;
        let text = dual
            .literals
            .get(idx)
            .ok_or_else(|| format!("futhark_true: {} literal {idx} is past the list", row.name))?;
        text.parse::<f32>()
            .map_err(|e| format!("futhark_true: literal {text:?} is not a real: {e}"))
    }
}

fn as_fun(v: &Value) -> Result<Unary, String> {
    match v {
        Value::Fun(f) => Ok(*f),
        other => Err(format!("futhark_true: expected a function, got {other:?}")),
    }
}

fn as_arr(v: &Value) -> Result<Vec<f32>, String> {
    match v {
        Value::Arr(a) => Ok(a.clone()),
        other => Err(format!("futhark_true: expected an array, got {other:?}")),
    }
}

/// The elementwise combination of two arrays; the shorter one decides.
fn zip(a: &Value, b: &Value, f: impl Fn(f32, f32) -> f32) -> Result<Value, String> {
    let xs = as_arr(a)?;
    let ys = as_arr(b)?;
    Ok(Value::Arr(xs.iter().zip(ys.iter()).map(|(x, y)| f(*x, *y)).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Futhark's `f32.round` rounds a half to the nearest even.
    ///
    /// Measured against Futhark 0.28.1: `map f32.round (scan (+) 0.0 xs)`
    /// over `[0.25, 0.5, 0.75, 1.0, ...]` gives `[0.0, 1.0, 2.0, 2.0, ...]`,
    /// so `round(1.5)` is 2 and `round(2.5)` is 2. Rust's `f32::round`
    /// gives 2 and 3, and the difference propagates through a product and a
    /// tangent into -8.45 against 6.77.
    #[test]
    fn round_rounds_a_half_to_the_nearest_even() {
        assert_eq!(Unary::Round.apply(2.5), 2.0);
        assert_eq!(Unary::Round.apply(1.5), 2.0);
        assert_eq!(Unary::Round.apply(3.5), 4.0);
        assert_eq!(Unary::Round.apply(0.5), 0.0);
        assert_eq!(Unary::Round.apply(-2.5), -2.0);
        assert_eq!(Unary::Round.apply(-1.5), -2.0);
        assert_eq!(Unary::Round.apply(3.75), 4.0);
        // The ordinary cases are the same either way.
        assert_eq!(Unary::Round.apply(2.4), 2.0);
        assert_eq!(Unary::Round.apply(2.6), 3.0);
    }
}
