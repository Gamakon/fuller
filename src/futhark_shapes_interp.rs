//! The shape kingdom's interpreter: the same expressed tree, evaluated
//! without a compiler.
//!
//! The 1-D kingdom's interpreter is what makes its strongest claim possible —
//! the rendered program and the interpreter agree element for element, so the
//! kingdom means what it says. The shape kingdom needs the same check, and it
//! needs it in two dimensions: its operators are `map` over the ROWS of a
//! matrix, `transpose`, and `matmul`.
//!
//! The semantics here are the *Futhark* semantics of the kingdom's spellings,
//! not a convenient idealisation:
//!
//! * `map (\row -> …) A` applies the row function to each row;
//! * `map2 (\x _y -> …) A B` applies the binary row function to each PAIR of
//!   rows, so the two matrices must have the same shape — which is the
//!   relation the class set makes static;
//! * `transpose` swaps the two dimensions;
//! * the `matmul` helper is `map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a
//!   b)) (transpose B)) A`, so a result element is a left fold over the
//!   products of one row of `A` and one COLUMN of `B`, from `0.0`.

use std::collections::BTreeMap;

use crate::futhark_kingdom::{FutharkKingdom, FutharkKind, FutharkRow};
use crate::geneframe::Ty;
use crate::typed_chromosome::{AdfSet, Epoch};
use crate::typed_gep::Tree;

/// Futhark's textual form for a matrix with no elements: `empty([3][0]f32)`.
///
/// The form is parsed by hand rather than with a pattern: the crate does not
/// depend on a regular-expression engine, and the form is fixed.
fn parse_empty_matrix(text: &str) -> Option<Result<Mat, String>> {
    let rest = text.strip_prefix("empty(")?.strip_suffix(')')?;
    let rest = rest.strip_suffix("f32")?;
    let rest = rest.strip_prefix('[')?;
    let rest = rest.strip_suffix(']')?;
    let (rows, cols) = rest.split_once("][")?;
    let rows: usize = rows.parse().ok()?;
    let cols: usize = cols.parse().ok()?;
    Some(Ok(Mat { rows, cols, data: Vec::new() }))
}

/// A matrix: `rows` rows of `cols` columns of `f32`, in row-major order.
///
/// The dimensions are explicit, because `Vec<Vec<f32>>` cannot tell a `[0][m]`
/// matrix from a `[0][0]` one — and Futhark's `transpose` of a `[0][m]` matrix
/// is `[m][0]`, that is `m` empty rows, not `[]`. A kingdom whose classes *are*
/// shapes cannot afford to lose a dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub cols: usize,
    /// The values, row-major: `rows * cols` of them.
    pub data: Vec<f32>,
}

impl Mat {
    /// A matrix of the given shape and values.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Mat, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "a {rows}x{cols} matrix holds {} values, not {}",
                rows * cols,
                data.len()
            ));
        }
        Ok(Mat { rows, cols, data })
    }

    /// A matrix of the given shape, filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// A matrix from its rows, which must all have the same length.
    ///
    /// An empty list of rows is a `[0][0]` matrix: nothing in the rows says
    /// how many columns there had been.
    pub fn from_rows(rows: Vec<Vec<f32>>) -> Result<Mat, String> {
        let cols = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (i, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(format!("row {i} has {} values, but the first row has {cols}", r.len()));
            }
            data.extend_from_slice(r);
        }
        Ok(Mat { rows: rows.len(), cols, data })
    }

    /// The values of one row.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// One value.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

/// A row function, `[]f32 -> []f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowFn {
    /// `\xs -> xs`
    Id,
    /// `\xs -> map (+1.0) xs`
    Add1,
    /// `\xs -> map (*2.0) xs`
    Scale2,
    /// `\xs -> map (\_ -> 0.0) xs` — the constant row function, whose every
    /// element is zero.
    ///
    /// It is spelled as a *constant* and not as an addition of zero on
    /// purpose: Futhark's simplifier folds `x + 0` away (an identity fold, and
    /// a sound one under Futhark's semantics, which do not distinguish signed
    /// zeros), so a row spelled `map (+0.0)` would compile to the identity and
    /// differ from the identity only in the sign of a zero — a difference the
    /// comparison cannot see. Spelled as a constant, this row differs from the
    /// identity by VALUE.
    Zero,
}

/// A binary row function, `[]f32 -> []f32 -> []f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row2Fn {
    /// `\xs _ys -> xs`
    Id2,
    /// `\xs _ys -> map (+1.0) xs`
    Add1_2,
    /// `\xs _ys -> map (\_ -> 0.0) xs` — the constant binary row function.
    Zero2,
    /// `\xs ys -> map2 (+) xs ys`
    Add2,
}

impl RowFn {
    /// Apply the function to a row.
    pub fn apply(self, row: &[f32]) -> Vec<f32> {
        match self {
            RowFn::Id => row.to_vec(),
            RowFn::Add1 => row.iter().map(|x| x + 1.0).collect(),
            RowFn::Scale2 => row.iter().map(|x| x * 2.0).collect(),
            RowFn::Zero => vec![0.0; row.len()],
        }
    }
}

impl Row2Fn {
    /// Apply the function to a pair of rows.
    ///
    /// Futhark's `map2` requires the two arrays to have the same length, and
    /// so does this: the second operand is *checked*, not ignored. (Most of
    /// the kingdom's binary row functions ignore its second operand — their
    /// spellings are `\xs _ys -> ...` — but the shape obligation is real
    /// either way, and the class system is what guarantees it statically.)
    pub fn apply(self, x: &[f32], y: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != y.len() {
            return Err(format!(
                "a binary row function needs two rows of the same length, got {} and {}",
                x.len(),
                y.len()
            ));
        }
        Ok(match self {
            Row2Fn::Id2 => x.to_vec(),
            Row2Fn::Add1_2 => x.iter().map(|v| v + 1.0).collect(),
            Row2Fn::Zero2 => vec![0.0; x.len()],
            Row2Fn::Add2 => x.iter().zip(y.iter()).map(|(u, v)| u + v).collect(),
        })
    }
}

/// A value of the shape kingdom.
#[derive(Debug, Clone, PartialEq)]
pub enum MValue {
    Mat(Mat),
    Fun(RowFn),
    Fun2(Row2Fn),
}

/// The row function a kingdom row denotes, if it is one.
fn row_fn(name: &str) -> Option<RowFn> {
    match name {
        "fn.arr_f32_arr_f32.id" => Some(RowFn::Id),
        "fn.arr_f32_arr_f32.add1" => Some(RowFn::Add1),
        "fn.arr_f32_arr_f32.scale2" => Some(RowFn::Scale2),
        "fn.arr_f32_arr_f32.const" => Some(RowFn::Zero),
        _ => None,
    }
}

/// The binary row function a kingdom row denotes, if it is one.
fn row2_fn(name: &str) -> Option<Row2Fn> {
    match name {
        "fn.arr2_f32_arr_f32_arr_f32.id" => Some(Row2Fn::Id2),
        "fn.arr2_f32_arr_f32_arr_f32.add1" => Some(Row2Fn::Add1_2),
        "fn.arr2_f32_arr_f32_arr_f32.add" => Some(Row2Fn::Add2),
        "fn.arr2_f32_arr_f32_arr_f32.const" => Some(Row2Fn::Zero2),
        _ => None,
    }
}

/// The transpose of a matrix.
///
/// Futhark's `transpose` swaps the dimensions, so a `[0][m]` matrix transposes
/// to `[m][0]` — `m` empty rows. A representation that cannot hold a row count
/// loses that, and the difference is visible in the result.
pub fn transpose(m: &Mat) -> Mat {
    let mut data = vec![0.0f32; m.rows * m.cols];
    for i in 0..m.rows {
        for j in 0..m.cols {
            data[j * m.rows + i] = m.data[i * m.cols + j];
        }
    }
    Mat { rows: m.cols, cols: m.rows, data }
}

/// The kingdom's `matmul`: `map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a b))
/// (transpose B)) A`.
///
/// A result element is a left fold over the products of one row of `A` and one
/// COLUMN of `B`, from `0.0`. An empty inner dimension therefore gives a
/// zero, which is what Futhark's `reduce (+) 0.0` over an empty array gives.
///
/// One caveat, measured: a reduction over a **singleton** inner dimension is
/// folded by Futhark's simplifier (`0.0 + x` is an identity fold), so the
/// compiled product of a `[-0.0]` row is `-0.0` where this left fold gives
/// `+0.0`. The comparison for a reduction is [`equal`], which accepts the two
/// zeros; the kingdom's own rows are unaffected, because their inner
/// dimension is the interface's `m`.
pub fn matmul(a: &Mat, b: &Mat) -> Result<Mat, String> {
    if a.cols != b.rows {
        return Err(format!(
            "matmul: the inner dimensions disagree ({} and {})",
            a.cols, b.rows
        ));
    }
    let mut data = vec![0.0f32; a.rows * b.cols];
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut acc = 0.0f32;
            for k in 0..a.cols {
                acc += a.get(i, k) * b.get(k, j);
            }
            data[i * b.cols + j] = acc;
        }
    }
    Ok(Mat { rows: a.rows, cols: b.cols, data })
}

/// Whether two values agree, under the kingdom's comparison policy.
///
/// The policy is **exact**: two values agree when they are the same number,
/// or when both are NaN (a NaN's payload is not part of its value). There is
/// **no tolerance**, and there is none because the tolerance's stated reason
/// was false: Futhark 0.28.1's `c` backend reduces with a sequential left
/// fold, measured **bit-identical** to this interpreter's over 600 of 600
/// elements, and an order-sensitive probe (`[16777216, 1, -16777216, 1]`)
/// distinguishes a left fold from a tree sum in both. A margin that is never
/// needed is a blind spot: it would hide a last-ulp wrong result from a
/// spelling that drifted.
///
/// The **signed zeros agree** here, and that is the one place where the
/// compiled program and the source semantics differ. Futhark's simplifier
/// folds `x + 0` away — an identity fold, and a sound one under Futhark's
/// semantics, which do not distinguish signed zeros — so the compiled
/// `x + 0.0` is the identity and keeps `-0.0`'s sign, where the source
/// semantics turns it into `+0.0`. What is narrow is the *identity fold*,
/// not the sign of a zero: the compiled program still tells the zeros apart
/// wherever the fold does not apply — `1.0/x` on `-0.0` gives `-f32.inf`.
///
/// No row of the table relies on the fold. The `const` rows are spelled as
/// `map (\_ -> 0.0)`, a genuine constant, so they differ from the identity by
/// VALUE and not only in the sign of a zero. For the paths whose arithmetic
/// is exactly specified — a row function, a `map`, a `map2`, a `transpose` —
/// use [`identical`], which does tell the signed zeros apart.
pub fn equal(x: f32, y: f32) -> bool {
    x == y || (x.is_nan() && y.is_nan())
}

/// Whether two values agree **bit for bit**.
///
/// The same as [`equal`], except that the signed zeros are different values:
/// `-0.0` and `+0.0` have different bit patterns, so they do not agree. A
/// NaN agrees with a NaN, because a NaN's payload is not part of its value.
///
/// This is the comparison for the paths whose arithmetic is exactly
/// specified, where the compiled program must agree with the interpreter down
/// to the last bit — and where a row that differs from another only in the
/// sign of a zero would otherwise slip through. It is **not** the comparison
/// for a reduction: there the compiled program may legitimately differ (see
/// [`equal`]).
pub fn identical(x: f32, y: f32) -> bool {
    if x.is_nan() && y.is_nan() {
        return true;
    }
    x.to_bits() == y.to_bits()
}

fn as_mat(v: &MValue) -> Result<&Mat, String> {
    match v {
        MValue::Mat(m) => Ok(m),
        other => Err(format!("a matrix was expected, got {other:?}")),
    }
}

fn as_fun(v: &MValue) -> Result<RowFn, String> {
    match v {
        MValue::Fun(f) => Ok(*f),
        other => Err(format!("a row function was expected, got {other:?}")),
    }
}

fn as_fun2(v: &MValue) -> Result<Row2Fn, String> {
    match v {
        MValue::Fun2(f) => Ok(*f),
        other => Err(format!("a binary row function was expected, got {other:?}")),
    }
}

impl FutharkKingdom {
    /// Evaluate an expressed tree of the shape kingdom.
    ///
    /// `inputs` holds one matrix per class, keyed by the class's type. A
    /// terminal of a matrix class is that class's input; a terminal of a
    /// function class is the constant function of that class.
    pub fn eval_matrices(&self, tree: &Tree, inputs: &BTreeMap<Ty, Mat>) -> Result<MValue, String> {
        self.eval_matrix_at(tree, inputs)
    }

    /// Evaluate a FOLDED tree (over the epoch alphabet): its ADF references
    /// are unfolded first, so the interpreter only ever sees the kingdom's rows.
    pub fn eval_matrices_folded(&self, epoch: &Epoch, adfs: &AdfSet, tree: &Tree, inputs: &BTreeMap<Ty, Mat>) -> Result<MValue, String> {
        self.eval_matrix_at(&adfs.unfold(epoch, tree), inputs)
    }

    fn eval_matrix_at(&self, tree: &Tree, inputs: &BTreeMap<Ty, Mat>) -> Result<MValue, String> {
        let row = self
            .rows
            .get(tree.symbol as usize)
            .ok_or_else(|| format!("{}: symbol {} is not a row of this kingdom", self.kingdom, tree.symbol))?;
        match row.kind {
            FutharkKind::Const => Err(format!("{}: the shape kingdom has no constant placeholder", self.kingdom)),
            FutharkKind::Terminal => {
                if let Some(m) = inputs.get(&row.output) {
                    return Ok(MValue::Mat(m.clone()));
                }
                if let Some(f) = row_fn(&row.name) {
                    return Ok(MValue::Fun(f));
                }
                if let Some(f) = row2_fn(&row.name) {
                    return Ok(MValue::Fun2(f));
                }
                Err(format!("{}: the terminal {} has no input", self.kingdom, row.name))
            }
            FutharkKind::Function => {
                let kids: Vec<MValue> = tree
                    .kids
                    .iter()
                    .map(|c| self.eval_matrix_at(c, inputs))
                    .collect::<Result<Vec<_>, _>>()?;
                self.apply_row(row, &kids)
            }
        }
    }

    fn apply_row(&self, row: &FutharkRow, kids: &[MValue]) -> Result<MValue, String> {
        if let Some(f) = row_fn(&row.name) {
            return Ok(MValue::Fun(f));
        }
        if let Some(f) = row2_fn(&row.name) {
            return Ok(MValue::Fun2(f));
        }
        let name = row.name.as_str();
        if name.starts_with("map_rows@") {
            let f = as_fun(&kids[0])?;
            let m = as_mat(&kids[1])?;
            let mut data = Vec::with_capacity(m.data.len());
            for i in 0..m.rows {
                data.extend_from_slice(&f.apply(m.row(i)));
            }
            return Ok(MValue::Mat(Mat { rows: m.rows, cols: m.cols, data }));
        }
        if name.starts_with("map2_rows@") {
            let f = as_fun2(&kids[0])?;
            let a = as_mat(&kids[1])?;
            let b = as_mat(&kids[2])?;
            if a.rows != b.rows || a.cols != b.cols {
                return Err(format!(
                    "{name}: the two matrices are {}x{} and {}x{}",
                    a.rows, a.cols, b.rows, b.cols
                ));
            }
            let mut data = Vec::with_capacity(a.data.len());
            for i in 0..a.rows {
                data.extend_from_slice(&f.apply(a.row(i), b.row(i))?);
            }
            return Ok(MValue::Mat(Mat { rows: a.rows, cols: a.cols, data }));
        }
        if name.starts_with("transpose@") {
            let m = as_mat(&kids[0])?;
            return Ok(MValue::Mat(transpose(m)));
        }
        if name.starts_with("matmul@") {
            let a = as_mat(&kids[0])?;
            let b = as_mat(&kids[1])?;
            return Ok(MValue::Mat(matmul(a, b)?));
        }
        Err(format!("{}: the shape kingdom cannot evaluate {}", self.kingdom, row.name))
    }
}

// ---------------------------------------------------------------------------
// Futhark's textual format for a matrix.
// ---------------------------------------------------------------------------

/// A matrix in Futhark's textual format: `[[1.0,2.0],[3.0,4.0]]`.
///
/// A matrix with no elements has no rows to write, so Futhark writes its
/// SHAPE instead: `empty([3][0]f32)`. That is the only way the format can
/// carry a degenerate dimension — `[[],[],[]]` is not accepted by Futhark's
/// own reader — so the format is used here too.
pub fn format_matrix(m: &Mat) -> String {
    if m.rows == 0 || m.cols == 0 {
        return format!("empty([{}][{}]f32)", m.rows, m.cols);
    }
    let rows: Vec<String> = (0..m.rows)
        .map(|i| {
            let vals: Vec<String> = m.row(i).iter().map(|v| format_f32(*v)).collect();
            format!("[{}]", vals.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

/// A value in Futhark's textual format.
///
/// Rust writes `NaN` and `inf`, and Futhark reads neither: it writes and
/// reads `f32.nan`, `f32.inf` and `-f32.inf`. So the specials are spelled
/// Futhark's way, and everything else is Rust's — which is what Futhark
/// accepts (an integer-looking literal is read as an `f32`).
fn format_f32(v: f32) -> String {
    if v.is_nan() {
        return "f32.nan".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() { "-f32.inf".to_string() } else { "f32.inf".to_string() };
    }
    format!("{v}")
}

/// Parse a matrix in Futhark's textual format.
///
/// The output format carries a type suffix (`58f32`), so the suffix is
/// stripped before the number is read. An empty matrix carries its shape
/// instead — `empty([3][0]f32)` — and that shape is kept: a `[0][m]` matrix
/// is not a `[0][0]` one, and Futhark's `transpose` of the first is `[m][0]`.
pub fn parse_matrix(text: &str) -> Result<Mat, String> {
    let t = text.trim();
    if let Some(empty) = parse_empty_matrix(t) {
        return empty;
    }
    let inner = t
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("{t:?} is not a matrix"))?;
    let mut rows = Vec::new();
    for part in split_top_level(inner) {
        let row = part.trim();
        let cells = row
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| format!("{row:?} is not a row"))?;
        let mut values = Vec::new();
        for cell in split_top_level(cells) {
            values.push(parse_f32(cell.trim())?);
        }
        rows.push(values);
    }
    Mat::from_rows(rows)
}

/// Split on the commas that are not inside brackets.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in text.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// A number in Futhark's output format: `58f32`, `-0.5`, `f32.inf`,
/// `-f32.inf`, `f32.nan`.
///
/// The special values are checked FIRST, and they carry Futhark's own type
/// prefix: Futhark writes `f32.inf` and `f32.nan`, not `inf` and `nan`. The
/// prefix is optional here, so both spellings are read, but the one Futhark
/// writes is the one the test uses.
fn parse_f32(text: &str) -> Result<f32, String> {
    let t = text.trim();
    let unsigned = t.strip_prefix('-').unwrap_or(t);
    let unsigned = unsigned.strip_prefix("f32.").unwrap_or(unsigned);
    if unsigned.starts_with("inf") {
        return Ok(if t.starts_with('-') { f32::NEG_INFINITY } else { f32::INFINITY });
    }
    if unsigned.starts_with("nan") {
        return Ok(f32::NAN);
    }
    let numeric: String = t
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.' || *c == 'e' || *c == 'E')
        .collect();
    if numeric.is_empty() {
        return Err(format!("{t:?} is not a number"));
    }
    numeric
        .parse::<f32>()
        .map_err(|e| format!("{numeric:?} is not a number: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: usize, cols: usize, data: &[f32]) -> Mat {
        Mat::new(rows, cols, data.to_vec()).expect("the shape matches the values")
    }

    #[test]
    fn a_matrix_round_trips_through_futharks_textual_format() {
        let mat = m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let text = format_matrix(&mat);
        assert_eq!(text, "[[1,2,3],[4,5,6]]");
        assert_eq!(parse_matrix(&text).expect("parses"), mat);
    }

    #[test]
    fn the_output_formats_type_suffix_is_stripped() {
        let mat = parse_matrix("[[58f32, 64f32], [139f32, 154f32]]").expect("parses");
        assert_eq!(mat, m(2, 2, &[58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn the_transpose_swaps_the_dimensions() {
        let mat = m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(transpose(&mat), m(3, 2, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn the_matmul_is_the_kernels_own_formula() {
        // The same numbers Futhark computed: A = [[1,2,3],[4,5,6]],
        // B = [[7,8],[9,10],[11,12]].
        let a = m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = m(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(matmul(&a, &b).expect("the shapes agree"), m(2, 2, &[58.0, 64.0, 139.0, 154.0]));
    }

    /// The degenerate shapes are where a representation that cannot hold a
    /// dimension goes wrong: Futhark's `transpose` of a `[0][m]` matrix is
    /// `[m][0]` — `m` empty rows, not `[]` — and a `matmul` over an empty
    /// inner dimension is a zero matrix, not an empty one.
    #[test]
    fn a_matrix_with_no_rows_keeps_its_column_count() {
        let empty = Mat { rows: 0, cols: 3, data: Vec::new() };
        let t = transpose(&empty);
        assert_eq!((t.rows, t.cols), (3, 0), "the transpose of [0][3] is [3][0]");
        assert!(t.data.is_empty());
    }

    #[test]
    fn a_matmul_over_an_empty_inner_dimension_is_a_zero_matrix() {
        // A : [2][0] and B : [0][2] multiply to a [2][2] matrix of zeros,
        // because Futhark's `reduce (+) 0.0` over an empty array is 0.0.
        let a = Mat { rows: 2, cols: 0, data: Vec::new() };
        let b = Mat { rows: 0, cols: 2, data: Vec::new() };
        let got = matmul(&a, &b).expect("the inner dimensions agree");
        assert_eq!((got.rows, got.cols), (2, 2));
        assert_eq!(got.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_matmul_with_mismatched_inner_dimensions_is_refused() {
        let a = m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = m(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(matmul(&a, &b).is_err(), "3 is not 2");
    }

    #[test]
    fn the_special_values_parse_with_their_signs() {
        // Futhark writes `f32.inf`, `-f32.inf` and `f32.nan` -- not `inf` and
        // `nan` -- so those are the spellings the parser has to read.
        let mat = parse_matrix("[[f32.inf, -f32.inf, f32.nan]]").expect("parses");
        assert_eq!(mat.data[0], f32::INFINITY);
        assert_eq!(mat.data[1], f32::NEG_INFINITY);
        assert!(mat.data[2].is_nan(), "a NaN is a NaN, even though it is not equal to itself");
    }

    /// The specials go out in Futhark's spelling too: Rust writes `inf` and
    /// `NaN`, and Futhark reads neither.
    #[test]
    fn the_special_values_are_written_in_futharks_spelling() {
        let mat = Mat::new(1, 3, vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).expect("the shape fits");
        assert_eq!(format_matrix(&mat), "[[f32.inf,-f32.inf,f32.nan]]");
    }

    /// The comparison policy, stated as tests.
    ///
    /// There are two comparisons, and the difference between them is the
    /// whole point of this episode:
    ///
    /// * [`equal`] is **exact**: the same number, or two NaNs. It has no
    ///   tolerance, because the tolerance's stated reason was false — the
    ///   backend reduces with a sequential left fold, bit-identical to this
    ///   interpreter's over 600 of 600 elements;
    /// * [`identical`] is exact **and** bit-for-bit, so the signed zeros are
    ///   different values. It is the comparison for the paths whose
    ///   arithmetic is exactly specified.
    ///
    /// The signed zeros agree under `equal` because Futhark's simplifier folds
    /// `x + 0` away, so the compiled program keeps `-0.0`'s sign where the
    /// source semantics would turn it into `+0.0`. They do NOT agree under
    /// `identical`, which is what keeps a row that differs from another only
    /// in the sign of a zero from slipping through.
    #[test]
    fn the_agreement_policy_is_exact_and_has_two_comparisons() {
        assert!(equal(1.0, 1.0));
        assert!(equal(0.0, 0.0));
        assert!(equal(0.0, -0.0), "the identity fold keeps the sign of a zero");
        assert!(equal(f32::NAN, f32::NAN), "a NaN agrees with a NaN");
        assert!(!equal(f32::NAN, 1.0));
        assert!(equal(f32::INFINITY, f32::INFINITY));
        assert!(!equal(f32::INFINITY, f32::NEG_INFINITY));
        assert!(!equal(f32::INFINITY, f32::MAX));

        assert!(identical(1.0, 1.0));
        assert!(identical(0.0, 0.0));
        assert!(!identical(0.0, -0.0), "the signed zeros are different values");
        assert!(identical(f32::NAN, f32::NAN), "a NaN's payload is not its value");
        assert!(!identical(f32::NAN, 1.0));
        assert!(identical(f32::INFINITY, f32::INFINITY));
        assert!(!identical(f32::INFINITY, f32::NEG_INFINITY));
    }

    /// There is no tolerance, so a small wrong result is caught.
    #[test]
    fn a_small_wrong_result_is_not_a_zero() {
        assert!(!equal(0.0, 0.00009), "9e-5 is not zero");
        assert!(!equal(0.0, 1e-5), "1e-5 is not zero");
        assert!(!equal(0.0, 1e-8), "a rounding of a zero is still a different value");
        assert!(!equal(0.0, f32::NAN));
        assert!(!equal(1e5, 1e5 + 1.0), "a whole unit at that magnitude is too far");
        // The very next float up is a difference: there is no margin at all.
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        assert!(next > 1.0, "the next float up is bigger");
        assert!(!equal(1.0, next), "the margin is gone: one ulp is a difference");
    }
}
