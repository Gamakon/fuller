//! HOW CLOSE A WRONG KERNEL IS (2026-10-12, Andrew: looser measures of how
//! close a wrong kernel's output is to the reference, as fitness columns for
//! an infeasible kernel, folded into the HFF angle). Each column is in
//! [0, 1], 0 the reference itself, computed on the host from the outputs the
//! constraint already reads back (the 128 cases, `d × d` row-major) and a
//! failing pair's. A feasible kernel never reaches them: its fitness is its
//! own six columns, and an infeasible one stays in the band above every
//! feasible fitness (phylu `wgsl_fit`).
//!
//! | column | 0 when | catches |
//! |---|---|---|
//! | `cases` | every case passes | the constraint's count, as before |
//! | `correlation` | Pearson 1 | the right pattern at any scale or offset |
//! | `scale` | the output is a constant multiple of the reference | a factor of 2, a missed half, a factor of K |
//! | `shift` | the output, or a shift of it by 1 or 2 rows or columns, matches | an off-by-one or -two in an index |
//! | `transpose` | the output or its transpose matches | rows and columns swapped |
//! | `sorted` | the sorted values match | the right numbers in the wrong cells |
//! | `sums` | every row and column sum matches | the right totals, spread wrongly |
//! | `written` | every non-zero reference cell is non-zero | cells never written (a read-back cell is cleared to 0 first) |
//! | `within` | every cell within the tolerance | the share of cells already right |
//! | `sensitivity` | the output changes whenever the input does | a kernel blind to Φ |
//!
//! A relative error `e` (an L2 norm over the reference's) enters as
//! `min(1, e)`; Pearson's ρ as `(1 − ρ) / 2`; a share `s` of good cells as
//! `1 − s`. A non-finite output cell where the reference is finite makes
//! every error column of that case 1.

/// The columns, in order.
pub const COLUMNS: [&str; 10] = ["cases", "correlation", "scale", "shift", "transpose", "sorted", "sums", "written", "within", "sensitivity"];

/// The number of columns.
pub const N: usize = COLUMNS.len();

/// The per-cell tolerance of `within`, relative to the reference's largest
/// magnitude (the constraint's own pass bound per cell, `probes::PASS`).
pub const CELL_TOLERANCE: f64 = 1e-5;

/// The shifts tried, rows and columns: every (dr, dc) in −2..=2, (0, 0)
/// among them, so the reference itself scores 0.
const SHIFT: i64 = 2;

/// One case's columns but `cases` and `sensitivity` (0 there): `h` the
/// kernel's output, `want` the reference, both `d × d` row-major.
pub fn case_columns(h: &[f32], want: &[f64], d: usize) -> [f64; N] {
    tensor_columns(h, want, d, d)
}

/// [`case_columns`] for any output tensor, read as `rows × cols`
/// row-major (a vector is one row; the generic harness, `harness.rs`, reads
/// a problem's output so). A square tensor gives [`case_columns`]' values
/// bit for bit; on a non-square one `transpose` is the plain error and
/// `sums` compares row sums with row sums and column sums with column sums.
pub fn tensor_columns(h: &[f32], want: &[f64], rows: usize, cols: usize) -> [f64; N] {
    let mut out = [0.0; N];
    let n = rows * cols;
    if h.len() != n || want.len() != n || n == 0 {
        out[1..9].iter_mut().for_each(|x| *x = 1.0);
        return out;
    }
    let g: Vec<f64> = h.iter().map(|v| f64::from(*v)).collect();
    if g.iter().zip(want).any(|(a, b)| b.is_finite() && !a.is_finite()) {
        out[1..9].iter_mut().for_each(|x| *x = 1.0);
        return out;
    }
    // Non-finite reference cells (the non-finite probes) are left out of
    // every norm: those cases are judged by `within` and the count.
    let finite: Vec<bool> = want.iter().map(|w| w.is_finite()).collect();
    let r = |i: usize| if finite[i] { want[i] } else { 0.0 };
    let o = |i: usize| if finite[i] { g[i] } else { 0.0 };
    let norm_r = (0..n).map(|i| r(i) * r(i)).sum::<f64>().sqrt();
    let rel = |err2: f64| -> f64 {
        if norm_r > 0.0 {
            (err2.sqrt() / norm_r).min(1.0)
        } else if err2 == 0.0 {
            0.0
        } else {
            1.0
        }
    };
    // Pearson.
    let mean_r = (0..n).map(r).sum::<f64>() / n as f64;
    let mean_o = (0..n).map(o).sum::<f64>() / n as f64;
    let (mut sro, mut srr, mut soo) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (r(i) - mean_r, o(i) - mean_o);
        sro += a * b;
        srr += a * a;
        soo += b * b;
    }
    out[1] = if srr > 0.0 && soo > 0.0 {
        ((1.0 - sro / (srr * soo).sqrt()) / 2.0).clamp(0.0, 1.0)
    } else if (0..n).all(|i| o(i) == r(i)) {
        0.0
    } else {
        1.0
    };
    // The best constant multiple k = <o, r> / <r, r>, and the error left.
    let rr: f64 = (0..n).map(|i| r(i) * r(i)).sum();
    let k = if rr > 0.0 { (0..n).map(|i| o(i) * r(i)).sum::<f64>() / rr } else { 0.0 };
    out[2] = rel((0..n).map(|i| (o(i) - k * r(i)).powi(2)).sum());
    // Shifts: the output read at (row + dr, col + dc), outside the matrix 0.
    let at = |row: i64, col: i64| -> f64 {
        if row < 0 || col < 0 || row >= rows as i64 || col >= cols as i64 {
            0.0
        } else {
            o(row as usize * cols + col as usize)
        }
    };
    let mut best_shift = f64::INFINITY;
    for dr in -SHIFT..=SHIFT {
        for dc in -SHIFT..=SHIFT {
            let mut e = 0.0;
            for row in 0..rows {
                for col in 0..cols {
                    let x = at(row as i64 + dr, col as i64 + dc) - r(row * cols + col);
                    e += x * x;
                }
            }
            best_shift = best_shift.min(e);
        }
    }
    out[3] = rel(best_shift);
    let plain: f64 = (0..n).map(|i| (o(i) - r(i)).powi(2)).sum();
    let d = rows;
    let transposed: f64 = if rows == cols { (0..d).flat_map(|row| (0..d).map(move |col| (row, col))).map(|(row, col)| (o(col * d + row) - r(row * d + col)).powi(2)).sum() } else { plain };
    out[4] = rel(plain.min(transposed));
    // Sorted values.
    let mut so: Vec<f64> = (0..n).map(o).collect();
    let mut sr: Vec<f64> = (0..n).map(r).collect();
    so.sort_by(f64::total_cmp);
    sr.sort_by(f64::total_cmp);
    out[5] = rel(so.iter().zip(&sr).map(|(a, b)| (a - b).powi(2)).sum());
    // Row and column sums, against the reference's (or against its column
    // and row sums, whichever is closer: a transpose keeps its totals).
    let (mut e_sums, mut e_swapped, mut n_sums) = (0.0, 0.0, 0.0);
    if rows == cols {
        for i in 0..d {
            let (ro, rr_) = ((0..d).map(|c| o(i * d + c)).sum::<f64>(), (0..d).map(|c| r(i * d + c)).sum::<f64>());
            let (co, cr) = ((0..d).map(|x| o(x * d + i)).sum::<f64>(), (0..d).map(|x| r(x * d + i)).sum::<f64>());
            e_sums += (ro - rr_).powi(2) + (co - cr).powi(2);
            e_swapped += (ro - cr).powi(2) + (co - rr_).powi(2);
            n_sums += rr_ * rr_ + cr * cr;
        }
    } else {
        for i in 0..rows {
            let (ro, rr_) = ((0..cols).map(|c| o(i * cols + c)).sum::<f64>(), (0..cols).map(|c| r(i * cols + c)).sum::<f64>());
            e_sums += (ro - rr_).powi(2);
            n_sums += rr_ * rr_;
        }
        for j in 0..cols {
            let (co, cr) = ((0..rows).map(|x| o(x * cols + j)).sum::<f64>(), (0..rows).map(|x| r(x * cols + j)).sum::<f64>());
            e_sums += (co - cr).powi(2);
            n_sums += cr * cr;
        }
        e_swapped = e_sums;
    }
    let e_sums = e_sums.min(e_swapped);
    out[6] = if n_sums > 0.0 { (e_sums.sqrt() / n_sums.sqrt()).min(1.0) } else if e_sums == 0.0 { 0.0 } else { 1.0 };
    // Written: the non-zero reference cells the output holds non-zero.
    let nonzero: Vec<usize> = (0..n).filter(|&i| want[i] != 0.0).collect();
    out[7] = if nonzero.is_empty() { 0.0 } else { nonzero.iter().filter(|&&i| g[i] == 0.0).count() as f64 / nonzero.len() as f64 };
    // Within the tolerance, cell by cell (a non-finite reference cell must be matched exactly).
    let scale = (0..n).map(|i| r(i).abs()).fold(0.0f64, f64::max).max(1.0);
    let good = (0..n)
        .filter(|&i| if finite[i] { (g[i] - want[i]).abs() / scale <= CELL_TOLERANCE } else { (g[i].is_nan() && want[i].is_nan()) || g[i] == want[i] })
        .count();
    out[8] = 1.0 - good as f64 / n as f64;
    out
}

/// A run's columns: the mean of its cases' (a case that never ran counts 1
/// in every column), `cases` the share failed, and `sensitivity` the share
/// of case pairs of one shape whose references differ while the outputs are
/// bit-identical. `per_case` is every case's columns, `None` for one never
/// run; `outputs` the hash of each case's output and of its reference with
/// its `d` (`None` for one never run).
pub fn columns(per_case: &[Option<[f64; N]>], passed: usize, outputs: &[Option<(u64, u64, usize)>]) -> [f64; N] {
    let cases = per_case.len().max(1);
    let mut out = [0.0; N];
    for c in per_case {
        let c = c.unwrap_or([1.0; N]);
        for (o, x) in out.iter_mut().zip(c) {
            *o += x / cases as f64;
        }
    }
    out[0] = 1.0 - passed.min(cases) as f64 / cases as f64;
    let ran: Vec<(u64, u64, usize)> = outputs.iter().flatten().copied().collect();
    let (mut differ, mut blind) = (0usize, 0usize);
    for (i, a) in ran.iter().enumerate() {
        for b in &ran[i + 1..] {
            if a.2 == b.2 && a.1 != b.1 {
                differ += 1;
                if a.0 == b.0 {
                    blind += 1;
                }
            }
        }
    }
    out[9] = if ran.is_empty() { 1.0 } else if differ == 0 { 0.0 } else { blind as f64 / differ as f64 };
    out
}

/// The columns out of the angle, logged only: `scale` pointed the wrong way
/// on all three saved runs (2026-10-12: kernels one rule-step from correct
/// had the larger error after the best constant multiple; AUC 0.32, 0.22,
/// 0.25), a known defect until understood.
pub const LOGGED_ONLY: [&str; 1] = ["scale"];

/// The HFF angle (`gram::truenorth`'s `acos(1 − mean(x²))`) over the columns
/// but [`LOGGED_ONLY`]: an infeasible kernel's place in the band.
pub fn angle(c: &[f64; N]) -> f64 {
    let kept: Vec<f64> = c.iter().zip(COLUMNS).filter(|(_, name)| !LOGGED_ONLY.contains(name)).map(|(x, _)| *x).collect();
    let energy = kept.iter().map(|v| v.clamp(0.0, 1.0).powi(2)).sum::<f64>() / kept.len().max(1) as f64;
    let cos = (1.0 - energy.min(1.0)).clamp(-1.0, 1.0);
    if cos > 1.0 - 1.1920929e-7 {
        0.0
    } else {
        cos.acos()
    }
}

/// Every column 1: a kernel with no output to compare (it did not compile,
/// render, or run).
pub const WORST: [f64; N] = [1.0; N];

/// A hash of a slice of floats' bits (FNV-1a), for `sensitivity`.
pub fn hash_bits<T: Copy>(values: &[T], bits: impl Fn(T) -> u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        h ^= bits(*v);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(d: usize) -> Vec<f64> {
        (0..d * d).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect()
    }

    fn f32s(v: &[f64]) -> Vec<f32> {
        v.iter().map(|x| *x as f32).collect()
    }

    /// The reference itself scores 0 in every case column; each kind of
    /// wrongness is caught by its own column and not by the plain error.
    #[test]
    fn each_column_catches_its_own_wrongness() {
        let d = 8;
        let want = reference(d);
        let same = case_columns(&f32s(&want), &want, d);
        assert!(same.iter().all(|x| *x == 0.0), "{same:?}");
        let col = |name: &str| COLUMNS.iter().position(|c| *c == name).unwrap();
        // A factor of 2: scale and correlation 0, within near 1.
        let doubled: Vec<f64> = want.iter().map(|x| 2.0 * x).collect();
        let c = case_columns(&f32s(&doubled), &want, d);
        assert!(c[col("scale")] < 1e-9 && c[col("correlation")] < 1e-9 && c[col("within")] > 0.8, "{c:?}");
        // A transpose.
        let t: Vec<f64> = (0..d * d).map(|i| want[(i % d) * d + i / d]).collect();
        let c = case_columns(&f32s(&t), &want, d);
        assert!(c[col("transpose")] < 1e-9 && c[col("sorted")] < 1e-9 && c[col("sums")] < 1e-9 && c[col("scale")] > 0.1, "{c:?}");
        // A shift of one column: the output at (r, c + 1) holds want (r, c).
        let s: Vec<f64> = (0..d * d).map(|i| if i % d == 0 { 0.0 } else { want[i - 1] }).collect();
        let c = case_columns(&f32s(&s), &want, d);
        assert!(c[col("shift")] < c[col("scale")] && c[col("shift")] < 0.5, "{c:?}");
        // The right numbers in the wrong cells.
        let mut p = want.clone();
        p.reverse();
        let c = case_columns(&f32s(&p), &want, d);
        assert!(c[col("sorted")] < 1e-9 && c[col("within")] > 0.5, "{c:?}");
        // Nothing written: every non-zero cell 0.
        let c = case_columns(&vec![0.0f32; d * d], &want, d);
        assert!(c[col("written")] > 0.9 && c[col("correlation")] == 1.0, "{c:?}");
        // A non-finite output where the reference is finite: every error column 1.
        let mut nan = f32s(&want);
        nan[3] = f32::NAN;
        let c = case_columns(&nan, &want, d);
        assert!(c[1..9].iter().all(|x| *x == 1.0), "{c:?}");
    }

    /// The run's columns: the share of cases failed, a case never run as
    /// every column 1, and a kernel whose output never changes blind.
    #[test]
    fn a_runs_columns_count_failed_cases_and_blindness() {
        let one = [0.0; N];
        let cols = columns(&[Some(one), Some(one), None, None], 2, &[Some((1, 10, 8)), Some((1, 11, 8)), None, None]);
        assert_eq!(cols[0], 0.5);
        assert_eq!(cols[1], 0.5, "two of four cases never ran");
        assert_eq!(cols[9], 1.0, "the references differ, the outputs do not");
        let seeing = columns(&[Some(one), Some(one)], 2, &[Some((1, 10, 8)), Some((2, 11, 8))]);
        assert_eq!(seeing, [0.0; N]);
        assert_eq!(columns(&[None], 0, &[None]), WORST);
    }

    /// The angle leaves `scale` out: changing it alone moves nothing, any
    /// other column moves it; all 0 is 0, all 1 is π/2.
    #[test]
    fn the_angle_leaves_scale_out() {
        let mut c = [0.2; N];
        let base = angle(&c);
        c[2] = 0.9;
        assert_eq!(angle(&c), base);
        c[1] = 0.9;
        assert!(angle(&c) > base);
        assert_eq!(angle(&[0.0; N]), 0.0);
        assert!((angle(&WORST) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }
}
