//! SMOTE for regression: a synthetic row ON THE SEGMENT between a real row and
//! one of its nearest neighbours — inputs and target interpolated by the same
//! `u` in [0, 1). No noise, no embedding: the row is as near the manifold as two
//! neighbouring real rows are. It is the companion of [`super::smogd`] in HFF's
//! third block (tournaments only; never fitted on, never the stop bar).
//!
//! Neighbours are found in the scaled inputs, `(x - median) / (p95 - p5)` per
//! column, so units do not matter. Randomness is the engine's counter-based
//! `draw`: the rows are a function of (data, seed).

use super::smogd::unit;
use super::{below, draw};

pub const STREAM_SMOTE: u32 = 320;
pub const K_NEIGHBORS: usize = 5;

/// numpy's default (linear-interpolation) percentile of a sorted column.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let at = q / 100.0 * (sorted.len() - 1) as f64;
    let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo as f64)
}

/// One synthetic row's provenance: `parent + u * (neighbour - parent)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub parent: usize,
    pub neighbour: usize,
    pub u: f64,
}

/// `count` pairs: a random real row, one of its `K_NEIGHBORS` nearest rows
/// (ties to the lower index), and the interpolation weight.
pub fn pairs(x: &[f64], width: usize, count: usize, seed: u32) -> Vec<Pair> {
    if width == 0 || x.len() < 2 * width {
        return Vec::new();
    }
    let n = x.len() / width;
    let mut z = x.to_vec();
    for c in 0..width {
        let mut column: Vec<f64> = (0..n).map(|r| x[r * width + c]).collect();
        column.sort_by(f64::total_cmp);
        let spread = percentile(&column, 95.0) - percentile(&column, 5.0);
        let (median, spread) = (percentile(&column, 50.0), if spread == 0.0 { 1.0 } else { spread });
        for r in 0..n {
            z[r * width + c] = (x[r * width + c] - median) / spread;
        }
    }
    let k = K_NEIGHBORS.min(n - 1);
    (0..count as u32)
        .map(|i| {
            let parent = below(draw(seed, 0, i, 0, STREAM_SMOTE), n as u32) as usize;
            let a = &z[parent * width..(parent + 1) * width];
            let mut order: Vec<(f64, usize)> = (0..n)
                .filter(|&o| o != parent)
                .map(|o| (a.iter().zip(&z[o * width..(o + 1) * width]).map(|(p, q)| (p - q) * (p - q)).sum(), o))
                .collect();
            order.select_nth_unstable_by(k - 1, |p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
            order.truncate(k);
            order.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
            let neighbour = order[below(draw(seed, 0, i, 1, STREAM_SMOTE), k as u32) as usize].1;
            Pair { parent, neighbour, u: unit(draw(seed, 0, i, 2, STREAM_SMOTE)) }
        })
        .collect()
}

/// The synthetic rows: `(x, y)`, `x` row-major with `width` columns.
pub fn rows(x: &[f64], y: &[f64], width: usize, count: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
    let chosen = pairs(x, width, count, seed);
    let lerp = |a: f64, b: f64, u: f64| a + u * (b - a);
    let out_x = chosen.iter().flat_map(|p| (0..width).map(move |c| lerp(x[p.parent * width + c], x[p.neighbour * width + c], p.u))).collect();
    let out_y = chosen.iter().map(|p| lerp(y[p.parent], y[p.neighbour], p.u)).collect();
    (out_x, out_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..400).flat_map(|i| [f64::from(i % 20), 1000.0 * f64::from(i / 20)]).collect();
        let y: Vec<f64> = x.chunks(2).map(|r| 2.0 * r[0] - r[1] / 1000.0).collect();
        (x, y)
    }

    #[test]
    fn a_row_lies_on_the_segment_between_its_parents() {
        let (x, y) = data();
        let chosen = pairs(&x, 2, 100, 7012);
        let (sx, sy) = rows(&x, &y, 2, 100, 7012);
        assert_eq!((sx.len(), sy.len()), (200, 100));
        for (i, p) in chosen.iter().enumerate() {
            assert!((0.0..1.0).contains(&p.u) && p.parent != p.neighbour);
            for c in 0..2 {
                let (a, b) = (x[p.parent * 2 + c], x[p.neighbour * 2 + c]);
                assert!((sx[i * 2 + c] - (a + p.u * (b - a))).abs() < 1e-12);
                assert!(sx[i * 2 + c] >= a.min(b) && sx[i * 2 + c] <= a.max(b));
            }
            // The target is linear here, so the interpolated target is exact.
            assert!((sy[i] - (2.0 * sx[i * 2] - sx[i * 2 + 1] / 1000.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn neighbours_are_near_in_scaled_inputs_whatever_the_units() {
        // Column 1 is 1000x column 0's units: unscaled, only column 1 would count.
        let (x, _) = data();
        for p in pairs(&x, 2, 200, 7012) {
            let step = |c: usize, unit: f64| ((x[p.parent * 2 + c] - x[p.neighbour * 2 + c]) / unit).abs();
            assert!(step(0, 1.0) <= 2.0 && step(1, 1000.0) <= 2.0, "{p:?}");
        }
    }

    #[test]
    fn it_is_deterministic_in_the_seed_and_safe_on_tiny_data() {
        let (x, y) = data();
        assert_eq!(rows(&x, &y, 2, 50, 7012), rows(&x, &y, 2, 50, 7012));
        assert_ne!(rows(&x, &y, 2, 50, 7012), rows(&x, &y, 2, 50, 7013));
        assert!(pairs(&[1.0, 2.0], 2, 5, 1).is_empty());
        assert_eq!(pairs(&[1.0, 2.0, 3.0, 4.0], 2, 5, 1).len(), 5);
    }
}
