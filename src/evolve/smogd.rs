//! SMOGD — Synthetic Minority Over-sampling with Generated Data — the Rust of
//! `ManifoldGridSampler` (`sampler.py`: `recursive_grid_split`,
//! `generate_synthetic_points_smogd`, step 8 of `synthetic_adaptive_grid_split`).
//!
//! The rows are embedded in 2D, the plane is cut into density-adaptive cells, and
//! in each cell new points are PLACED at random, kept only when no real or
//! synthetic point is within `too_close`. A placed point's features and target
//! are drawn from N(mean, variance) of its `k` nearest real rows of the cell,
//! inverse-distance weighted in 2D. No formula is used: the rows come from the
//! data alone. They are noisy on purpose — they rank individuals in the
//! tournaments (HFF's third block); they never decide that a fit is exact.
//!
//! Randomness is the engine's counter-based `draw`, so the rows are a function
//! of (data, embedding, seed). numpy's generator cannot be matched draw for
//! draw; the grid and the weights are deterministic and match the Python.

use super::draw;

pub const STREAM_SMOGD_PLACE: u32 = 310;
pub const STREAM_SMOGD_NOISE: u32 = 311;

/// `ManifoldGridSampler`'s defaults.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub min_points_per_cell: usize,
    pub max_depth: u32,
    pub synth_multiplier: f64,
    pub too_close: f64,
    pub k_neighbors: usize,
    pub max_attempts: u32,
}

impl Params {
    pub fn defaults() -> Params {
        Params { min_points_per_cell: 10, max_depth: 8, synth_multiplier: 2.0, too_close: 0.3, k_neighbors: 5, max_attempts: 50 }
    }
}

/// `recursive_grid_split`: the cells, each a list of row indices. A box's upper
/// edges are open and a sub-cell's box is recomputed from its own rows, as in the
/// Python, so the rows on a box's maximum x or y belong to no cell.
pub fn grid(points: &[[f64; 2]], min_points: usize, max_depth: u32) -> Vec<Vec<usize>> {
    let all: Vec<usize> = (0..points.len()).collect();
    split(points, &all, min_points, max_depth, 0)
}

fn split(points: &[[f64; 2]], rows: &[usize], min_points: usize, max_depth: u32, depth: u32) -> Vec<Vec<usize>> {
    if rows.is_empty() {
        return Vec::new();
    }
    if rows.len() < min_points || depth >= max_depth {
        return vec![rows.to_vec()];
    }
    let span = |axis: usize| rows.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &r| (lo.min(points[r][axis]), hi.max(points[r][axis])));
    let ((x_min, x_max), (y_min, y_max)) = (span(0), span(1));
    let (x_mid, y_mid) = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0);
    let mut cells = Vec::new();
    for (x_lo, x_hi) in [(x_min, x_mid), (x_mid, x_max)] {
        for (y_lo, y_hi) in [(y_min, y_mid), (y_mid, y_max)] {
            let inside: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&r| points[r][0] >= x_lo && points[r][0] < x_hi && points[r][1] >= y_lo && points[r][1] < y_hi)
                .collect();
            if inside.len() > min_points * 4 {
                cells.extend(split(points, &inside, min_points, max_depth, depth + 1));
            } else if !inside.is_empty() {
                cells.push(inside);
            }
        }
    }
    cells
}

/// A uniform f64 in [0, 1) from one draw.
pub(crate) fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// A standard normal from two draws (Box–Muller).
fn normal(a: u64, b: u64) -> f64 {
    let u = (1.0 - unit(a)).max(f64::MIN_POSITIVE);
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * unit(b)).cos()
}

/// numpy's `isclose` at its default tolerances.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// The inverse-distance weights of the `k` nearest of `dists` (ties to the lower
/// index): `(index, weight)`, weights summing to 1.
pub fn idw(dists: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut order: Vec<usize> = (0..dists.len()).collect();
    order.sort_by(|&a, &b| dists[a].total_cmp(&dists[b]).then(a.cmp(&b)));
    order.truncate(k);
    let inverse: Vec<f64> = order.iter().map(|&i| 1.0 / dists[i].max(1e-12)).collect();
    let total: f64 = inverse.iter().sum();
    order.into_iter().zip(inverse).map(|(i, w)| (i, w / total)).collect()
}

/// The synthetic rows: `(x, y)`, `x` row-major with `width` columns. `x`/`y` are
/// the real rows, `embedding` their 2D positions.
pub fn rows(x: &[f64], y: &[f64], width: usize, embedding: &[[f64; 2]], seed: u32, p: &Params) -> (Vec<f64>, Vec<f64>) {
    let (mut out_x, mut out_y) = (Vec::new(), Vec::new());
    for (cell_at, cell) in grid(embedding, p.min_points_per_cell, p.max_depth).iter().enumerate() {
        let wanted = (cell.len() as f64 * p.synth_multiplier) as usize;
        if cell.len() < 2 || wanted == 0 {
            continue;
        }
        let span = |axis: usize| cell.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &r| (lo.min(embedding[r][axis]), hi.max(embedding[r][axis])));
        let ((x_min, x_max), (y_min, y_max)) = (span(0), span(1));
        if close(x_min, x_max) && close(y_min, y_max) {
            continue;
        }
        let mut placed: Vec<[f64; 2]> = Vec::new();
        let mut counter = 0u32;
        for _ in 0..wanted {
            for _ in 0..p.max_attempts {
                let at = [
                    x_min + (x_max - x_min) * unit(draw(seed, 0, cell_at as u32, counter, STREAM_SMOGD_PLACE)),
                    y_min + (y_max - y_min) * unit(draw(seed, 0, cell_at as u32, counter + 1, STREAM_SMOGD_PLACE)),
                ];
                counter += 2;
                let far = |q: &[f64; 2]| ((q[0] - at[0]).powi(2) + (q[1] - at[1]).powi(2)).sqrt();
                let dists: Vec<f64> = cell.iter().map(|&r| far(&embedding[r])).collect();
                if dists.iter().any(|&d| d < p.too_close) || placed.iter().any(|q| far(q) < p.too_close) {
                    continue;
                }
                let near = idw(&dists, p.k_neighbors);
                let mut noise_at = placed.len() as u32 * 2 * (width as u32 + 1);
                let mut sample = |value: &dyn Fn(usize) -> f64| {
                    let mean: f64 = near.iter().map(|&(i, w)| w * value(cell[i])).sum();
                    let variance: f64 = near.iter().map(|&(i, w)| w * (value(cell[i]) - mean).powi(2)).sum();
                    let r = normal(draw(seed, 0, cell_at as u32, noise_at, STREAM_SMOGD_NOISE), draw(seed, 0, cell_at as u32, noise_at + 1, STREAM_SMOGD_NOISE));
                    noise_at += 2;
                    r * variance.sqrt() + mean
                };
                for c in 0..width {
                    out_x.push(sample(&|r| x[r * width + c]));
                }
                out_y.push(sample(&|r| y[r]));
                placed.push(at);
                break;
            }
        }
    }
    (out_x, out_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lattice(side: usize, step: f64) -> Vec<[f64; 2]> {
        (0..side * side).map(|i| [(i % side) as f64 * step, (i / side) as f64 * step]).collect()
    }

    #[test]
    fn no_row_is_in_two_cells_and_big_cells_are_split() {
        // The Python recomputes a sub-cell's box from its own rows and keeps the
        // upper edges open, so at every level the rows on a box's maximum fall
        // out (about 3% of 5,000 uniform rows). That is ported as it is.
        let points = lattice(20, 1.0);
        let cells = grid(&points, 10, 8);
        let mut seen = vec![0u32; points.len()];
        for r in cells.iter().flatten() {
            seen[*r] += 1;
        }
        assert!(seen.iter().all(|&n| n <= 1));
        assert!(seen[0] == 1, "the lower corner is always inside");
        assert!(seen[points.len() - 1] == 0, "the upper corner is on two open edges");
        assert!(cells.iter().all(|c| c.len() <= 40), "a cell above 4 x min_points was not split");
    }

    #[test]
    fn the_weights_are_inverse_distance_over_the_k_nearest() {
        let near = idw(&[4.0, 1.0, 2.0, 8.0], 2);
        assert_eq!(near.iter().map(|n| n.0).collect::<Vec<_>>(), vec![1, 2]);
        assert!((near[0].1 - 2.0 / 3.0).abs() < 1e-12 && (near[1].1 - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rows_keep_their_distance_and_stay_near_their_neighbours() {
        // A sparse lattice (spacing 1.0 > 2 x too_close) leaves room between rows.
        let points = lattice(12, 1.0);
        let x: Vec<f64> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
        let y: Vec<f64> = points.iter().map(|p| 3.0 * p[0] - p[1]).collect();
        let (sx, sy) = rows(&x, &y, 2, &points, 7012, &Params::defaults());
        assert!(sy.len() > 20, "only {} rows placed", sy.len());
        assert_eq!(sx.len(), sy.len() * 2);
        assert!(sx.iter().chain(&sy).all(|v| v.is_finite()));
        // A linear target over a lattice: the noisy rows still follow the plane.
        let worst = sx.chunks(2).zip(&sy).map(|(p, t)| (3.0 * p[0] - p[1] - t).abs()).fold(0.0, f64::max);
        assert!(worst < 8.0, "a synthetic target is {worst} from the plane");
    }

    #[test]
    fn it_is_deterministic_in_the_seed() {
        let points = lattice(12, 1.0);
        let x: Vec<f64> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
        let y: Vec<f64> = points.iter().map(|p| p[0] * p[1]).collect();
        let a = rows(&x, &y, 2, &points, 7012, &Params::defaults());
        assert_eq!(a, rows(&x, &y, 2, &points, 7012, &Params::defaults()));
        assert_ne!(a, rows(&x, &y, 2, &points, 7013, &Params::defaults()));
    }
}
