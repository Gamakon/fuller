//! THE K-MEANS HARNESS OF THE RESIDENT-SAMPLE PATH (fuller `futhark_sig::kmeans_converged`, the `k` command of
//! `host.rs`): UltraDim's assignment and its Euclidean inertia, the two host-side steps of UltraDim's loop, in plain
//! Rust over `std` alone. The text is the harness fuller runs on the c backend (`kmeans_converged::assign`,
//! `euclidean_inertia`), so the GPU path's stopping rule reads the same bits: fuller's test
//! `the_runtimes_harness_is_fullers` includes this file by path and checks both functions against fuller's on the
//! dbpedia rows.

/// UltraDim's `compute_distances` + `assign_points`: each point's nearest centroid by Euclidean distance (the sum of
/// squared differences in order, in `f32`, then `sqrt`), the first centroid on a tie (`distance < best_distance`
/// replaces only a strictly smaller one; a NaN distance never replaces). The points are assigned in parallel over
/// chunks of points, each point's sums in order, so the result is the sequential one bit for bit.
pub fn assign(points: &[f32], d: usize, centroids: &[Vec<f32>]) -> Vec<usize> {
    let n = points.len() / d.max(1);
    let one = |i: usize| {
        let p = &points[i * d..(i + 1) * d];
        let dist = |c: &[f32]| {
            let mut s = 0f32;
            for t in 0..d {
                s += (p[t] - c[t]) * (p[t] - c[t]);
            }
            s.sqrt()
        };
        let mut best = 0usize;
        let mut best_distance = centroids.first().map_or(f32::INFINITY, |c| dist(c));
        for (j, c) in centroids.iter().enumerate().skip(1) {
            let x = dist(c);
            if x < best_distance {
                best_distance = x;
                best = j;
            }
        }
        best
    };
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).clamp(1, 64);
    let chunk = n.div_ceil(threads).max(1);
    let mut out = vec![0usize; n];
    std::thread::scope(|s| {
        for (c, slot) in out.chunks_mut(chunk).enumerate() {
            let first = c * chunk;
            s.spawn(move || {
                for (j, a) in slot.iter_mut().enumerate() {
                    *a = one(first + j);
                }
            });
        }
    });
    out
}

/// UltraDim's `calculate_inertia` (the stopping rule's number): the sum over points of the squared EUCLIDEAN distance
/// to the assigned centroid, in `f32`, in point order, an assignment out of range contributing nothing.
pub fn euclidean_inertia(points: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d: usize) -> f32 {
    let n = points.len() / d.max(1);
    let mut inertia = 0f32;
    for i in 0..n {
        let a = assignments[i];
        if a < centroids.len() {
            let v = &points[i * d..(i + 1) * d];
            let dist_sq = v.iter().zip(centroids[a].iter()).map(|(v, c)| (v - c).powi(2)).sum::<f32>();
            inertia += dist_sq;
        }
    }
    inertia
}
