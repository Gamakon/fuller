// THE SCATTER FORM of UltraDim's centroid update, as a module of five entries UltraDim can dispatch in order: A6's
// stable counting sort by label (qdrant lib/projection-voting-kmeans/src/lib.rs: block_histogram,
// cluster_offsets, scatter_ids, gpu_group_ids lines 2072-2190) and its segmented sum (mean_update, pass B), with A7's
// semantics (gpu_kmeans.rs update_centroids): each cluster's points summed with Kahan compensation in ascending point
// order (the counting sort is stable, so a cluster's ids ascend, the order A7 scans them in: the same arithmetic, so the
// same bits), divided by the count, an empty cluster's centroid not written (it keeps the old one).
// Bindings 0-3 are A7's; binding 4 is the module's scratch: [block counts nb*k][block prefixes nb*k][lens k][starts k]
// [grouped ids n], nb = ceil(n / 256). Dispatches (problems/kmeans_update_segmented/problem.toml), in order:
//   hist      ceil(nb * k / 256) workgroups     one invocation per (block, cluster): its count in the block
//   offsets   ceil(k / 256)                     one per cluster: each block's exclusive prefix, the cluster's length
//   starts    1                                 invocation 0: each cluster's first slot (exclusive prefix of lens)
//   scatter   ceil(n / 256)                     one per point: its slot, starts + block prefix + rank in the block
//   mean      ceil(k * d / 256)                 one per (cluster, dim) cell: A7's Kahan sum over the cluster's ids
@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn hist(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.x;
    let k = params.y;
    let nb = (n + 255u) / 256u;
    let t = global_id.x;
    if (t >= nb * k) { return; }
    let b = t / k;
    let c = t % k;
    var cnt: u32 = 0u;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        let v = b * 256u + i;
        if (v < n) {
            if (assignments[v] == c) { cnt = cnt + 1u; }
        }
    }
    scratch[t] = cnt;
}

@compute @workgroup_size(256, 1, 1)
fn offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.x;
    let k = params.y;
    let nb = (n + 255u) / 256u;
    let c = global_id.x;
    if (c >= k) { return; }
    var run: u32 = 0u;
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        scratch[nb * k + b * k + c] = run;
        run = run + scratch[b * k + c];
    }
    scratch[2u * nb * k + c] = run;
}

@compute @workgroup_size(256, 1, 1)
fn starts(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.x;
    let k = params.y;
    let nb = (n + 255u) / 256u;
    if (global_id.x != 0u) { return; }
    var s: u32 = 0u;
    for (var c: u32 = 0u; c < k; c = c + 1u) {
        scratch[2u * nb * k + k + c] = s;
        s = s + scratch[2u * nb * k + c];
    }
}

@compute @workgroup_size(256, 1, 1)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.x;
    let k = params.y;
    let nb = (n + 255u) / 256u;
    let v = global_id.x;
    if (v >= n) { return; }
    let a = assignments[v];
    let b = v / 256u;
    var rank: u32 = 0u;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        let u = b * 256u + i;
        if (u < v) {
            if (assignments[u] == a) { rank = rank + 1u; }
        }
    }
    let slot = scratch[2u * nb * k + k + a] + scratch[nb * k + b * k + a] + rank;
    scratch[2u * nb * k + 2u * k + slot] = v;
}

@compute @workgroup_size(256, 1, 1)
fn mean(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.x;
    let k = params.y;
    let d = params.z;
    let nb = (n + 255u) / 256u;
    let t = global_id.x;
    if (t >= k * d) { return; }
    let c = t / d;
    let j = t % d;
    let len = scratch[2u * nb * k + c];
    let first = scratch[2u * nb * k + k + c];
    var sum: f32 = 0.0;
    var sum_c: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        if (i >= len) { break; }
        let v = scratch[2u * nb * k + 2u * k + first + i];
        let term = vectors[v * d + j];
        let y = term - sum_c;
        let s2 = sum + y;
        sum_c = (s2 - sum) - y;
        sum = s2;
    }
    if (len > 0u) {
        centroids[t] = sum / f32(len);
    }
}
