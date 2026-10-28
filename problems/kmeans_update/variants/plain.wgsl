// A7 without the Kahan compensation: a plain sum in index order (a tolerance probe, not a candidate).
@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256, 1, 1)
fn update_centroids(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let centroid_dim_idx = global_id.x;
    let n_centroids = params.y;
    let dimensions = params.z;
    let n_vectors = params.x;
    
    if (centroid_dim_idx >= n_centroids * dimensions) { return; }
    
    let centroid_idx = centroid_dim_idx / dimensions;
    let dim_idx = centroid_dim_idx % dimensions;
    
    var sum: f32 = 0.0;
    var count: u32 = 0u;

    for (var v: u32 = 0u; v < n_vectors; v = v + 1u) {
        if (assignments[v] == centroid_idx) {
            sum = sum + vectors[v * dimensions + dim_idx];
            count = count + 1u;
        }
    }
    
    if (count > 0u) {
        centroids[centroid_dim_idx] = sum / f32(count);
    }
}
