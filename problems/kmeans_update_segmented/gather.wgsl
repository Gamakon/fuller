// THE GATHER in the scatter form's interface (problems/kmeans_update_segmented): UltraDim's A7 kernel unchanged as the
// last entry `mean`, the four counting-sort entries empty. Its outputs are A7's; a P run on this interface starts from
// the gather and may reach the scatter (the counting sort and the segmented sum) as a sequence of kernels.
@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn hist(@builtin(global_invocation_id) global_id: vec3<u32>) {
}

@compute @workgroup_size(256, 1, 1)
fn offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
}

@compute @workgroup_size(256, 1, 1)
fn starts(@builtin(global_invocation_id) global_id: vec3<u32>) {
}

@compute @workgroup_size(256, 1, 1)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
}

@compute @workgroup_size(256, 1, 1)
fn mean(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let centroid_dim_idx = global_id.x;
    let n_centroids = params.y;
    let dimensions = params.z;
    let n_vectors = params.x;
    
    if (centroid_dim_idx >= n_centroids * dimensions) { return; }
    
    let centroid_idx = centroid_dim_idx / dimensions;
    let dim_idx = centroid_dim_idx % dimensions;
    
    var sum: f32 = 0.0;
    var sum_c: f32 = 0.0;  // Kahan compensation
    var count: u32 = 0u;

    for (var v: u32 = 0u; v < n_vectors; v = v + 1u) {
        if (assignments[v] == centroid_idx) {
            let term = vectors[v * dimensions + dim_idx];
            let y = term - sum_c;
            let t = sum + y;
            sum_c = (t - sum) - y;
            sum = t;
            count = count + 1u;
        }
    }
    
    if (count > 0u) {
        centroids[centroid_dim_idx] = sum / f32(count);
    }
}
