// grid 256 1 1 1
@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    f0 = -3.4e38f;
    for (t = 0u; t < min((params.y), 16777216u); t = t + 1u) {
    f1 = 0.0f;
    i0 = 0u;
    for (k = 0u; k < min((params.z), 16777216u); k = k + 1u) {
    f1 = (f1 + (point_keys[(((gid.x * params.z) + k))] * centroid_keys[(((t * params.z) + k))]));
    }
    if ((f1 > f0)) {
    f0 = f1;
    i1 = t;
    }
    }
    if ((!(gid.x >= params.x))) {
    assignments[((params.w + gid.x))] = (i1);
    }
}
