@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = gid.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (r >= width) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;
    // the first hash round is the same for all ten slots: fused out of the loop
    let h1 = fmix(key_lo ^ r);

    var sum_z: f32 = 0.0;
    var comp: f32 = 0.0;
    var sum_x: f32 = 0.0;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let x = buffer_1_global[buffer_0_global[base + k]];
        let h = fmix(h1 ^ (key_hi ^ k));
        let term = buffer_2_global[h >> 22u] * x - comp;
        let t = sum_z + term;
        comp = (t - sum_z) - term;
        sum_z = t;
        sum_x = sum_x + x;
    }

    let n_h = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[n_h >> 22u];
    let pre = buffer_7_global[1u] * sum_z + buffer_7_global[0u] * sum_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th: f32 = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
