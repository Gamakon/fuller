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

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute
@workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = (global_id.z * num_workgroups.x * 256u * num_workgroups.y) + (global_id.y * num_workgroups.x * 256u) + global_id.x;
    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let base = r * 10u;
    let k1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var sum = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s: u32 = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let x = buffer_1_global[src];
        let w = buffer_2_global[fmix(k1 ^ (k2 ^ s)) >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }
    let o = buffer_6_global[2u] ^ buffer_6_global[3u];
    let z = buffer_2_global[fmix(fmix(buffer_6_global[1u] ^ r) ^ o) >> 22u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1f - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
