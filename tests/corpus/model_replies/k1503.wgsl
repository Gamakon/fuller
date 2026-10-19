@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(h: u32) -> u32 {
    var x = h;
    x = x ^ (x >> 16u);
    x = x * 2246822507u;
    x = x ^ (x >> 13u);
    x = x * 3266489909u;
    x = x ^ (x >> 16u);
    return x;
}

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x;
    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(buffer_5_global[1u] ^ r);
    let base = r * 10u;
    var acc_z = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let v = buffer_1_global[src];
        let hh = fmix32(h0 ^ (key_hi ^ s));
        let w = buffer_2_global[hh >> 22u];
        let y = w * v - comp;
        let t = acc_z + y;
        comp = (t - acc_z) - y;
        acc_z = t;
        acc_x = acc_x + v;
    }
    let zh = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[zh >> 22u];
    let pre = buffer_7_global[1u] * acc_z + buffer_7_global[0u] * acc_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
