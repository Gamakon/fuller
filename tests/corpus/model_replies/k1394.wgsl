@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(hin: u32) -> u32 {
    var h = hin;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * 256u * nwg.y;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    let h0 = fmix32(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var sum = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let xv = buffer_1_global[buffer_0_global[base + s]];
        let w = buffer_2_global[fmix32(h0 ^ (k2 ^ s)) >> 22u];
        let y = w * xv - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        acc_x = acc_x + xv;
    }

    let hn = fmix32(buffer_6_global[1u] ^ r);
    let z = buffer_2_global[fmix32(hn ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * acc_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
