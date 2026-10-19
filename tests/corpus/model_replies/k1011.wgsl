@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(h0: u32) -> u32 {
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
    let id = gid.x + gid.y * nwg.x * 256u;
    let r = id + buffer_6_global[0u];
    let n = buffer_5_global[0u];
    if (r >= n) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;
    let h1 = fmix32(key_lo ^ r);

    var src: array<u32, 10>;
    var xs: array<f32, 10>;
    for (var k = 0u; k < 10u; k++) {
        src[k] = buffer_0_global[base + k];
    }
    for (var k = 0u; k < 10u; k++) {
        xs[k] = buffer_1_global[src[k]];
    }

    var s = 0.0f;
    var c = 0.0f;
    var ax = 0.0f;
    for (var k = 0u; k < 10u; k++) {
        let hh = fmix32(h1 ^ (key_hi ^ k));
        let x = xs[k];
        let p = buffer_2_global[hh >> 22u] * x;
        let y = p - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
        ax = ax + x;
    }

    let nz = buffer_2_global[fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let pre = ((buffer_7_global[1u] * s + buffer_7_global[0u] * ax) + buffer_3_global[r]) + buffer_7_global[4u] * nz;
    var th = 1.0f;
    if (pre <= 43.0f) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0f - leak) * buffer_1_global[r] + leak * th;
}
