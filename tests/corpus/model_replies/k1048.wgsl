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
    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;

    var src: array<u32, 10>;
    for (var k = 0u; k < 10u; k = k + 1u) {
        src[k] = buffer_0_global[base + k];
    }
    var xs: array<f32, 10>;
    for (var k = 0u; k < 10u; k = k + 1u) {
        xs[k] = buffer_1_global[src[k]];
    }
    let h1 = fmix(key_lo ^ r);
    var hs: array<u32, 10>;
    for (var k = 0u; k < 10u; k = k + 1u) {
        hs[k] = fmix(h1 ^ (key_hi ^ k)) >> 22u;
    }
    var ws: array<f32, 10>;
    for (var k = 0u; k < 10u; k = k + 1u) {
        ws[k] = buffer_2_global[hs[k]];
    }

    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let x = xs[k];
        let f4 = ws[k] * x - f2;
        let f5 = f0 + f4;
        f2 = (f5 - f0) - f4;
        f0 = f5;
        f1 = f1 + x;
    }

    let n1 = fmix(buffer_6_global[1u] ^ r);
    let nh = fmix(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u;
    let z = buffer_2_global[nh];

    let pre = ((buffer_7_global[1u] * f0 + buffer_7_global[0u] * f1) + buffer_3_global[r]) + buffer_7_global[4u] * z;
    var th: f32 = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
