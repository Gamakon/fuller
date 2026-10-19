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
    let r = (gid.x + ((gid.y * nwg.x) * 256u) + (((gid.z * nwg.x) * 256u) * nwg.y)) + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    let x0 = buffer_1_global[buffer_0_global[base]];
    let x1 = buffer_1_global[buffer_0_global[base + 1u]];
    let x2 = buffer_1_global[buffer_0_global[base + 2u]];
    let x3 = buffer_1_global[buffer_0_global[base + 3u]];
    let x4 = buffer_1_global[buffer_0_global[base + 4u]];
    let x5 = buffer_1_global[buffer_0_global[base + 5u]];
    let x6 = buffer_1_global[buffer_0_global[base + 6u]];
    let x7 = buffer_1_global[buffer_0_global[base + 7u]];
    let x8 = buffer_1_global[buffer_0_global[base + 8u]];
    let x9 = buffer_1_global[buffer_0_global[base + 9u]];

    let h = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];
    let c0 = buffer_2_global[fmix(h ^ (k2 ^ 0u)) >> 22u];
    let c1 = buffer_2_global[fmix(h ^ (k2 ^ 1u)) >> 22u];
    let c2 = buffer_2_global[fmix(h ^ (k2 ^ 2u)) >> 22u];
    let c3 = buffer_2_global[fmix(h ^ (k2 ^ 3u)) >> 22u];
    let c4 = buffer_2_global[fmix(h ^ (k2 ^ 4u)) >> 22u];
    let c5 = buffer_2_global[fmix(h ^ (k2 ^ 5u)) >> 22u];
    let c6 = buffer_2_global[fmix(h ^ (k2 ^ 6u)) >> 22u];
    let c7 = buffer_2_global[fmix(h ^ (k2 ^ 7u)) >> 22u];
    let c8 = buffer_2_global[fmix(h ^ (k2 ^ 8u)) >> 22u];
    let c9 = buffer_2_global[fmix(h ^ (k2 ^ 9u)) >> 22u];

    var s: f32 = 0.0;
    var c: f32 = 0.0;
    var y: f32;
    var t: f32;
    y = c0 * x0 - c; t = s + y; c = (t - s) - y; s = t;
    y = c1 * x1 - c; t = s + y; c = (t - s) - y; s = t;
    y = c2 * x2 - c; t = s + y; c = (t - s) - y; s = t;
    y = c3 * x3 - c; t = s + y; c = (t - s) - y; s = t;
    y = c4 * x4 - c; t = s + y; c = (t - s) - y; s = t;
    y = c5 * x5 - c; t = s + y; c = (t - s) - y; s = t;
    y = c6 * x6 - c; t = s + y; c = (t - s) - y; s = t;
    y = c7 * x7 - c; t = s + y; c = (t - s) - y; s = t;
    y = c8 * x8 - c; t = s + y; c = (t - s) - y; s = t;
    y = c9 * x9 - c; t = s + y; c = (t - s) - y; s = t;

    var ax: f32 = 0.0;
    ax = ax + x0;
    ax = ax + x1;
    ax = ax + x2;
    ax = ax + x3;
    ax = ax + x4;
    ax = ax + x5;
    ax = ax + x6;
    ax = ax + x7;
    ax = ax + x8;
    ax = ax + x9;

    let hn = fmix(buffer_6_global[1u] ^ r);
    let zi = fmix(hn ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u;
    let z = buffer_2_global[zi];
    let leak = buffer_7_global[2u];
    let pre = ((buffer_7_global[1u] * s + buffer_7_global[0u] * ax) + buffer_3_global[r]) + buffer_7_global[4u] * z;
    var th: f32 = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
