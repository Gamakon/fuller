@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn cbk(h1: u32, k2: u32) -> f32 {
    return buffer_2_global[fmix(h1 ^ k2) >> 22u];
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 256u;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let b = r * 10u;
    let a0 = buffer_0_global[b];
    let a1 = buffer_0_global[b + 1u];
    let a2 = buffer_0_global[b + 2u];
    let a3 = buffer_0_global[b + 3u];
    let a4 = buffer_0_global[b + 4u];
    let a5 = buffer_0_global[b + 5u];
    let a6 = buffer_0_global[b + 6u];
    let a7 = buffer_0_global[b + 7u];
    let a8 = buffer_0_global[b + 8u];
    let a9 = buffer_0_global[b + 9u];
    let x0 = buffer_1_global[a0];
    let x1 = buffer_1_global[a1];
    let x2 = buffer_1_global[a2];
    let x3 = buffer_1_global[a3];
    let x4 = buffer_1_global[a4];
    let x5 = buffer_1_global[a5];
    let x6 = buffer_1_global[a6];
    let x7 = buffer_1_global[a7];
    let x8 = buffer_1_global[a8];
    let x9 = buffer_1_global[a9];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var s = 0.0;
    var c = 0.0;
    var p = cbk(h1, k2 ^ 0u) * x0;
    var y = p - c; var t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 1u) * x1;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 2u) * x2;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 3u) * x3;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 4u) * x4;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 5u) * x5;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 6u) * x6;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 7u) * x7;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 8u) * x8;
    y = p - c; t = s + y; c = (t - s) - y; s = t;
    p = cbk(h1, k2 ^ 9u) * x9;
    y = p - c; t = s + y; c = (t - s) - y; s = t;

    var ax = 0.0;
    ax = ax + x0; ax = ax + x1; ax = ax + x2; ax = ax + x3; ax = ax + x4;
    ax = ax + x5; ax = ax + x6; ax = ax + x7; ax = ax + x8; ax = ax + x9;

    let zh = fmix(buffer_6_global[1u] ^ r);
    let z = cbk(zh, buffer_6_global[2u] ^ buffer_6_global[3u]);
    let pre = buffer_7_global[1u] * s + buffer_7_global[0u] * ax + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
