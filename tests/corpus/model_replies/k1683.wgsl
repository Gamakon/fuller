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

fn cb(m: u32, s: u32) -> f32 {
    return buffer_2_global[fmix(m ^ s) >> 22u];
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x + buffer_6_global[0u];
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
    let xr = buffer_1_global[r];
    let dr = buffer_3_global[r];

    let m = fmix(buffer_5_global[1u] ^ r) ^ buffer_5_global[2u];
    let w0 = cb(m, 0u);
    let w1 = cb(m, 1u);
    let w2 = cb(m, 2u);
    let w3 = cb(m, 3u);
    let w4 = cb(m, 4u);
    let w5 = cb(m, 5u);
    let w6 = cb(m, 6u);
    let w7 = cb(m, 7u);
    let w8 = cb(m, 8u);
    let w9 = cb(m, 9u);

    let zn = buffer_2_global[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    var sum = 0.0;
    var c = 0.0;
    var p: f32;
    var y: f32;
    var t: f32;
    p = w0 * x0; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w1 * x1; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w2 * x2; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w3 * x3; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w4 * x4; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w5 * x5; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w6 * x6; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w7 * x7; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w8 * x8; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;
    p = w9 * x9; y = p - c; t = sum + y; c = (t - sum) - y; sum = t;

    var ax = 0.0;
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

    let leak = buffer_7_global[2u];
    let pre = ((buffer_7_global[1u] * sum + buffer_7_global[0u] * ax) + dr) + buffer_7_global[4u] * zn;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * xr + leak * th;
}
