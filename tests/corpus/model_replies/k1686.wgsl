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

fn prod(h: u32, k2: u32, s: u32, x: f32) -> f32 {
    return buffer_2_global[fmix(h ^ (k2 ^ s)) >> 22u] * x;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 256u;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    let i0 = buffer_0_global[base];
    let i1 = buffer_0_global[base + 1u];
    let i2 = buffer_0_global[base + 2u];
    let i3 = buffer_0_global[base + 3u];
    let i4 = buffer_0_global[base + 4u];
    let i5 = buffer_0_global[base + 5u];
    let i6 = buffer_0_global[base + 6u];
    let i7 = buffer_0_global[base + 7u];
    let i8 = buffer_0_global[base + 8u];
    let i9 = buffer_0_global[base + 9u];
    let x0 = buffer_1_global[i0];
    let x1 = buffer_1_global[i1];
    let x2 = buffer_1_global[i2];
    let x3 = buffer_1_global[i3];
    let x4 = buffer_1_global[i4];
    let x5 = buffer_1_global[i5];
    let x6 = buffer_1_global[i6];
    let x7 = buffer_1_global[i7];
    let x8 = buffer_1_global[i8];
    let x9 = buffer_1_global[i9];
    let h = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var sum = 0f;
    var c = 0f;
    var y: f32;
    var t: f32;

    y = prod(h, k2, 0u, x0) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 1u, x1) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 2u, x2) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 3u, x3) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 4u, x4) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 5u, x5) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 6u, x6) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 7u, x7) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 8u, x8) - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = prod(h, k2, 9u, x9) - c; t = sum + y; c = (t - sum) - y; sum = t;

    var ax = 0f;
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

    let nh = fmix(buffer_6_global[1u] ^ r);
    let z = buffer_2_global[fmix(nh ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let pre = ((buffer_7_global[1u] * sum + buffer_7_global[0u] * ax) + buffer_3_global[r]) + buffer_7_global[4u] * z;
    var th = 1.0f;
    if (pre <= 43.0f) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0f - leak) * buffer_1_global[r] + leak * th;
}
