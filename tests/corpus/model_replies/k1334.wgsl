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
    let r = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * 256u * nwg.y + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    let s0 = buffer_0_global[base];
    let s1 = buffer_0_global[base + 1u];
    let s2 = buffer_0_global[base + 2u];
    let s3 = buffer_0_global[base + 3u];
    let s4 = buffer_0_global[base + 4u];
    let s5 = buffer_0_global[base + 5u];
    let s6 = buffer_0_global[base + 6u];
    let s7 = buffer_0_global[base + 7u];
    let s8 = buffer_0_global[base + 8u];
    let s9 = buffer_0_global[base + 9u];
    let x0 = buffer_1_global[s0];
    let x1 = buffer_1_global[s1];
    let x2 = buffer_1_global[s2];
    let x3 = buffer_1_global[s3];
    let x4 = buffer_1_global[s4];
    let x5 = buffer_1_global[s5];
    let x6 = buffer_1_global[s6];
    let x7 = buffer_1_global[s7];
    let x8 = buffer_1_global[s8];
    let x9 = buffer_1_global[s9];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];
    let w0 = buffer_2_global[fmix(h1 ^ (k2 ^ 0u)) >> 22u];
    let w1 = buffer_2_global[fmix(h1 ^ (k2 ^ 1u)) >> 22u];
    let w2 = buffer_2_global[fmix(h1 ^ (k2 ^ 2u)) >> 22u];
    let w3 = buffer_2_global[fmix(h1 ^ (k2 ^ 3u)) >> 22u];
    let w4 = buffer_2_global[fmix(h1 ^ (k2 ^ 4u)) >> 22u];
    let w5 = buffer_2_global[fmix(h1 ^ (k2 ^ 5u)) >> 22u];
    let w6 = buffer_2_global[fmix(h1 ^ (k2 ^ 6u)) >> 22u];
    let w7 = buffer_2_global[fmix(h1 ^ (k2 ^ 7u)) >> 22u];
    let w8 = buffer_2_global[fmix(h1 ^ (k2 ^ 8u)) >> 22u];
    let w9 = buffer_2_global[fmix(h1 ^ (k2 ^ 9u)) >> 22u];

    var sum = 0.0;
    var c = 0.0;
    var z = w0 * x0 - c;
    var t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w1 * x1 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w2 * x2 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w3 * x3 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w4 * x4 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w5 * x5 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w6 * x6 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w7 * x7 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w8 * x8 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;
    z = w9 * x9 - c;
    t = sum + z;
    c = (t - sum) - z;
    sum = t;

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

    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let nz = buffer_2_global[nh >> 22u];
    let pre = ((buffer_7_global[1u] * sum + buffer_7_global[0u] * ax) + buffer_3_global[r]) + buffer_7_global[4u] * nz;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
