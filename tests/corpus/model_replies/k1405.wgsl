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

fn kahan(sc: vec2<f32>, p: f32) -> vec2<f32> {
    let y = p - sc.y;
    let t = sc.x + y;
    let c = (t - sc.x) - y;
    return vec2<f32>(t, c);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * 256u * nwg.y;
    let r = id + buffer_6_global[0u];
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
    let kh = buffer_5_global[2u];

    let w0 = buffer_2_global[fmix(h1 ^ (kh ^ 0u)) >> 22u];
    let w1 = buffer_2_global[fmix(h1 ^ (kh ^ 1u)) >> 22u];
    let w2 = buffer_2_global[fmix(h1 ^ (kh ^ 2u)) >> 22u];
    let w3 = buffer_2_global[fmix(h1 ^ (kh ^ 3u)) >> 22u];
    let w4 = buffer_2_global[fmix(h1 ^ (kh ^ 4u)) >> 22u];
    let w5 = buffer_2_global[fmix(h1 ^ (kh ^ 5u)) >> 22u];
    let w6 = buffer_2_global[fmix(h1 ^ (kh ^ 6u)) >> 22u];
    let w7 = buffer_2_global[fmix(h1 ^ (kh ^ 7u)) >> 22u];
    let w8 = buffer_2_global[fmix(h1 ^ (kh ^ 8u)) >> 22u];
    let w9 = buffer_2_global[fmix(h1 ^ (kh ^ 9u)) >> 22u];

    var sc = vec2<f32>(0.0, 0.0);
    sc = kahan(sc, w0 * x0);
    sc = kahan(sc, w1 * x1);
    sc = kahan(sc, w2 * x2);
    sc = kahan(sc, w3 * x3);
    sc = kahan(sc, w4 * x4);
    sc = kahan(sc, w5 * x5);
    sc = kahan(sc, w6 * x6);
    sc = kahan(sc, w7 * x7);
    sc = kahan(sc, w8 * x8);
    sc = kahan(sc, w9 * x9);

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

    let n1 = fmix(buffer_6_global[1u] ^ r);
    let n2 = fmix(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[n2 >> 22u];

    let pre = buffer_7_global[1u] * sc.x + buffer_7_global[0u] * ax + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
