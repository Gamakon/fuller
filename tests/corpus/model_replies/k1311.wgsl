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

fn prod(h1: u32, k2: u32, s: u32, v: f32) -> f32 {
    return buffer_2_global[fmix(h1 ^ (k2 ^ s)) >> 22u] * v;
}

fn kst(sc: vec2<f32>, p: f32) -> vec2<f32> {
    let y = p - sc.y;
    let t = sc.x + y;
    let c = (t - sc.x) - y;
    return vec2<f32>(t, c);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
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
    let v0 = buffer_1_global[s0];
    let v1 = buffer_1_global[s1];
    let v2 = buffer_1_global[s2];
    let v3 = buffer_1_global[s3];
    let v4 = buffer_1_global[s4];
    let v5 = buffer_1_global[s5];
    let v6 = buffer_1_global[s6];
    let v7 = buffer_1_global[s7];
    let v8 = buffer_1_global[s8];
    let v9 = buffer_1_global[s9];
    let xr = buffer_1_global[r];
    let dr = buffer_3_global[r];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var sc = vec2<f32>(0.0, 0.0);
    sc = kst(sc, prod(h1, k2, 0u, v0));
    sc = kst(sc, prod(h1, k2, 1u, v1));
    sc = kst(sc, prod(h1, k2, 2u, v2));
    sc = kst(sc, prod(h1, k2, 3u, v3));
    sc = kst(sc, prod(h1, k2, 4u, v4));
    sc = kst(sc, prod(h1, k2, 5u, v5));
    sc = kst(sc, prod(h1, k2, 6u, v6));
    sc = kst(sc, prod(h1, k2, 7u, v7));
    sc = kst(sc, prod(h1, k2, 8u, v8));
    sc = kst(sc, prod(h1, k2, 9u, v9));

    var ax = 0.0;
    ax = ax + v0;
    ax = ax + v1;
    ax = ax + v2;
    ax = ax + v3;
    ax = ax + v4;
    ax = ax + v5;
    ax = ax + v6;
    ax = ax + v7;
    ax = ax + v8;
    ax = ax + v9;

    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[nh >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = ((scale * sc.x) + (loc * ax)) + dr + (sigma * z);
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = ((1.0 - leak) * xr) + (leak * th);
}
