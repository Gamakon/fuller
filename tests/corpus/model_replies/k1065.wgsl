@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> cb: array<f32, 1024>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn kstep(a: f32, c: f32, p: f32) -> vec2<f32> {
    let y = p - c;
    let t = a + y;
    return vec2<f32>(t, (t - a) - y);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    for (var i = 0u; i < 4u; i += 1u) {
        let j = lidx + i * 256u;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let b = r * 10u;
    let s0 = buffer_0_global[b];
    let s1 = buffer_0_global[b + 1u];
    let s2 = buffer_0_global[b + 2u];
    let s3 = buffer_0_global[b + 3u];
    let s4 = buffer_0_global[b + 4u];
    let s5 = buffer_0_global[b + 5u];
    let s6 = buffer_0_global[b + 6u];
    let s7 = buffer_0_global[b + 7u];
    let s8 = buffer_0_global[b + 8u];
    let s9 = buffer_0_global[b + 9u];
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
    let xr = buffer_1_global[r];
    let dr = buffer_3_global[r];

    let h1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var v = kstep(0.0, 0.0, cb[fmix(h1 ^ (k2 ^ 0u)) >> 22u] * x0);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 1u)) >> 22u] * x1);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 2u)) >> 22u] * x2);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 3u)) >> 22u] * x3);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 4u)) >> 22u] * x4);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 5u)) >> 22u] * x5);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 6u)) >> 22u] * x6);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 7u)) >> 22u] * x7);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 8u)) >> 22u] * x8);
    v = kstep(v.x, v.y, cb[fmix(h1 ^ (k2 ^ 9u)) >> 22u] * x9);
    let accz = v.x;

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
    let z = cb[nh >> 22u];

    let pre = buffer_7_global[1u] * accz + buffer_7_global[0u] * ax + dr + buffer_7_global[4u] * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * xr + leak * th;
}
