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

fn fm(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    cb[lidx] = buffer_2_global[lidx];
    cb[lidx + 256u] = buffer_2_global[lidx + 256u];
    cb[lidx + 512u] = buffer_2_global[lidx + 512u];
    cb[lidx + 768u] = buffer_2_global[lidx + 768u];
    workgroupBarrier();

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
    let sv = buffer_1_global[r];
    let dv = buffer_3_global[r];

    let h1 = fm(buffer_5_global[1u] ^ r) ^ buffer_5_global[2u];
    let w0 = cb[fm(h1 ^ 0u) >> 22u];
    let w1 = cb[fm(h1 ^ 1u) >> 22u];
    let w2 = cb[fm(h1 ^ 2u) >> 22u];
    let w3 = cb[fm(h1 ^ 3u) >> 22u];
    let w4 = cb[fm(h1 ^ 4u) >> 22u];
    let w5 = cb[fm(h1 ^ 5u) >> 22u];
    let w6 = cb[fm(h1 ^ 6u) >> 22u];
    let w7 = cb[fm(h1 ^ 7u) >> 22u];
    let w8 = cb[fm(h1 ^ 8u) >> 22u];
    let w9 = cb[fm(h1 ^ 9u) >> 22u];

    var s = 0.0;
    var c = 0.0;
    var y = w0 * x0 - c;
    var t = s + y;
    c = (t - s) - y;
    s = t;
    y = w1 * x1 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w2 * x2 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w3 * x3 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w4 * x4 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w5 * x5 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w6 * x6 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w7 * x7 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w8 * x8 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    y = w9 * x9 - c;
    t = s + y;
    c = (t - s) - y;
    s = t;

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

    let zh = fm(fm(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[zh >> 22u];

    let pre = buffer_7_global[1u] * s + buffer_7_global[0u] * ax + dv + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * sv + leak * th;
}
