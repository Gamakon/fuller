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

fn fmix(h: u32) -> u32 {
    var x = h;
    x = x ^ (x >> 16u);
    x = x * 2246822507u;
    x = x ^ (x >> 13u);
    x = x * 3266489909u;
    x = x ^ (x >> 16u);
    return x;
}

fn term(h1: u32, k2: u32, s: u32, st: f32) -> f32 {
    return cb[fmix(h1 ^ (k2 ^ s)) >> 22u] * st;
}

// state = (sum, compensation)
fn kahan(a: vec2<f32>, v: f32) -> vec2<f32> {
    let y = v - a.y;
    let t = a.x + y;
    let c = (t - a.x) - y;
    return vec2<f32>(t, c);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    cb[li] = buffer_2_global[li];
    cb[li + 256u] = buffer_2_global[li + 256u];
    cb[li + 512u] = buffer_2_global[li + 512u];
    cb[li + 768u] = buffer_2_global[li + 768u];
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let k1 = buffer_5_global[1u];
    let k2 = buffer_5_global[2u];
    let base = r * 10u;

    let e0 = buffer_0_global[base];
    let e1 = buffer_0_global[base + 1u];
    let e2 = buffer_0_global[base + 2u];
    let e3 = buffer_0_global[base + 3u];
    let e4 = buffer_0_global[base + 4u];
    let e5 = buffer_0_global[base + 5u];
    let e6 = buffer_0_global[base + 6u];
    let e7 = buffer_0_global[base + 7u];
    let e8 = buffer_0_global[base + 8u];
    let e9 = buffer_0_global[base + 9u];

    let x0 = buffer_1_global[e0];
    let x1 = buffer_1_global[e1];
    let x2 = buffer_1_global[e2];
    let x3 = buffer_1_global[e3];
    let x4 = buffer_1_global[e4];
    let x5 = buffer_1_global[e5];
    let x6 = buffer_1_global[e6];
    let x7 = buffer_1_global[e7];
    let x8 = buffer_1_global[e8];
    let x9 = buffer_1_global[e9];
    let self_x = buffer_1_global[r];
    let drv = buffer_3_global[r];

    let h1 = fmix(k1 ^ r);

    var a = vec2<f32>(0.0, 0.0);
    a = kahan(a, term(h1, k2, 0u, x0));
    a = kahan(a, term(h1, k2, 1u, x1));
    a = kahan(a, term(h1, k2, 2u, x2));
    a = kahan(a, term(h1, k2, 3u, x3));
    a = kahan(a, term(h1, k2, 4u, x4));
    a = kahan(a, term(h1, k2, 5u, x5));
    a = kahan(a, term(h1, k2, 6u, x6));
    a = kahan(a, term(h1, k2, 7u, x7));
    a = kahan(a, term(h1, k2, 8u, x8));
    a = kahan(a, term(h1, k2, 9u, x9));

    let acc_x = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)) + (x8 + x9);

    let nh = fmix(buffer_6_global[1u] ^ r);
    let z = cb[fmix(nh ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * a.x + loc * acc_x + drv + sigma * z;
    var th: f32 = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * self_x + leak * th;
}
