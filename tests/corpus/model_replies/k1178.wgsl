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

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let l = lid.x;
    cb[l] = buffer_2_global[l];
    cb[l + 256u] = buffer_2_global[l + 256u];
    cb[l + 512u] = buffer_2_global[l + 512u];
    cb[l + 768u] = buffer_2_global[l + 768u];
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let base = r * 10u;
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let kh = buffer_5_global[2u];

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

    let p0 = cb[fmix(h1 ^ (kh ^ 0u)) >> 22u] * x0;
    let p1 = cb[fmix(h1 ^ (kh ^ 1u)) >> 22u] * x1;
    let p2 = cb[fmix(h1 ^ (kh ^ 2u)) >> 22u] * x2;
    let p3 = cb[fmix(h1 ^ (kh ^ 3u)) >> 22u] * x3;
    let p4 = cb[fmix(h1 ^ (kh ^ 4u)) >> 22u] * x4;
    let p5 = cb[fmix(h1 ^ (kh ^ 5u)) >> 22u] * x5;
    let p6 = cb[fmix(h1 ^ (kh ^ 6u)) >> 22u] * x6;
    let p7 = cb[fmix(h1 ^ (kh ^ 7u)) >> 22u] * x7;
    let p8 = cb[fmix(h1 ^ (kh ^ 8u)) >> 22u] * x8;
    let p9 = cb[fmix(h1 ^ (kh ^ 9u)) >> 22u] * x9;

    var sum = 0.0;
    var c = 0.0;
    var y = p0 - c; var t = sum + y; c = (t - sum) - y; sum = t;
    y = p1 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p2 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p3 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p4 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p5 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p6 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p7 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p8 - c; t = sum + y; c = (t - sum) - y; sum = t;
    y = p9 - c; t = sum + y; c = (t - sum) - y; sum = t;

    var ax = 0.0;
    ax = ax + x0; ax = ax + x1; ax = ax + x2; ax = ax + x3; ax = ax + x4;
    ax = ax + x5; ax = ax + x6; ax = ax + x7; ax = ax + x8; ax = ax + x9;

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * ax + buffer_3_global[r] + buffer_7_global[4u] * nz;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
