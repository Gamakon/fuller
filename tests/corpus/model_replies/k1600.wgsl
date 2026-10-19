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
    let xr = buffer_1_global[r];
    let dr = buffer_3_global[r];

    let key_hi = buffer_5_global[2u];
    let h0 = fmix(buffer_5_global[1u] ^ r);

    var s = 0.0;
    var c = 0.0;
    var y = 0.0;
    var t = 0.0;

    y = cb[fmix(h0 ^ (key_hi ^ 0u)) >> 22u] * x0 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 1u)) >> 22u] * x1 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 2u)) >> 22u] * x2 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 3u)) >> 22u] * x3 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 4u)) >> 22u] * x4 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 5u)) >> 22u] * x5 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 6u)) >> 22u] * x6 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 7u)) >> 22u] * x7 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 8u)) >> 22u] * x8 - c; t = s + y; c = (t - s) - y; s = t;
    y = cb[fmix(h0 ^ (key_hi ^ 9u)) >> 22u] * x9 - c; t = s + y; c = (t - s) - y; s = t;

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

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = ((scale * s + loc * ax) + dr) + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * xr + leak * th;
}
