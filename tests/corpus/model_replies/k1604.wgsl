@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_cb: array<f32, 1024>;

fn fmix(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    wg_cb[lidx] = buffer_2_global[lidx];
    wg_cb[lidx + 256u] = buffer_2_global[lidx + 256u];
    wg_cb[lidx + 512u] = buffer_2_global[lidx + 512u];
    wg_cb[lidx + 768u] = buffer_2_global[lidx + 768u];
    workgroupBarrier();

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

    let h1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var sum = 0.0;
    var c = 0.0;
    var t: f32;
    var y: f32;
    var n: f32;

    t = wg_cb[fmix(h1 ^ (k2 ^ 0u)) >> 22u] * v0;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 1u)) >> 22u] * v1;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 2u)) >> 22u] * v2;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 3u)) >> 22u] * v3;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 4u)) >> 22u] * v4;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 5u)) >> 22u] * v5;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 6u)) >> 22u] * v6;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 7u)) >> 22u] * v7;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 8u)) >> 22u] * v8;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;
    t = wg_cb[fmix(h1 ^ (k2 ^ 9u)) >> 22u] * v9;
    y = t - c; n = sum + y; c = (n - sum) - y; sum = n;

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

    let nh = fmix(buffer_6_global[1u] ^ r);
    let nz = wg_cb[fmix(nh ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * ax + buffer_3_global[r] + buffer_7_global[4u] * nz;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
