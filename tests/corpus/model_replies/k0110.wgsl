@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_a: array<f32, 1024>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn cbk(h: u32) -> f32 {
    return wg_a[(h >> 22u) & 1023u];
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var t = 0u; t < 1024u; t += 256u) {
        wg_a[t + lid.x] = buffer_2_global[t + lid.x];
    }
    workgroupBarrier();

    let r = ((gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x) + buffer_6_global[0u];
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

    let k1 = buffer_5_global[2u];
    let hr = fmix(buffer_5_global[1u] ^ r);

    var sum = 0.0;
    var c = 0.0;
    var y = 0.0;
    var tt = 0.0;

    y = cbk(fmix(hr ^ (k1 ^ 0u))) * x0 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 1u))) * x1 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 2u))) * x2 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 3u))) * x3 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 4u))) * x4 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 5u))) * x5 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 6u))) * x6 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 7u))) * x7 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 8u))) * x8 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;
    y = cbk(fmix(hr ^ (k1 ^ 9u))) * x9 - c; tt = sum + y; c = (tt - sum) - y; sum = tt;

    var accx = 0.0;
    accx = accx + x0;
    accx = accx + x1;
    accx = accx + x2;
    accx = accx + x3;
    accx = accx + x4;
    accx = accx + x5;
    accx = accx + x6;
    accx = accx + x7;
    accx = accx + x8;
    accx = accx + x9;

    let hn = fmix(buffer_6_global[1u] ^ r);
    let z = cbk(fmix(hn ^ (buffer_6_global[2u] ^ buffer_6_global[3u])));

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = ((scale * sum) + (loc * accx) + buffer_3_global[r]) + (sigma * z);
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = ((1.0 - leak) * buffer_1_global[r]) + (leak * th);
}
