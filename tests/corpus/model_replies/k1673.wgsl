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

fn term(h1: u32, kx: u32, v: f32) -> f32 {
    return cb[fmix(h1 ^ kx) >> 22u] * v;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 4u; k += 1u) {
        let j = k * 256u + lidx;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let id = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * nwg.y * 256u;
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
    let khi = buffer_5_global[2u];

    var sum = 0.0;
    var c = 0.0;
    var y = term(h1, khi ^ 0u, v0) - c;
    var t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 1u, v1) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 2u, v2) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 3u, v3) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 4u, v4) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 5u, v5) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 6u, v6) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 7u, v7) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 8u, v8) - c;
    t = sum + y; c = (t - sum) - y; sum = t;
    y = term(h1, khi ^ 9u, v9) - c;
    t = sum + y; c = (t - sum) - y; sum = t;

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

    let hn = fmix(buffer_6_global[1u] ^ r);
    let z = cb[fmix(hn ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = (((scale * sum) + (loc * ax)) + dr) + (sigma * z);
    let th = select(tanh(pre), 1.0, pre > 43.0);
    buffer_4_global[r] = ((1.0 - leak) * xr) + (leak * th);
}
