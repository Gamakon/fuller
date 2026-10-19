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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    // stage codebook
    for (var i = 0u; i < 4u; i += 1u) {
        let j = lidx + i * 256u;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    // each workgroup covers 256 rows; the first 128 threads take two rows each
    if (lidx >= 128u) {
        return;
    }
    let wflat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let nt = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = nhi ^ nt;

    let base = wflat * 256u + lidx + off;
    let r0 = base;
    let r1 = base + 128u;
    let ok0 = r0 < n;
    let ok1 = r1 < n;

    var h0 = fmix(key_lo ^ r0);
    var h1 = fmix(key_lo ^ r1);

    var s0 = 0.0; var c0 = 0.0; var x0 = 0.0;
    var s1 = 0.0; var c1 = 0.0; var x1 = 0.0;

    let b0 = r0 * 10u;
    let b1 = r1 * 10u;

    for (var s = 0u; s < 10u; s += 1u) {
        let k = key_hi ^ s;
        let w0 = cb[fmix(h0 ^ k) >> 22u];
        let w1 = cb[fmix(h1 ^ k) >> 22u];
        var v0 = 0.0;
        var v1 = 0.0;
        if (ok0) { v0 = buffer_1_global[buffer_0_global[b0 + s]]; }
        if (ok1) { v1 = buffer_1_global[buffer_0_global[b1 + s]]; }
        let p0 = w0 * v0 - c0;
        let t0 = s0 + p0;
        c0 = (t0 - s0) - p0;
        s0 = t0;
        x0 = x0 + v0;
        let p1 = w1 * v1 - c1;
        let t1 = s1 + p1;
        c1 = (t1 - s1) - p1;
        s1 = t1;
        x1 = x1 + v1;
    }

    if (ok0) {
        let z = cb[fmix(fmix(nlo ^ r0) ^ nkey) >> 22u];
        let pre = scale * s0 + loc * x0 + buffer_3_global[r0] + sigma * z;
        buffer_4_global[r0] = (1.0 - leak) * buffer_1_global[r0] + leak * safe_tanh(pre);
    }
    if (ok1) {
        let z = cb[fmix(fmix(nlo ^ r1) ^ nkey) >> 22u];
        let pre = scale * s1 + loc * x1 + buffer_3_global[r1] + sigma * z;
        buffer_4_global[r1] = (1.0 - leak) * buffer_1_global[r1] + leak * safe_tanh(pre);
    }
}
