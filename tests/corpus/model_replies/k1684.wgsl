@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> src_s: array<u32, 2560>;

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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let n = buffer_5_global[0u];
    let r0 = wid.x * 256u + buffer_6_global[0u];
    let total = n * 10u;
    let ebase = r0 * 10u;
    for (var k = 0u; k < 10u; k += 1u) {
        let j = k * 256u + lidx;
        let e = ebase + j;
        if (e < total) {
            src_s[j] = buffer_0_global[e];
        }
    }
    workgroupBarrier();
    let r = r0 + lidx;
    if (r >= n) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);
    let sb = lidx * 10u;

    var p: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        p[s] = buffer_1_global[src_s[sb + s]];
    }
    var cb: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        cb[s] = buffer_2_global[fmix(h0 ^ (key_hi ^ s)) >> 22u];
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let nt = buffer_6_global[3u];
    let z = buffer_2_global[fmix(fmix(nlo ^ r) ^ (nhi ^ nt)) >> 22u];

    var sum = 0.0;
    var c = 0.0;
    var ax = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let v = p[s];
        let t0 = cb[s] * v - c;
        let t1 = sum + t0;
        c = (t1 - sum) - t0;
        sum = t1;
        ax = ax + v;
    }
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = scale * sum + loc * ax + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
