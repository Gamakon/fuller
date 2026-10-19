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

fn do_row(r: u32, key_lo: u32, key_hi: u32, noise: f32, loc: f32, scale: f32, leak: f32, sigma: f32) {
    let hr = fmix(key_lo ^ r);
    let base = r * 10u;
    var sum = 0.0;
    var c = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let x = buffer_1_global[buffer_0_global[base + s]];
        let h = fmix(hr ^ (key_hi ^ s));
        let t = wg_a[(h >> 22u) & 1023u] * x - c;
        let n = sum + t;
        c = (n - sum) - t;
        sum = n;
        accx = accx + x;
    }
    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * noise;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 8u; k = k + 1u) {
        let j = lidx + k * 128u;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    if (lidx >= 64u) {
        return;
    }
    let wflat = wid.y * nwg.x + wid.x;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nlo = buffer_6_global[1u];
    let nx = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let r0 = wflat * 128u + lidx;
    for (var q = 0u; q < 2u; q = q + 1u) {
        let r = r0 + q * 64u + off;
        if (r < width) {
            let nz = wg_a[(fmix(fmix(nlo ^ r) ^ nx) >> 22u) & 1023u];
            do_row(r, key_lo, key_hi, nz, loc, scale, leak, sigma);
        }
    }
}
