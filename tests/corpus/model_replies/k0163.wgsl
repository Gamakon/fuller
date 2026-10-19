var<workgroup> wg_a: array<f32, 1024>;
@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn do_row(r: u32, key_lo: u32, key_hi: u32, nz: u32, loc: f32, scale: f32, leak: f32, sigma: f32) {
    let h1 = fmix(key_lo ^ r);
    let base = r * 10u;
    var s = 0.0;
    var c = 0.0;
    var ax = 0.0;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let src = buffer_0_global[base + k];
        let x = buffer_1_global[src];
        let h = fmix(h1 ^ (key_hi ^ k));
        let term = wg_a[(h >> 22u) & 1023u] * x;
        let y = term - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
        ax = ax + x;
    }
    let z = wg_a[(nz >> 22u) & 1023u];
    let pre = scale * s + loc * ax + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}

@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = 0u; i < 16u; i = i + 1u) {
        let j = i * 64u + lid.x;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    let g = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let first = g * 256u;
    if (first + off >= width) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nlo = buffer_6_global[1u];
    let nk = buffer_6_global[2u] ^ buffer_6_global[3u];
    for (var j = 0u; j < 4u; j = j + 1u) {
        let r = first + j * 64u + lid.x + off;
        if (r < width) {
            let nz = fmix(fmix(nlo ^ r) ^ nk);
            do_row(r, key_lo, key_hi, nz, loc, scale, leak, sigma);
        }
    }
}
