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

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 1024u; k += 256u) {
        wg_a[k + lidx] = buffer_2_global[k + lidx];
    }
    workgroupBarrier();

    // each workgroup covers 256 rows; the lower 128 threads take 2 adjacent rows each
    if (lidx >= 128u) {
        return;
    }
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off0 = buffer_6_global[0u];
    let noise_lo = buffer_6_global[1u];
    let noise_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let base = (wid.y * nwg.x + wid.x) * 256u + off0 + lidx * 2u;
    let ra = base;
    let rb = base + 1u;
    let va = ra < width;
    let vb = rb < width;

    var za = 0.0; var ca = 0.0; var xa = 0.0;
    var zb = 0.0; var cb = 0.0; var xb = 0.0;
    var ha = 0u; var hb = 0u;
    if (va) { ha = fmix(key_lo ^ ra); }
    if (vb) { hb = fmix(key_lo ^ rb); }

    for (var s = 0u; s < 10u; s += 1u) {
        let t = key_hi ^ s;
        if (va) {
            let src = buffer_0_global[ra * 10u + s];
            let v = buffer_1_global[src];
            let w = wg_a[fmix(ha ^ t) >> 22u];
            let y = w * v - ca;
            let n = za + y;
            ca = (n - za) - y;
            za = n;
            xa += v;
        }
        if (vb) {
            let src = buffer_0_global[rb * 10u + s];
            let v = buffer_1_global[src];
            let w = wg_a[fmix(hb ^ t) >> 22u];
            let y = w * v - cb;
            let n = zb + y;
            cb = (n - zb) - y;
            zb = n;
            xb += v;
        }
    }

    if (va) {
        let z = wg_a[fmix(fmix(noise_lo ^ ra) ^ noise_tail) >> 22u];
        let pre = scale * za + loc * xa + buffer_3_global[ra] + sigma * z;
        buffer_4_global[ra] = (1.0 - leak) * buffer_1_global[ra] + leak * safe_tanh(pre);
    }
    if (vb) {
        let z = wg_a[fmix(fmix(noise_lo ^ rb) ^ noise_tail) >> 22u];
        let pre = scale * zb + loc * xb + buffer_3_global[rb] + sigma * z;
        buffer_4_global[rb] = (1.0 - leak) * buffer_1_global[rb] + leak * safe_tanh(pre);
    }
}
