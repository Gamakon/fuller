enable-free;
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

fn fmix32(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // each live workgroup covers 2 * 256 rows; the upper half of the grid exits (uniform per workgroup)
    if (wid.x * 2u >= nwg.x) {
        return;
    }
    let t = lid.x;
    wg_cb[t] = buffer_2_global[t];
    wg_cb[t + 256u] = buffer_2_global[t + 256u];
    wg_cb[t + 512u] = buffer_2_global[t + 512u];
    wg_cb[t + 768u] = buffer_2_global[t + 768u];
    workgroupBarrier();

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
    let one_minus_leak = 1.0 - leak;

    for (var j = 0u; j < 2u; j = j + 1u) {
        let id = wid.x * 512u + j * 256u + t;
        let r = id + off0;
        if (r < width) {
            let h1 = fmix32(key_lo ^ r);
            let base = r * 10u;
            var acc_z = 0.0;
            var comp = 0.0;
            var acc_x = 0.0;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let x = buffer_1_global[buffer_0_global[base + s]];
                let hh = fmix32(h1 ^ (key_hi ^ s));
                let w = wg_cb[hh >> 22u];
                let term = w * x - comp;
                let nsum = acc_z + term;
                comp = (nsum - acc_z) - term;
                acc_z = nsum;
                acc_x = acc_x + x;
            }
            let hn = fmix32(fmix32(noise_lo ^ r) ^ noise_tail);
            let z = wg_cb[hn >> 22u];
            let pre = scale * acc_z + loc * acc_x + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = one_minus_leak * buffer_1_global[r] + leak * th;
        }
    }
}
