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

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    for (var k = 0u; k < 4u; k = k + 1u) {
        let idx = k * 256u + lidx;
        wg_a[idx] = buffer_2_global[idx];
    }
    workgroupBarrier();
    if (lidx >= 64u) {
        return;
    }
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nk = n_hi ^ n_t;
    let base = wid.x * 256u + lidx * 4u + off;

    for (var j = 0u; j < 4u; j = j + 1u) {
        let r = base + j;
        if (r < width) {
            let hr = fmix(key_lo ^ r);
            var sum = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            let rb = r * 10u;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let x = buffer_1_global[buffer_0_global[rb + s]];
                let hv = fmix(hr ^ (key_hi ^ s));
                let y = wg_a[(hv >> 22u) & 1023u] * x - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                ax = ax + x;
            }
            let hn = fmix(fmix(n_lo ^ r) ^ nk);
            let z = wg_a[(hn >> 22u) & 1023u];
            let pre = scale * sum + loc * ax + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
