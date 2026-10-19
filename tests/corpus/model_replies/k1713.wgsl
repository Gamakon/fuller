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

fn fmix32(hin: u32) -> u32 {
    var h = hin;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    // each workgroup covers 4 * 256 consecutive rows; surplus workgroups exit uniformly
    let base = wid.x * 1024u;
    if (base >= width) {
        return;
    }
    for (var k = 0u; k < 4u; k += 1u) {
        let j = k * 256u + lidx;
        wg_cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let noise_lo = buffer_6_global[1u];
    let noise_hi = buffer_6_global[2u];
    let tstep = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nk = noise_hi ^ tstep;
    let oml = 1.0 - leak;

    for (var k = 0u; k < 4u; k += 1u) {
        let id = base + k * 256u + lidx;
        let r = id + off;
        if (r < width) {
            let h1 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var acc = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let x = buffer_1_global[src];
                let hh = fmix32(h1 ^ (key_hi ^ s));
                let w = wg_cb[hh >> 22u];
                let y = w * x - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                accx = accx + x;
            }
            let z = wg_cb[fmix32(fmix32(noise_lo ^ r) ^ nk) >> 22u];
            let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = oml * buffer_1_global[r] + leak * th;
        }
    }
}
