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

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let l = lid.x;
    for (var k = 0u; k < 1024u; k += 128u) {
        let j = k + l;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    if (l >= 32u) {
        return;
    }
    let wg = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
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
    let nkey = n_hi ^ n_t;
    let base = wg * 128u + l;

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + j * 32u + off;
        if (r < width) {
            let hr = fmix(key_lo ^ r);
            let rb = r * 10u;
            var sa = 0.0;
            var c = 0.0;
            var sx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let x = buffer_1_global[src];
                let w = cb[fmix(hr ^ (key_hi ^ s)) >> 22u];
                let y = w * x - c;
                let t = sa + y;
                c = (t - sa) - y;
                sa = t;
                sx = sx + x;
            }
            let z = cb[fmix(fmix(n_lo ^ r) ^ nkey) >> 22u];
            let pre = scale * sa + loc * sx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
