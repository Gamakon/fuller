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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let RPT = 4u;
    // each active workgroup covers 256*RPT rows; surplus workgroups exit (uniform)
    if (wid.x * 256u * RPT >= buffer_5_global[0u] + 0u && wid.x * 256u * RPT >= buffer_5_global[0u]) {
        return;
    }
    for (var k = 0u; k < 4u; k += 1u) {
        wg_a[k * 256u + lidx] = buffer_2_global[k * 256u + lidx];
    }
    workgroupBarrier();

    let off = buffer_6_global[0u];
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let tt = buffer_6_global[3u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let c_loc = buffer_7_global[0u];
    let c_scale = buffer_7_global[1u];
    let c_leak = buffer_7_global[2u];
    let c_sigma = buffer_7_global[4u];
    let nz = n_hi ^ tt;
    let base = wid.x * 256u * RPT;

    for (var j = 0u; j < RPT; j += 1u) {
        let r = base + j * 256u + lidx + off;
        if (r < width) {
            let h0 = fmix(key_lo ^ r);
            let b = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let xv = buffer_1_global[buffer_0_global[b + s]];
                let hh = fmix(h0 ^ (key_hi ^ s));
                let term = wg_a[hh >> 22u] * xv;
                let y = term - comp;
                let tsum = sum + y;
                comp = (tsum - sum) - y;
                sum = tsum;
                accx = accx + xv;
            }
            let z = wg_a[fmix(fmix(n_lo ^ r) ^ nz) >> 22u];
            let pre = c_scale * sum + c_loc * accx + buffer_3_global[r] + c_sigma * z;
            let st = buffer_1_global[r];
            buffer_4_global[r] = (1.0 - c_leak) * st + c_leak * safe_tanh(pre);
        }
    }
}
