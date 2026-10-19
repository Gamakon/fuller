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

@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let width = buffer_5_global[0u];
    let base = wid.x * 256u;
    let off = buffer_6_global[0u];
    // uniform early exit: workgroup covers rows [base, base+256)
    if (base + off >= width) {
        return;
    }
    for (var t = 0u; t < 1024u; t += 64u) {
        wg_a[t + lidx] = buffer_2_global[t + lidx];
    }
    workgroupBarrier();

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

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + j * 64u + lidx + off;
        if (r < width) {
            let h1 = fmix(key_lo ^ r);
            let sb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[sb + s];
                let xv = buffer_1_global[src];
                let hh = fmix(h1 ^ (key_hi ^ s));
                let term = wg_a[hh >> 22u] * xv;
                let y = term - comp;
                let tt = sum + y;
                comp = (tt - sum) - y;
                sum = tt;
                accx = accx + xv;
            }
            let hn = fmix(fmix(n_lo ^ r) ^ nk);
            let z = wg_a[hn >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
