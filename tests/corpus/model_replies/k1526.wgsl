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

fn fmix32(h0: u32) -> u32 {
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
    // each workgroup covers 256 rows (64 threads x 4 adjacent rows); surplus workgroups exit (uniform)
    let base = wid.x * 256u;
    let width = buffer_5_global[0u];
    let off0 = buffer_6_global[0u];
    if (base + off0 >= width) {
        return;
    }
    for (var k = 0u; k < 1024u; k += 64u) {
        let j = k + lidx;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_x = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let keep = 1.0 - leak;

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + lidx * 4u + j + off0;
        if (r < width) {
            let h1 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let xv = buffer_1_global[buffer_0_global[rb + s]];
                let hh = fmix32(h1 ^ (key_hi ^ s));
                let t = wg_a[hh >> 22u] * xv;
                let y = t - comp;
                let nsum = sum + y;
                comp = (nsum - sum) - y;
                sum = nsum;
                accx = accx + xv;
            }
            let hn = fmix32(fmix32(n_lo ^ r) ^ n_x);
            let z = wg_a[hn >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = keep * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
