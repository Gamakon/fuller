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

const ROWS = 4u;

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let base = wid.x * 256u * ROWS;
    // uniform per workgroup: whole group has no rows
    if (base + off >= width) {
        return;
    }
    for (var k = 0u; k < 4u; k = k + 1u) {
        wg_a[lidx + k * 256u] = buffer_2_global[lidx + k * 256u];
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
    let omleak = 1.0 - leak;

    for (var j = 0u; j < ROWS; j = j + 1u) {
        let r = off + base + lidx * ROWS + j;
        if (r < width) {
            let h0 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let src = buffer_0_global[rb + s];
                let xv = buffer_1_global[src];
                let hh = fmix(h0 ^ (key_hi ^ s));
                let w = wg_a[(hh >> 22u) & 1023u];
                let term = w * xv;
                let y = term - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                accx = accx + xv;
            }
            let nh = fmix(fmix(noise_lo ^ r) ^ nk);
            let z = wg_a[(nh >> 22u) & 1023u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
