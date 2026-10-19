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

fn fmix(hin: u32) -> u32 {
    var h = hin;
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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 4u; k = k + 1u) {
        let j = k * 256u + lidx;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let id = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * 256u * nwg.y;
    let base = buffer_6_global[0u];
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

    // each thread handles 4 adjacent rows; surplus threads exit
    if (id >= (width + 3u) / 4u) {
        return;
    }

    for (var q = 0u; q < 4u; q = q + 1u) {
        let r = id * 4u + q + base;
        if (r < width) {
            let h1 = fmix(key_lo ^ r);
            var s_sum = 0.0;
            var c = 0.0;
            var accx = 0.0;
            let rb = r * 10u;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let src = buffer_0_global[rb + s];
                let x = buffer_1_global[src];
                let hh = fmix(h1 ^ (key_hi ^ s));
                let w = wg_a[hh >> 22u];
                let y = w * x - c;
                let t = s_sum + y;
                c = (t - s_sum) - y;
                s_sum = t;
                accx = accx + x;
            }
            let nh = fmix(fmix(n_lo ^ r) ^ nkey);
            let z = wg_a[nh >> 22u];
            let pre = scale * s_sum + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
