@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(hin: u32) -> u32 {
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

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // each workgroup covers 256 rows (2 per thread); the surplus workgroups exit
    if (wid.x * 2u >= nwg.x) {
        return;
    }
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let noise_key = buffer_6_global[1u];
    let noise_b = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let base = wid.x * 256u;

    for (var k = 0u; k < 2u; k += 1u) {
        let id = base + k * 128u + lid.x;
        let r = id + off;
        if (r < n) {
            let h0 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let xv = buffer_1_global[src];
                let hh = fmix32(h0 ^ (key_hi ^ s));
                let w = buffer_2_global[hh >> 22u];
                let term = w * xv;
                let y = term - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                accx = accx + xv;
            }
            let nh = fmix32(fmix32(noise_key ^ r) ^ noise_b);
            let z = buffer_2_global[nh >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
