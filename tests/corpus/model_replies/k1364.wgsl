@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let noise_lo = buffer_6_global[1u];
    let noise_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = noise_hi ^ step_t;
    let base = wid.x * 256u;

    for (var k = 0u; k < 4u; k += 1u) {
        let r = base + k * 64u + lid.x + off;
        if (r < width) {
            let h1 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let xv = buffer_1_global[buffer_0_global[rb + s]];
                let w = buffer_2_global[fmix32(h1 ^ (key_hi ^ s)) >> 22u];
                let y = w * xv - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                accx = accx + xv;
            }
            let z = buffer_2_global[fmix32(fmix32(noise_lo ^ r) ^ nkey) >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
