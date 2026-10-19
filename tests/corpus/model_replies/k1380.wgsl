@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

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
    if (x > 43.0) {
        return 1.0;
    }
    return tanh(x);
}

fn row_step(r: u32, key_lo: u32, key_hi: u32, loc: f32, scale: f32, leak: f32, sigma: f32, noise_h: u32) {
    let hr = fmix(key_lo ^ r);
    let base = r * 10u;
    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let src = buffer_0_global[base + s];
        let xv = buffer_1_global[src];
        let h = fmix(hr ^ (key_hi ^ s));
        let w = buffer_2_global[h >> 22u];
        let term = w * xv;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xv;
    }
    let zh = fmix(fmix(buffer_6_global[1u] ^ r) ^ noise_h);
    let z = buffer_2_global[zh >> 22u];
    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x >= 32u) {
        return;
    }
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let noise_h = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let id0 = wid.x * 128u + lid.x * 4u;
    for (var k = 0u; k < 4u; k += 1u) {
        let r = id0 + k + off;
        if (r < n) {
            row_step(r, key_lo, key_hi, loc, scale, leak, sigma, noise_h);
        }
    }
}
