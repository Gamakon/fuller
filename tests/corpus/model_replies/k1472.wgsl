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

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * 256u * nwg.y;
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
    let nz = noise_hi ^ step_t;
    let q = (width + 3u) / 4u;
    if g >= q {
        return;
    }
    for (var j = 0u; j < 4u; j += 1u) {
        let r = g + j * q + off;
        if r >= width {
            continue;
        }
        let base = r * 10u;
        let hk = fmix32(key_lo ^ r);
        var acc = 0.0;
        var comp = 0.0;
        var accx = 0.0;
        for (var s = 0u; s < 10u; s += 1u) {
            let src = buffer_0_global[base + s];
            let xv = buffer_1_global[src];
            let h = fmix32(hk ^ (key_hi ^ s));
            let term = buffer_2_global[h >> 22u] * xv;
            let y = term - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
            accx = accx + xv;
        }
        let zh = fmix32(fmix32(noise_lo ^ r) ^ nz);
        let z = buffer_2_global[zh >> 22u];
        let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
        buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
    }
}
