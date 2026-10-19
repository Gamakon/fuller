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
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let base_off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let one_m_leak = 1.0 - leak;

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base_off + wg * 1024u + j * 256u + lid.x;
        if (r < width) {
            let h1 = fmix(key_lo ^ r);
            let b = r * 10u;
            var acc = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[b + s];
                let xv = buffer_1_global[src];
                let hh = fmix(h1 ^ (key_hi ^ s));
                let term = buffer_2_global[hh >> 22u] * xv;
                let y = term - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                accx = accx + xv;
            }
            let hn = fmix(fmix(n_lo ^ r) ^ n_tail);
            let z = buffer_2_global[hn >> 22u];
            let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = one_m_leak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
