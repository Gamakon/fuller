@compute @workgroup_size(64, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let t = wid.x * 64u + lid;
    let base = t * 4u;
    if base + off >= width {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let keep = 1.0 - leak;

    for (var k: u32 = 0u; k < 4u; k++) {
        let r = base + k + off;
        if r >= width {
            break;
        }
        let h0 = fmix32(key_lo ^ r);
        let rb = r * 10u;
        var sum = 0.0;
        var comp = 0.0;
        var accx = 0.0;
        for (var s: u32 = 0u; s < 10u; s++) {
            let x = buffer_1_global[buffer_0_global[rb + s]];
            let h = fmix32(h0 ^ (key_hi ^ s));
            let w = buffer_2_global[h >> 22u];
            let y = w * x - comp;
            let tt = sum + y;
            comp = (tt - sum) - y;
            sum = tt;
            accx += x;
        }
        let zh = fmix32(fmix32(n_lo ^ r) ^ n_tail);
        let z = buffer_2_global[zh >> 22u];
        let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
        buffer_4_global[r] = keep * buffer_1_global[r] + leak * safe_tanh(pre);
    }
}

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
