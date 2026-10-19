@compute @workgroup_size(128, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    if (lid.x >= 64u) {
        return;
    }
    let wg = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let noise_lo = buffer_6_global[1u];
    let noise_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = noise_hi ^ step_t;

    let base = wg * 128u + lid.x * 2u + off;

    for (var k = 0u; k < 2u; k += 1u) {
        let r = base + k;
        if (r < width) {
            let h0 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0f;
            var comp = 0f;
            var acc_x = 0f;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let xv = buffer_1_global[src];
                let hh = fmix32(h0 ^ (key_hi ^ s));
                let w = buffer_2_global[hh >> 22u];
                let y = w * xv - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                acc_x = acc_x + xv;
            }
            let zh = fmix32(fmix32(noise_lo ^ r) ^ nkey);
            let z = buffer_2_global[zh >> 22u];
            let pre = scale * sum + loc * acc_x + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = (1f - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
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
