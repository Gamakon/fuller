@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    if (lidx >= 32u) {
        return;
    }
    let n = buffer_5_global[0u];
    let base = wid.x * 128u + buffer_6_global[0u] + lidx;
    if (base >= n) {
        return;
    }
    let rows = vec4<u32>(base, base + 32u, base + 64u, base + 96u);
    let valid = rows < vec4<u32>(n);
    let rr = select(vec4<u32>(base), rows, valid);

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix4(vec4<u32>(key_lo) ^ rr);

    var acc = vec4<f32>(0.0);
    var comp = vec4<f32>(0.0);
    var accx = vec4<f32>(0.0);
    let rb = rr * vec4<u32>(10u);

    for (var s: u32 = 0u; s < 10u; s++) {
        let src = vec4<u32>(
            buffer_0_global[rb.x + s],
            buffer_0_global[rb.y + s],
            buffer_0_global[rb.z + s],
            buffer_0_global[rb.w + s]
        );
        let xv = vec4<f32>(
            buffer_1_global[src.x],
            buffer_1_global[src.y],
            buffer_1_global[src.z],
            buffer_1_global[src.w]
        );
        let hh = fmix4(h1 ^ vec4<u32>(key_hi ^ s)) >> vec4<u32>(22u);
        let w = vec4<f32>(
            buffer_2_global[hh.x],
            buffer_2_global[hh.y],
            buffer_2_global[hh.z],
            buffer_2_global[hh.w]
        );
        let term = w * xv;
        let y = term - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + xv;
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let step = buffer_6_global[3u];
    let nh = fmix4(fmix4(vec4<u32>(nlo) ^ rr) ^ vec4<u32>(nhi ^ step)) >> vec4<u32>(22u);
    let z = vec4<f32>(
        buffer_2_global[nh.x],
        buffer_2_global[nh.y],
        buffer_2_global[nh.z],
        buffer_2_global[nh.w]
    );

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let om = 1.0 - leak;

    let drive = vec4<f32>(
        buffer_3_global[rr.x],
        buffer_3_global[rr.y],
        buffer_3_global[rr.z],
        buffer_3_global[rr.w]
    );
    let st = vec4<f32>(
        buffer_1_global[rr.x],
        buffer_1_global[rr.y],
        buffer_1_global[rr.z],
        buffer_1_global[rr.w]
    );
    let pre = scale * acc + loc * accx + drive + sigma * z;
    let o = om * st + leak * vec4<f32>(
        safe_tanh(pre.x),
        safe_tanh(pre.y),
        safe_tanh(pre.z),
        safe_tanh(pre.w)
    );
    buffer_4_global[rows.x] = o.x;
    if (valid.y) { buffer_4_global[rows.y] = o.y; }
    if (valid.z) { buffer_4_global[rows.z] = o.z; }
    if (valid.w) { buffer_4_global[rows.w] = o.w; }
}

fn fmix4(h: vec4<u32>) -> vec4<u32> {
    var x = h;
    x = x ^ (x >> vec4<u32>(16u));
    x = x * vec4<u32>(2246822507u);
    x = x ^ (x >> vec4<u32>(13u));
    x = x * vec4<u32>(3266489909u);
    x = x ^ (x >> vec4<u32>(16u));
    return x;
}

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}
