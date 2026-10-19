var<workgroup> wg_cb: array<f32, 1024>;

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
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    for (var q = 0u; q < 4u; q++) {
        let i = lidx + q * 256u;
        wg_cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let wgi = wid.y * nwg.x + wid.x;
    let base = wgi * 1024u;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if base + off >= width {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_xor = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let one_m_leak = 1.0 - leak;

    for (var j = 0u; j < 4u; j++) {
        let r = base + j * 256u + lidx + off;
        if r < width {
            let h1 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var si: array<u32, 10>;
            var sv: array<f32, 10>;
            for (var s = 0u; s < 10u; s++) {
                si[s] = buffer_0_global[rb + s];
            }
            for (var s = 0u; s < 10u; s++) {
                sv[s] = buffer_1_global[si[s]];
            }
            var acc = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s++) {
                let hh = fmix(h1 ^ (key_hi ^ s));
                let v = wg_cb[hh >> 22u] * sv[s];
                let y = v - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                accx = accx + sv[s];
            }
            let zh = fmix(fmix(n_lo ^ r) ^ n_xor);
            let z = wg_cb[zh >> 22u];
            let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = one_m_leak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
