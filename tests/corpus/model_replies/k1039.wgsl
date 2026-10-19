@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> cb: array<f32, 1024>;

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

@compute @workgroup_size(64, 1, 1)
fn generated_step(
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let lid = lid3.x;
    // stage codebook (1024 entries, 64 threads)
    for (var i = 0u; i < 16u; i += 1u) {
        let j = i * 64u + lid;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let g = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let base = g * 256u + off;
    if base >= n {
        return;
    }

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nlo = buffer_6_global[1u];
    let nxor = buffer_6_global[2u] ^ buffer_6_global[3u];
    let one_m_leak = 1.0 - leak;

    for (var k = 0u; k < 4u; k += 1u) {
        let r = base + k * 64u + lid;
        if r < n {
            let h0 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let xv = buffer_1_global[src];
                let hh = fmix32(h0 ^ (key_hi ^ s));
                let w = cb[hh >> 22u];
                let term = w * xv;
                let y = term - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                accx = accx + xv;
            }
            let z = cb[fmix32(fmix32(nlo ^ r) ^ nxor) >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = one_m_leak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
