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

@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // each active workgroup (every 4th) covers 256 rows, 4 per thread, coalesced
    if ((wid.x & 3u) != 0u) {
        return;
    }
    for (var k = 0u; k < 16u; k += 1u) {
        let i = k * 64u + lid.x;
        wg_cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let n_lo = buffer_6_global[1u];
    let n_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let base = wid.x * 64u + lid.x;

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + j * 64u + off;
        if (r < width) {
            let rb = r * 10u;
            let h0 = fmix(key_lo ^ r);
            var acc = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let x = buffer_1_global[src];
                let h = fmix(h0 ^ (key_hi ^ s));
                let t = wg_cb[h >> 22u] * x - comp;
                let ns = acc + t;
                comp = (ns - acc) - t;
                acc = ns;
                ax = ax + x;
            }
            let hn = fmix(fmix(n_lo ^ r) ^ n_tail);
            let z = wg_cb[hn >> 22u];
            let pre = scale * acc + loc * ax + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
