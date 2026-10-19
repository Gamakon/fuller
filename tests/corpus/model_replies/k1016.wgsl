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
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let base = wg * 256u + off;
    if (base >= n) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nlo = buffer_6_global[1u];
    let nk = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let omleak = 1.0 - leak;

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + j * 64u + lid.x;
        if (r < n) {
            let hk = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var acc = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let x = buffer_1_global[buffer_0_global[rb + s]];
                let h = fmix32(hk ^ (key_hi ^ s));
                let t = buffer_2_global[h >> 22u] * x - comp;
                let ns = acc + t;
                comp = (ns - acc) - t;
                acc = ns;
                ax = ax + x;
            }
            let z = buffer_2_global[fmix32(fmix32(nlo ^ r) ^ nk) >> 22u];
            let pre = scale * acc + loc * ax + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (!(pre > 43.0)) {
                th = tanh(pre);
            }
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * th;
        }
    }
}
