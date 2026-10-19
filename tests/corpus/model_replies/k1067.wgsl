@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let q = (nwg.x + 3u) / 4u;
    if (wid.x >= q) {
        return;
    }
    let stride = q * 64u;
    let width = buffer_5_global[0u];
    let base = buffer_6_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let tt = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = n_hi ^ tt;
    for (var j = 0u; j < 4u; j += 1u) {
        let r = gid.x + j * stride + base;
        if (r < width) {
            let h1 = fmix(key_lo ^ r);
            var acc = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            let rb = r * 10u;
            for (var s = 0u; s < 10u; s += 1u) {
                let x = buffer_1_global[buffer_0_global[rb + s]];
                let c = buffer_2_global[fmix(h1 ^ (key_hi ^ s)) >> 22u];
                let y = c * x - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                ax += x;
            }
            let z = buffer_2_global[fmix(fmix(n_lo ^ r) ^ nkey) >> 22u];
            let pre = scale * acc + loc * ax + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (!(pre > 43.0)) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h ^= h >> 16u;
    h *= 2246822507u;
    h ^= h >> 13u;
    h *= 3266489909u;
    h ^= h >> 16u;
    return h;
}

@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;
