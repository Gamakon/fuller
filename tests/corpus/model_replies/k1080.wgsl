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
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let nz = fmix(fmix(buffer_6_global[1u] ^ 0u) ^ 0u); // placeholder, recomputed per row below
    let nlo = buffer_6_global[1u];
    let nk = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let omleak = 1.0 - leak;
    let base = wid.x * 256u + lid.x;
    for (var j = 0u; j < 4u; j += 1u) {
        let id = base + j * 64u;
        let r = id + off;
        if (r < width) {
            let hr = fmix(key_lo ^ r);
            let b = r * 10u;
            var acc = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[b + s];
                let st = buffer_1_global[src];
                let hh = fmix(hr ^ (key_hi ^ s)) >> 22u;
                let y = buffer_2_global[hh] * st - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                ax = ax + st;
            }
            let z = buffer_2_global[fmix(fmix(nlo ^ r) ^ nk) >> 22u];
            let pre = scale * acc + loc * ax + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
