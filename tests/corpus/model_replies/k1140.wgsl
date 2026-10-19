var<workgroup> cb: array<f32, 1024>;
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

fn do_row(r: u32, key_lo: u32, key_hi: u32, noise_k: u32, loc: f32, scale: f32, leak: f32, sigma: f32) {
    let h0 = fmix(key_lo ^ r);
    let base = r * 10u;
    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let x = buffer_1_global[buffer_0_global[base + s]];
        let hh = fmix(h0 ^ (key_hi ^ s));
        let term = cb[hh >> 22u] * x;
        let y = term - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + x;
    }
    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ noise_k);
    let z = cb[nh >> 22u];
    let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}

@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let gbase = wid.x * 256u;
    if (gbase + off >= n) {
        return;
    }
    for (var i = 0u; i < 16u; i++) {
        let k = lid.x + i * 64u;
        cb[k] = buffer_2_global[k];
    }
    workgroupBarrier();
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let noise_k = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    for (var j = 0u; j < 4u; j++) {
        let r = gbase + j * 64u + lid.x + off;
        if (r < n) {
            do_row(r, key_lo, key_hi, noise_k, loc, scale, leak, sigma);
        }
    }
}
