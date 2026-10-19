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

fn do_row(r: u32, key_lo: u32, key_hi: u32, nz: u32, loc: f32, scale: f32, leak: f32, sigma: f32) {
    let k0 = fmix(key_lo ^ r);
    let base = r * 10u;
    var sum = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let st = buffer_1_global[src];
        let h = fmix(k0 ^ (key_hi ^ s));
        let w = cb[h >> 22u];
        let y = w * st - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + st;
    }
    let z = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ nz) >> 22u];
    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}

@compute @workgroup_size(64, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = lid.x;
    for (var i = 0u; i < 16u; i++) {
        let idx = t + i * 64u;
        cb[idx] = buffer_2_global[idx];
    }
    workgroupBarrier();
    if t >= 32u {
        return;
    }
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nz = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let r0 = wg.x * 64u + t * 2u + off;
    if r0 < width {
        do_row(r0, key_lo, key_hi, nz, loc, scale, leak, sigma);
    }
    let r1 = r0 + 1u;
    if r1 < width {
        do_row(r1, key_lo, key_hi, nz, loc, scale, leak, sigma);
    }
}
