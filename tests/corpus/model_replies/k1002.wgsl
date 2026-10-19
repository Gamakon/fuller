@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_cb: array<f32, 1024>;

fn fmix32(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn do_row(r: u32, key_lo: u32, key_hi: u32, noise_lo: u32, noise_hi: u32, tstep: u32,
          loc: f32, scale: f32, leak: f32, sigma: f32) {
    let h0 = fmix32(key_lo ^ r);
    let base = r * 10u;
    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let src = buffer_0_global[base + s];
        let xv = buffer_1_global[src];
        let h = fmix32(h0 ^ (key_hi ^ s));
        let w = wg_cb[(h >> 22u) & 1023u];
        let y = w * xv - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + xv;
    }
    let zn = wg_cb[(fmix32(fmix32(noise_lo ^ r) ^ (noise_hi ^ tstep)) >> 22u) & 1023u];
    let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * zn;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let li = lid.x;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let j = k * 256u + li;
        wg_cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    if (li >= 128u) {
        return;
    }
    let wgl = wid.y * nwg.x + wid.x;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let noise_lo = buffer_6_global[1u];
    let noise_hi = buffer_6_global[2u];
    let tstep = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let r0 = wgl * 256u + li * 2u + off;
    if (r0 < width) {
        do_row(r0, key_lo, key_hi, noise_lo, noise_hi, tstep, loc, scale, leak, sigma);
    }
    let r1 = r0 + 1u;
    if (r1 < width) {
        do_row(r1, key_lo, key_hi, noise_lo, noise_hi, tstep, loc, scale, leak, sigma);
    }
}
