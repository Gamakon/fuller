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

fn row_acc(r: u32, key_lo: u32, key_hi: u32) -> vec2<f32> {
    let hr = fmix(key_lo ^ r);
    let base = r * 10u;
    var s = 0.0;
    var c = 0.0;
    var ax = 0.0;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let src = buffer_0_global[base + k];
        let x = buffer_1_global[src];
        let h = fmix(hr ^ (key_hi ^ k));
        let y = wg_cb[h >> 22u] * x - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
        ax = ax + x;
    }
    return vec2<f32>(s, ax);
}

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = 0u; i < 1024u; i += 128u) {
        let j = i + lid.x;
        wg_cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    if (lid.x >= 64u) {
        return;
    }
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off0 = buffer_6_global[0u];
    let noise_lo = buffer_6_global[1u];
    let noise_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let wgi = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let r0 = wgi * 128u + lid.x * 2u + off0;
    let r1 = r0 + 1u;
    let keep = 1.0 - leak;
    if (r0 < width) {
        let a = row_acc(r0, key_lo, key_hi);
        let z = wg_cb[fmix(fmix(noise_lo ^ r0) ^ noise_tail) >> 22u];
        let pre = scale * a.x + loc * a.y + buffer_3_global[r0] + sigma * z;
        buffer_4_global[r0] = keep * buffer_1_global[r0] + leak * safe_tanh(pre);
    }
    if (r1 < width) {
        let a = row_acc(r1, key_lo, key_hi);
        let z = wg_cb[fmix(fmix(noise_lo ^ r1) ^ noise_tail) >> 22u];
        let pre = scale * a.x + loc * a.y + buffer_3_global[r1] + sigma * z;
        buffer_4_global[r1] = keep * buffer_1_global[r1] + leak * safe_tanh(pre);
    }
}
