var<workgroup> wg_a: array<f32, 1024>;
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

fn do_row(r: u32, key_lo: u32, key_hi: u32, nz_lo: u32, nz_hi: u32, step_t: u32,
          loc: f32, scale: f32, leak: f32, sigma: f32) {
    let h0 = fmix(key_lo ^ r);
    let base = r * 10u;
    var src: array<u32, 10>;
    for (var k = 0u; k < 10u; k++) {
        src[k] = buffer_0_global[base + k];
    }
    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var k = 0u; k < 10u; k++) {
        let x = buffer_1_global[src[k]];
        let hh = fmix(h0 ^ (key_hi ^ k));
        let w = wg_a[(hh >> 22u) & 1023u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }
    let nh = fmix(fmix(nz_lo ^ r) ^ (nz_hi ^ step_t));
    let z = wg_a[(nh >> 22u) & 1023u];
    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>,
                  @builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(num_workgroups) nwg: vec3<u32>) {
    let l = lid.x;
    for (var k = 0u; k < 4u; k++) {
        let j = k * 256u + l;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    if l >= 128u {
        return;
    }
    let wgl = wid.z * nwg.x * nwg.y + wid.y * nwg.x + wid.x;
    let offset = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nz_lo = buffer_6_global[1u];
    let nz_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let r0 = wgl * 256u + l + offset;
    let r1 = r0 + 128u;
    if r0 < width {
        do_row(r0, key_lo, key_hi, nz_lo, nz_hi, step_t, loc, scale, leak, sigma);
    }
    if r1 < width {
        do_row(r1, key_lo, key_hi, nz_lo, nz_hi, step_t, loc, scale, leak, sigma);
    }
}
