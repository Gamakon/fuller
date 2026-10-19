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

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    for (var j = 0u; j < 4u; j += 1u) {
        let q = lidx + j * 256u;
        cb[q] = buffer_2_global[q];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let k2 = buffer_5_global[2u];
    let base = r * 10u;

    var idx: array<u32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        idx[s] = buffer_0_global[base + s];
    }
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        xs[s] = buffer_1_global[idx[s]];
    }

    let h1 = fmix(key_lo ^ r);
    let hk = h1 ^ k2;
    var c = 0.0;
    var comp = 0.0;
    var sx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = xs[s];
        let w = cb[fmix(hk ^ s) >> 22u];
        let p = w * x;
        let y = p - comp;
        let t = c + y;
        comp = (t - c) - y;
        c = t;
        sx = sx + x;
    }

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let pre = buffer_7_global[1u] * c + buffer_7_global[0u] * sx + buffer_3_global[r] + buffer_7_global[4u] * nz;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
