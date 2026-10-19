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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    cb[lid] = buffer_2_global[lid];
    cb[lid + 256u] = buffer_2_global[lid + 256u];
    cb[lid + 512u] = buffer_2_global[lid + 512u];
    cb[lid + 768u] = buffer_2_global[lid + 768u];
    workgroupBarrier();

    let id = wid.x * 256u + lid;
    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let base = r * 10u;

    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        xs[s] = buffer_1_global[src[s]];
    }

    var sum = 0.0;
    var c = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let hh = fmix(h1 ^ (key_hi ^ s));
        let w = cb[hh >> 22u];
        let p = w * xs[s];
        let y = p - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
        accx = accx + xs[s];
    }

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * nz;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
