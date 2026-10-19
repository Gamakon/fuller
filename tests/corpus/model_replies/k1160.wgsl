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

@compute @workgroup_size(128, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    for (var i: u32 = li; i < 1024u; i += 128u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;

    var src: array<u32, 10>;
    for (var s: u32 = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    let h0 = fmix(key_lo ^ r);

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s: u32 = 0u; s < 10u; s++) {
        let xv = buffer_1_global[src[s]];
        let h = fmix(h0 ^ (key_hi ^ s));
        let w = cb[h >> 22u];
        let y = w * xv - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx += xv;
    }

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * nz;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
