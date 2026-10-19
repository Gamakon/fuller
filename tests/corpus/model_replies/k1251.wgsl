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
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    for (var i: u32 = t; i < 1024u; i += 128u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();
    if t >= 32u {
        return;
    }
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let step = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nk = n_hi ^ step;
    let base = off + wid.x * 128u + t * 4u;

    for (var j: u32 = 0u; j < 4u; j++) {
        let r = base + j;
        if r >= width {
            break;
        }
        let k1 = fmix(key_lo ^ r);
        let rb = r * 10u;
        var sum = 0f;
        var comp = 0f;
        var accx = 0f;
        for (var s: u32 = 0u; s < 10u; s++) {
            let src = buffer_0_global[rb + s];
            let xv = buffer_1_global[src];
            let h = fmix(k1 ^ (key_hi ^ s));
            let term = cb[h >> 22u] * xv;
            let y = term - comp;
            let tt = sum + y;
            comp = (tt - sum) - y;
            sum = tt;
            accx += xv;
        }
        let z = cb[fmix(fmix(n_lo ^ r) ^ nk) >> 22u];
        let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
        buffer_4_global[r] = (1f - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
    }
}
