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

fn fmix32(hin: u32) -> u32 {
    var h = hin;
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
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let lid = lid3.x;
    for (var i = 0u; i < 1024u; i += 256u) {
        cb[i + lid] = buffer_2_global[i + lid];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);
    let base = r * 10u;

    var sum = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;

    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let xv = buffer_1_global[src];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = cb[hh >> 22u];
        let y = w * xv - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        acc_x = acc_x + xv;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let zh = fmix32(fmix32(n_lo ^ r) ^ (n_hi ^ step_t));
    let z = cb[zh >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * acc_x + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
