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

fn fmix32(h_in: u32) -> u32 {
    var h = h_in;
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
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    for (var i = 0u; i < 1024u; i += 256u) {
        let j = i + lid;
        if (j < 1024u) {
            cb[j] = buffer_2_global[j];
        }
    }
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (r >= width) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(key_lo ^ r);
    let base = r * 10u;

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let xv = buffer_1_global[src];
        let h = fmix32(h0 ^ (key_hi ^ s));
        let term = cb[h >> 22u] * xv;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xv;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let hn = fmix32(fmix32(n_lo ^ r) ^ (n_hi ^ step_t));
    let z = cb[hn >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    let xr = buffer_1_global[r];
    buffer_4_global[r] = (1.0 - leak) * xr + leak * safe_tanh(pre);
}
