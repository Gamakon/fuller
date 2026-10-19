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

fn fmix32(h0: u32) -> u32 {
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
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let lid = lid3.x;
    for (var i = 0u; i < 1024u; i += 128u) {
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
    let base = r * 10u;
    let h1 = fmix32(key_lo ^ r);

    var sum = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let x = buffer_1_global[src];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = cb[hh >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let n1 = buffer_6_global[1u];
    let n2 = buffer_6_global[2u];
    let n3 = buffer_6_global[3u];
    let nh = fmix32(fmix32(n1 ^ r) ^ (n2 ^ n3));
    let z = cb[nh >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1f - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
