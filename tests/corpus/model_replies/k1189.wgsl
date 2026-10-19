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
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // stage codebook: 4 entries per thread
    cb[lid] = buffer_2_global[lid];
    cb[lid + 256u] = buffer_2_global[lid + 256u];
    cb[lid + 512u] = buffer_2_global[lid + 512u];
    cb[lid + 768u] = buffer_2_global[lid + 768u];
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }

    let key_hi = buffer_5_global[2u];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let base = r * 10u;

    var src: array<u32, 10>;
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    for (var s = 0u; s < 10u; s++) {
        xs[s] = buffer_1_global[src[s]];
    }

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let h = fmix(h1 ^ (key_hi ^ s));
        let w = cb[h >> 22u];
        let x = xs[s];
        let term = w * x - comp;
        let t = sum + term;
        comp = (t - sum) - term;
        sum = t;
        accx = accx + x;
    }

    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[hn >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
