var<workgroup> cb: array<f32, 1024>;

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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let id = (global_id.z * num_workgroups.x * 256u * num_workgroups.y) + (global_id.y * num_workgroups.x * 256u) + global_id.x;

    cb[li] = buffer_2_global[li];
    cb[li + 256u] = buffer_2_global[li + 256u];
    cb[li + 512u] = buffer_2_global[li + 512u];
    cb[li + 768u] = buffer_2_global[li + 768u];
    workgroupBarrier();

    let r = id + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if r >= width {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;
    let h1 = fmix(key_lo ^ r);

    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        xs[s] = buffer_1_global[src[s]];
    }

    var sum = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s = 0u; s < 10u; s++) {
        let x = xs[s];
        let hh = fmix(h1 ^ (key_hi ^ s));
        let w = cb[hh >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let z = cb[fmix(fmix(n_lo ^ r) ^ (n_hi ^ n_t)) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1f - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
