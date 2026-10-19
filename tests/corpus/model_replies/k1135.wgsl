const WG = 256u;

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

@compute
@workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let l = lid.x;
    cb[l] = buffer_2_global[l];
    cb[l + 256u] = buffer_2_global[l + 256u];
    cb[l + 512u] = buffer_2_global[l + 512u];
    cb[l + 768u] = buffer_2_global[l + 768u];
    workgroupBarrier();

    let id = (global_id.y * num_workgroups.x * 256u) + global_id.x;
    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let base = r * 10u;
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let khi = buffer_5_global[2u];

    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        xs[s] = buffer_1_global[src[s]];
    }

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let idx = fmix(h1 ^ (khi ^ s)) >> 22u;
        let x = xs[s];
        let term = cb[idx] * x - comp;
        let t = sum + term;
        comp = (t - sum) - term;
        sum = t;
        accx += x;
    }

    let zi = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u;
    let z = cb[zi];
    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
