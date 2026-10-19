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
    @builtin(local_invocation_index) li: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = (global_id.z * num_workgroups.x * 256u * num_workgroups.y) + (global_id.y * num_workgroups.x * 256u) + global_id.x;

    cb[li] = buffer_2_global[li];
    cb[li + 256u] = buffer_2_global[li + 256u];
    cb[li + 512u] = buffer_2_global[li + 512u];
    cb[li + 768u] = buffer_2_global[li + 768u];
    workgroupBarrier();

    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let base = r * 10u;
    let s0 = buffer_0_global[base];
    let s1 = buffer_0_global[base + 1u];
    let s2 = buffer_0_global[base + 2u];
    let s3 = buffer_0_global[base + 3u];
    let s4 = buffer_0_global[base + 4u];
    let s5 = buffer_0_global[base + 5u];
    let s6 = buffer_0_global[base + 6u];
    let s7 = buffer_0_global[base + 7u];
    let s8 = buffer_0_global[base + 8u];
    let s9 = buffer_0_global[base + 9u];

    var x: array<f32, 10>;
    x[0] = buffer_1_global[s0];
    x[1] = buffer_1_global[s1];
    x[2] = buffer_1_global[s2];
    x[3] = buffer_1_global[s3];
    x[4] = buffer_1_global[s4];
    x[5] = buffer_1_global[s5];
    x[6] = buffer_1_global[s6];
    x[7] = buffer_1_global[s7];
    x[8] = buffer_1_global[s8];
    x[9] = buffer_1_global[s9];

    let khi = buffer_5_global[2u];
    let h0 = fmix(buffer_5_global[1u] ^ r);

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s: u32 = 0u; s < 10u; s++) {
        let w = cb[fmix(h0 ^ (khi ^ s)) >> 22u];
        let xv = x[s];
        let term = w * xv;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xv;
    }

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * nz;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
