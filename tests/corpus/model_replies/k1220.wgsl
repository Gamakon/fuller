enable-free;

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
    @builtin(local_invocation_index) li: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    for (var i: u32 = li; i < 1024u; i += 256u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let id = (global_id.z * num_workgroups.x * 256u * num_workgroups.y) + (global_id.y * num_workgroups.x * 256u) + global_id.x;
    let r = id + buffer_6_global[0u];
    let n = buffer_5_global[0u];
    if r >= n {
        return;
    }
    let klo = buffer_5_global[1u];
    let khi = buffer_5_global[2u];
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

    let h1 = fmix(klo ^ r);
    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    var accx: f32 = 0.0;
    for (var s: u32 = 0u; s < 10u; s++) {
        let hh = fmix(h1 ^ (khi ^ s)) >> 22u;
        let xv = x[s];
        let term = cb[hh] * xv;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx += xv;
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let tt = buffer_6_global[3u];
    let z = cb[fmix(fmix(nlo ^ r) ^ (nhi ^ tt)) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
