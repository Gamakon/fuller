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

fn fmix(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn step_row(r: u32, key_lo: u32, key_hi: u32, nz: u32, loc: f32, scale: f32, leak: f32, sigma: f32) {
    let h1 = fmix(key_lo ^ r);
    let base = r * 10u;
    var s_acc: f32 = 0.0;
    var c: f32 = 0.0;
    var ax: f32 = 0.0;
    for (var s: u32 = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let x = buffer_1_global[src];
        let w = cb[fmix(h1 ^ (key_hi ^ s)) >> 22u];
        let y = w * x - c;
        let t = s_acc + y;
        c = (t - s_acc) - y;
        s_acc = t;
        ax = ax + x;
    }
    let off1 = buffer_6_global[1u];
    let z = cb[fmix(fmix(off1 ^ r) ^ nz) >> 22u];
    let pre = scale * s_acc + loc * ax + buffer_3_global[r] + sigma * z;
    var th: f32 = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}

@compute @workgroup_size(64, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    for (var i: u32 = 0u; i < 16u; i++) {
        let k = li + i * 64u;
        cb[k] = buffer_2_global[k];
    }
    workgroupBarrier();

    let id = global_id.x + global_id.y * num_workgroups.x * 64u;
    let offset = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let first = id * 4u + offset;
    if first >= width {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nz = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    for (var k: u32 = 0u; k < 4u; k++) {
        let r = first + k;
        if r < width {
            step_row(r, key_lo, key_hi, nz, loc, scale, leak, sigma);
        }
    }
}
