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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    // stage codebook (all threads participate, uniform control flow)
    cb[li] = buffer_2_global[li];
    cb[li + 256u] = buffer_2_global[li + 256u];
    cb[li + 512u] = buffer_2_global[li + 512u];
    cb[li + 768u] = buffer_2_global[li + 768u];
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);
    let base = r * 10u;

    var acc_z = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;

    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let x = buffer_1_global[src];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = cb[hh >> 22u];
        let y = w * x - comp;
        let t = acc_z + y;
        comp = (t - acc_z) - y;
        acc_z = t;
        acc_x = acc_x + x;
    }

    let nh = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[nh >> 22u];

    let pre = buffer_7_global[1u] * acc_z + buffer_7_global[0u] * acc_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
