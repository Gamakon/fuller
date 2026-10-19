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
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    cb[lid] = buffer_2_global[lid];
    cb[lid + 256u] = buffer_2_global[lid + 256u];
    cb[lid + 512u] = buffer_2_global[lid + 512u];
    cb[lid + 768u] = buffer_2_global[lid + 768u];
    workgroupBarrier();

    let width = buffer_5_global[0u];
    let r = gid.x + buffer_6_global[0u];
    if r >= width {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);
    let base = r * 10u;

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;

    var sv: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        sv[s] = buffer_1_global[buffer_0_global[base + s]];
    }
    for (var s = 0u; s < 10u; s++) {
        let w = cb[fmix(h1 ^ (key_hi ^ s)) >> 22u];
        let v = sv[s];
        let term = w * v;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + v;
    }

    let nz = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[nz >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
