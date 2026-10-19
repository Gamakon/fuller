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

@compute
@workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lx = lid.x;
    cb[lx] = buffer_2_global[lx];
    cb[lx + 256u] = buffer_2_global[lx + 256u];
    cb[lx + 512u] = buffer_2_global[lx + 512u];
    cb[lx + 768u] = buffer_2_global[lx + 768u];
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);
    let base = r * 10u;

    var sum = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s: u32 = 0u; s < 10u; s++) {
        let xv = buffer_1_global[buffer_0_global[base + s]];
        let h = fmix(h1 ^ (key_hi ^ s));
        let t = cb[h >> 22u] * xv;
        let y = t - comp;
        let ns = sum + y;
        comp = (ns - sum) - y;
        sum = ns;
        accx = accx + xv;
    }
    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[hn >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
