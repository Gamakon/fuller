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
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let l = lid.x;
    cb[l] = buffer_2_global[l];
    cb[l + 256u] = buffer_2_global[l + 256u];
    cb[l + 512u] = buffer_2_global[l + 512u];
    cb[l + 768u] = buffer_2_global[l + 768u];
    workgroupBarrier();

    let r = wg.x * 256u + l + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if r >= width {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;
    let h1 = fmix(key_lo ^ r);

    var acc = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s: u32 = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let v = buffer_1_global[src];
        let idx = fmix(h1 ^ (key_hi ^ s)) >> 22u;
        let term = cb[idx] * v;
        let y = term - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + v;
    }

    let nz = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[nz >> 22u];

    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
