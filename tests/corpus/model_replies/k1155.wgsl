var<workgroup> wg_cb: array<f32, 1024>;
var<workgroup> wg_src: array<u32, 2560>;

@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let l = lid.x;
    let base = wid.x * 256u + off;
    let r = base + l;

    // stage codebook
    wg_cb[l] = buffer_2_global[l];
    wg_cb[l + 256u] = buffer_2_global[l + 256u];
    wg_cb[l + 512u] = buffer_2_global[l + 512u];
    wg_cb[l + 768u] = buffer_2_global[l + 768u];

    // stage sources coalesced
    let total = width * 10u;
    let gbase = base * 10u;
    for (var j = 0u; j < 10u; j = j + 1u) {
        let idx = j * 256u + l;
        let g = gbase + idx;
        var v = 0u;
        if (g < total) {
            v = buffer_0_global[g];
        }
        wg_src[idx] = v;
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let rh = fmix32(key_lo ^ r);
    let kh = key_hi;
    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let sb = l * 10u;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let src = wg_src[sb + s];
        let x = buffer_1_global[src];
        let h = fmix32(rh ^ (kh ^ s));
        let w = wg_cb[h >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let nz = wg_cb[fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * nz;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
