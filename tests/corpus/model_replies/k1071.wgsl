var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_s: array<u32, 2560>;
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

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let l = lid.x;
    let off0 = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let r0 = wid.x * 256u + off0;
    let r = r0 + l;
    var n = 0u;
    if (width > r0) {
        n = width - r0;
    }
    let lim = n * 10u;
    let base = r0 * 10u;

    for (var k = 0u; k < 4u; k = k + 1u) {
        let j = l + k * 256u;
        wg_a[j] = buffer_2_global[j];
    }
    for (var k = 0u; k < 10u; k = k + 1u) {
        let j = l + k * 256u;
        if (j < lim) {
            wg_s[j] = buffer_0_global[base + j];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);
    var sz = 0.0;
    var comp = 0.0;
    var sx = 0.0;
    let sb = l * 10u;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let src = wg_s[sb + s];
        let x = buffer_1_global[src];
        let hh = fmix(h0 ^ (key_hi ^ s));
        let y = wg_a[(hh >> 22u) & 1023u] * x - comp;
        let t = sz + y;
        comp = (t - sz) - y;
        sz = t;
        sx = sx + x;
    }
    let nz = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[(nz >> 22u) & 1023u];
    let pre = buffer_7_global[1u] * sz + buffer_7_global[0u] * sx + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
