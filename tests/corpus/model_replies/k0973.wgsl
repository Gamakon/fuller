@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_code: array<f32, 1024>;
var<workgroup> wg_src: array<u32, 2560>;

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
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let l = lid.x;
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let base = wid.x * 256u + off;
    let total = width * 10u;

    for (var k = 0u; k < 4u; k += 1u) {
        wg_code[k * 256u + l] = buffer_2_global[k * 256u + l];
    }
    let sbase = base * 10u;
    for (var k = 0u; k < 10u; k += 1u) {
        let li = k * 256u + l;
        let gi = sbase + li;
        var v = 0u;
        if (gi < total) {
            v = buffer_0_global[gi];
        }
        wg_src[li] = v;
    }
    workgroupBarrier();

    let r = base + l;
    if (r >= width) {
        return;
    }

    let h1 = fmix32(key_lo ^ r);
    var sum = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    let sb = l * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[wg_src[sb + s]];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = wg_code[hh >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        acc_x = acc_x + x;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let z = wg_code[fmix32(fmix32(n_lo ^ r) ^ (n_hi ^ n_t)) >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * acc_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
