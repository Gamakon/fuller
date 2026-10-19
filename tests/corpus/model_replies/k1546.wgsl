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
fn generated_step(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    let width = buffer_5_global[0u];
    let offs = buffer_6_global[0u];
    let rbase = wid.x * 256u + offs;

    for (var k = 0u; k < 4u; k += 1u) {
        let i = k * 256u + lid;
        wg_code[i] = buffer_2_global[i];
    }
    // cooperative coalesced load of this workgroup's 2560 source indices
    for (var j = 0u; j < 10u; j += 1u) {
        let i = j * 256u + lid;
        let row = rbase + i / 10u;
        if (row < width) {
            wg_src[i] = buffer_0_global[rbase * 10u + i];
        }
    }
    workgroupBarrier();

    let r = rbase + lid;
    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(key_lo ^ r);
    let sb = lid * 10u;

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let xv = buffer_1_global[wg_src[sb + s]];
        let w = wg_code[fmix32(h0 ^ (key_hi ^ s)) >> 22u];
        let term = w * xv;
        let y = term - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + xv;
    }

    let nz = wg_code[fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = ((scale * acc + loc * accx) + buffer_3_global[r]) + sigma * nz;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
