@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_cb: array<f32, 1024>;
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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let width = buffer_5_global[0u];
    let r0 = wid.x * 256u + buffer_6_global[0u];
    let r = r0 + lidx;

    // stage codebook coalesced
    for (var k = 0u; k < 4u; k += 1u) {
        let e = k * 256u + lidx;
        wg_cb[e] = buffer_2_global[e];
    }
    // stage sources coalesced: this workgroup's rows are contiguous
    let limit = width * 10u;
    let base = r0 * 10u;
    for (var j = 0u; j < 10u; j += 1u) {
        let e = j * 256u + lidx;
        let g = base + e;
        if (g < limit) {
            wg_src[e] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);
    let sb = lidx * 10u;

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[wg_src[sb + s]];
        let hs = fmix32(h1 ^ (key_hi ^ s));
        let t = wg_cb[hs >> 22u] * x;
        let y = t - comp;
        let ns = sum + y;
        comp = (ns - sum) - y;
        sum = ns;
        accx = accx + x;
    }

    let nz = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_cb[nz >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
