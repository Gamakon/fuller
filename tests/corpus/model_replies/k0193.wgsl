@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_src: array<u32, 2560>;
var<workgroup> wg_cb: array<f32, 1024>;

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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let base = wid.x * 256u + off;
    let r = base + lidx;

    // cooperative coalesced staging of codebook and this workgroup's sources
    for (var i = 0u; i < 1024u; i += 256u) {
        let j = i + lidx;
        wg_cb[j] = buffer_2_global[j];
    }
    let limit = width * 10u;
    let gbase = base * 10u;
    for (var i = 0u; i < 2560u; i += 256u) {
        let j = i + lidx;
        let g = gbase + j;
        if (g < limit) {
            wg_src[j] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);
    let sb = lidx * 10u;

    var comp = 0.0;
    var sum = 0.0;
    var accx = 0.0;
    for (var k = 0u; k < 10u; k += 1u) {
        let xs = buffer_1_global[wg_src[sb + k]];
        let h = fmix(h0 ^ (key_hi ^ k));
        let p = wg_cb[h >> 22u] * xs;
        let y = p - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xs;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let hn = fmix(fmix(n_lo ^ r) ^ (n_hi ^ n_t));
    let z = wg_cb[hn >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = (scale * sum + loc * accx) + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
