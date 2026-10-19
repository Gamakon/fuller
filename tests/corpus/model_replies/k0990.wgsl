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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = (gid.y * nwg.x * 256u) + gid.x;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let r = id + off;
    let rbase = (id - lidx) + off;

    // stage codebook, coalesced
    for (var k = 0u; k < 1024u; k += 256u) {
        wg_cb[k + lidx] = buffer_2_global[k + lidx];
    }
    // stage sources of the workgroup's rows, coalesced
    var nrows = 0u;
    if (rbase < width) {
        nrows = min(256u, width - rbase);
    }
    let nel = nrows * 10u;
    let sbase = rbase * 10u;
    for (var k = 0u; k < 2560u; k += 256u) {
        let j = k + lidx;
        if (j < nel) {
            wg_src[j] = buffer_0_global[sbase + j];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);
    let sb = lidx * 10u;

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[wg_src[sb + s]];
        let h = fmix(h1 ^ (key_hi ^ s));
        let term = wg_cb[h >> 22u] * x - comp;
        let t = acc + term;
        comp = (t - acc) - term;
        acc = t;
        accx = accx + x;
    }

    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_cb[nh >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
