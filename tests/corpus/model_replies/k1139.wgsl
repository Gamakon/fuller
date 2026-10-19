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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let width = buffer_5_global[0u];
    let base = wid.x * 256u + buffer_6_global[0u];
    let r = base + lidx;

    // coalesced staging of the codebook
    wg_code[lidx] = buffer_2_global[lidx];
    wg_code[lidx + 256u] = buffer_2_global[lidx + 256u];
    wg_code[lidx + 512u] = buffer_2_global[lidx + 512u];
    wg_code[lidx + 768u] = buffer_2_global[lidx + 768u];

    // coalesced staging of this workgroup's sources tile (256 rows * 10)
    let limit = width * 10u;
    let tbase = base * 10u;
    for (var j = 0u; j < 10u; j = j + 1u) {
        let idx = j * 256u + lidx;
        let g = tbase + idx;
        if (g < limit) {
            wg_src[idx] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let row_hash = fmix32(key_lo ^ r);

    var acc_z = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    let sb = lidx * 10u;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let src = wg_src[sb + s];
        let xv = buffer_1_global[src];
        let h = fmix32(row_hash ^ (key_hi ^ s));
        let w = wg_code[h >> 22u];
        let p = w * xv;
        let y = p - comp;
        let t = acc_z + y;
        comp = (t - acc_z) - y;
        acc_z = t;
        acc_x = acc_x + xv;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let nh = fmix32(fmix32(n_lo ^ r) ^ (n_hi ^ n_t));
    let z = wg_code[nh >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * acc_z + loc * acc_x + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
