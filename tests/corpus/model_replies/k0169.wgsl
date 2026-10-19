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
    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let r = id + off;
    let base = (id - lidx) + off;

    // codebook into workgroup memory (coalesced)
    for (var k = 0u; k < 4u; k += 1u) {
        let j = k * 256u + lidx;
        wg_cb[j] = buffer_2_global[j];
    }
    // sources of this workgroup's rows into workgroup memory (coalesced)
    for (var k = 0u; k < 10u; k += 1u) {
        let j = k * 256u + lidx;
        if (base + j / 10u < width) {
            wg_src[j] = buffer_0_global[base * 10u + j];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);
    var acc_z = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    let sbase = lidx * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let xv = buffer_1_global[wg_src[sbase + s]];
        let hh = fmix(h0 ^ (key_hi ^ s));
        let term = wg_cb[hh >> 22u] * xv;
        let y = term - comp;
        let t = acc_z + y;
        comp = (t - acc_z) - y;
        acc_z = t;
        acc_x = acc_x + xv;
    }
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let z = wg_cb[fmix(fmix(n_lo ^ r) ^ (n_hi ^ n_t)) >> 22u];
    let pre = buffer_7_global[1u] * acc_z + buffer_7_global[0u] * acc_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
