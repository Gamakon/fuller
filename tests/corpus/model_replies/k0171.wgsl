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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let l = lid.x;
    let row0 = wid.x * 256u + off;

    // stage codebook (coalesced)
    for (var i = 0u; i < 1024u; i += 256u) {
        wg_cb[i + l] = buffer_2_global[i + l];
    }
    // stage this workgroup's contiguous source block (coalesced)
    let base = row0 * 10u;
    let limit = width * 10u;
    for (var j = 0u; j < 2560u; j += 256u) {
        let e = base + j + l;
        if (e < limit) {
            wg_src[j + l] = buffer_0_global[e];
        }
    }
    workgroupBarrier();

    let r = gid.x + off;
    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);

    let sb = l * 10u;
    var v: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        v[s] = buffer_1_global[wg_src[sb + s]];
    }

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let h = fmix(h1 ^ (key_hi ^ s));
        let term = wg_cb[h >> 22u] * v[s];
        let y = term - comp;
        let tt = sum + y;
        comp = (tt - sum) - y;
        sum = tt;
        accx = accx + v[s];
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let hn = fmix(fmix(n_lo ^ r) ^ (n_hi ^ n_t));
    let z = wg_cb[hn >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
