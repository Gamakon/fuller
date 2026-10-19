@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_s: array<u32, 2560>;

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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    let offs = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let r = gid.x + offs;
    let base = wid.x * 256u + offs;
    let limit = width * 10u;

    for (var t = 0u; t < 1024u; t += 256u) {
        wg_a[t + lidx] = buffer_2_global[t + lidx];
    }
    // cooperative, contiguous load of this workgroup's source indices
    for (var i = 0u; i < 10u; i += 1u) {
        let j = i * 256u + lidx;
        let g = base * 10u + j;
        if (g < limit) {
            wg_s[j] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let sb = lidx * 10u;
    for (var k = 0u; k < 10u; k += 1u) {
        let src = wg_s[sb + k];
        let x = buffer_1_global[src];
        let h = fmix(h0 ^ (key_hi ^ k));
        let w = wg_a[h >> 22u];
        let y = w * x - comp;
        let tt = acc + y;
        comp = (tt - acc) - y;
        acc = tt;
        accx = accx + x;
    }

    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[hn >> 22u];
    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
