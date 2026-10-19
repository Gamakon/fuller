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
fn generated_step(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let wgl = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let base = wgl * 256u + off;
    let r = base + lidx;

    // stage codebook
    for (var k = 0u; k < 4u; k = k + 1u) {
        let j = k * 256u + lidx;
        wg_cb[j] = buffer_2_global[j];
    }
    // stage sources coalesced
    let total = select(0u, (n - base) * 10u, base < n);
    let gbase = base * 10u;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let j = k * 256u + lidx;
        if (j < total) {
            wg_src[j] = buffer_0_global[gbase + j];
        }
    }
    workgroupBarrier();

    if (r >= n) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(key_lo ^ r);
    let sb = lidx * 10u;

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let x = buffer_1_global[wg_src[sb + s]];
        let w = wg_cb[fmix32(h0 ^ (key_hi ^ s)) >> 22u];
        let y = w * x - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + x;
    }

    let nz = wg_cb[fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * nz;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
