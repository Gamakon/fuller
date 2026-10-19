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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let base = wid.x * 256u + off;

    // cooperative, coalesced staging of the codebook
    for (var k = 0u; k < 4u; k += 1u) {
        let j = lidx + k * 256u;
        wg_code[j] = buffer_2_global[j];
    }
    // cooperative, coalesced staging of this workgroup's source indices
    for (var k = 0u; k < 10u; k += 1u) {
        let j = lidx + k * 256u;
        let row = base + j / 10u;
        if (row < width) {
            wg_src[j] = buffer_0_global[base * 10u + j];
        } else {
            wg_src[j] = 0u;
        }
    }
    workgroupBarrier();

    let r = base + lidx;
    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let sb = lidx * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let src = wg_src[sb + s];
        let x = buffer_1_global[src];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = wg_code[hh >> 22u];
        let p = w * x;
        let y = p - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + x;
    }

    let n1 = fmix32(buffer_6_global[1u] ^ r);
    let n2 = fmix32(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_code[n2 >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = ((scale * acc + loc * accx) + buffer_3_global[r]) + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
