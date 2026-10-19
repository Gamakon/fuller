var<workgroup> cb: array<f32, 1024>;
var<workgroup> src: array<u32, 2560>;

@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix32(hin: u32) -> u32 {
    var h = hin;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let row0 = wid.x * 256u + off;

    // cooperative, coalesced staging of codebook and source indices
    for (var k = 0u; k < 4u; k += 1u) {
        let j = k * 256u + lidx;
        cb[j] = buffer_2_global[j];
    }
    for (var k = 0u; k < 10u; k += 1u) {
        let j = k * 256u + lidx;
        let rr = row0 + j / 10u;
        var v = 0u;
        if (rr < width) {
            v = buffer_0_global[row0 * 10u + j];
        }
        src[j] = v;
    }
    workgroupBarrier();

    let r = gid.x + off;
    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = lidx * 10u;
    let h0 = fmix32(key_lo ^ r);

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[src[base + s]];
        let h = fmix32(h0 ^ (key_hi ^ s));
        let y = cb[h >> 22u] * x - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + x;
    }

    let hz = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[hz >> 22u];

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
