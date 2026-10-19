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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let base = wid.x * 256u + off;

    for (var i = 0u; i < 4u; i += 1u) {
        let j = i * 256u + lidx;
        wg_cb[j] = buffer_2_global[j];
    }
    for (var i = 0u; i < 10u; i += 1u) {
        let j = i * 256u + lidx;
        let row = base + j / 10u;
        var v = 0u;
        if (row < width) {
            v = buffer_0_global[base * 10u + j];
        }
        wg_src[j] = v;
    }
    workgroupBarrier();

    let r = gid.x + off;
    if (r >= width) {
        return;
    }

    let h1 = fmix32(key_lo ^ r);
    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let sb = lidx * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[wg_src[sb + s]];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let y = wg_cb[hh >> 22u] * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let n1 = fmix32(buffer_6_global[1u] ^ r);
    let nh = fmix32(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_cb[nh >> 22u];

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
