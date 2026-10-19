@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> cb: array<f32, 1024>;
var<workgroup> src: array<u32, 2560>;

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
    let base_row = wid.x * 256u + off;
    let r = base_row + lidx;

    for (var q = 0u; q < 4u; q += 1u) {
        cb[q * 256u + lidx] = buffer_2_global[q * 256u + lidx];
    }
    let gbase = base_row * 10u;
    let lim = width * 10u;
    for (var q = 0u; q < 10u; q += 1u) {
        let j = q * 256u + lidx;
        let g = gbase + j;
        if (g < lim) {
            src[j] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);
    let hb = h1 ^ key_hi;

    let sb = lidx * 10u;
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        xs[s] = buffer_1_global[src[sb + s]];
    }

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = xs[s];
        let w = cb[fmix(hb ^ s) >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let n1 = fmix(buffer_6_global[1u] ^ r);
    let z = cb[fmix(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = ((scale * sum + loc * accx) + buffer_3_global[r]) + sigma * z;
    let th = select(tanh(pre), 1.0, pre > 43.0);
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
