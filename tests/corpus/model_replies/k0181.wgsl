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
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let width = buffer_5_global[0u];
    let offs = buffer_6_global[0u];
    let wgbase = (wid.y * nwg.x + wid.x) * 256u + offs;
    let l = lid.x;
    for (var k = 0u; k < 4u; k += 1u) {
        let j = k * 256u + l;
        wg_cb[j] = buffer_2_global[j];
    }
    let total = width * 10u;
    for (var k = 0u; k < 10u; k += 1u) {
        let j = k * 256u + l;
        let idx = wgbase * 10u + j;
        if (idx < total) {
            wg_src[j] = buffer_0_global[idx];
        }
    }
    workgroupBarrier();
    let r = wgbase + l;
    if (r >= width) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);
    var x: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        x[s] = buffer_1_global[wg_src[l * 10u + s]];
    }
    var acc = 0.0;
    var c = 0.0;
    var ax = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let hh = fmix(h1 ^ (key_hi ^ s));
        let p = wg_cb[hh >> 22u] * x[s];
        let t = p - c;
        let sum = acc + t;
        c = (sum - acc) - t;
        acc = sum;
        ax = ax + x[s];
    }
    let n1 = fmix(buffer_6_global[1u] ^ r);
    let n2 = fmix(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_cb[n2 >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = scale * acc + loc * ax + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
