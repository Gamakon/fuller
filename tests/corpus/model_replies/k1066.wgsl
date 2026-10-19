@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> tile: array<u32, 2560>;
var<workgroup> cb: array<f32, 1024>;

fn fmix(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    let n = buffer_5_global[0u];
    let off0 = buffer_6_global[0u];
    let wbase = wid.x * 256u + off0;
    let r = wbase + lid;
    let lim = n * 10u;
    let gbase = wbase * 10u;

    for (var i = 0u; i < 4u; i += 1u) {
        let j = i * 256u + lid;
        cb[j] = buffer_2_global[j];
    }
    for (var i = 0u; i < 10u; i += 1u) {
        let j = i * 256u + lid;
        let g = gbase + j;
        if (g < lim) {
            tile[j] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if (r >= n) {
        return;
    }

    let key1 = buffer_5_global[1u];
    let key2 = buffer_5_global[2u];
    let h1 = fmix(key1 ^ r);
    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let tb = lid * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[tile[tb + s]];
        let h = fmix(h1 ^ (key2 ^ s));
        let y = cb[h >> 22u] * x - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = x + accx;
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let st = buffer_6_global[3u];
    let hz = fmix(fmix(nlo ^ r) ^ (nhi ^ st));
    let z = cb[hz >> 22u];

    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    let th = select(tanh(pre), 1.0, pre > 43.0);
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
