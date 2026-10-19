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
    let offs = buffer_6_global[0u];
    let base = wid.x * 256u + offs;

    for (var i = 0u; i < 4u; i += 1u) {
        let j = lidx + i * 256u;
        cb[j] = buffer_2_global[j];
    }
    for (var i = 0u; i < 10u; i += 1u) {
        let j = lidx + i * 256u;
        let row = base + j / 10u;
        if (row < width) {
            src[j] = buffer_0_global[base * 10u + j];
        }
    }
    workgroupBarrier();

    let r = base + lidx;
    if (r >= width) {
        return;
    }
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(buffer_5_global[1u] ^ r);
    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let sb = lidx * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[src[sb + s]];
        let h = fmix32(h0 ^ (key_hi ^ s));
        let y = cb[h >> 22u] * x - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + x;
    }
    let hn = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[hn >> 22u];
    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
