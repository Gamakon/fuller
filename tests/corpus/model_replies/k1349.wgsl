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

fn fmix32(hin: u32) -> u32 {
    var h = hin;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let l = lid.x;
    for (var i = 0u; i < 4u; i += 1u) {
        let j = i * 256u + l;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (r >= width) {
        return;
    }
    let base = r * 10u;

    // issue all gathers up front
    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        src[s] = buffer_0_global[base + s];
    }
    var xv: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        xv[s] = buffer_1_global[src[s]];
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(key_lo ^ r);

    var sum = 0.0;
    var c = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let h = fmix32(h0 ^ (key_hi ^ s));
        let p = cb[h >> 22u] * xv[s];
        let y = p - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
        accx = accx + xv[s];
    }

    let n0 = buffer_6_global[1u];
    let n1 = buffer_6_global[2u];
    let n2 = buffer_6_global[3u];
    let z = cb[fmix32(fmix32(n0 ^ r) ^ (n1 ^ n2)) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = (scale * sum + loc * accx) + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
