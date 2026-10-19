@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    var src: array<u32, 10>;
    for (var k = 0u; k < 10u; k++) {
        src[k] = buffer_0_global[base + k];
    }
    var st: array<f32, 10>;
    for (var k = 0u; k < 10u; k++) {
        st[k] = buffer_1_global[src[k]];
    }
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let khi = buffer_5_global[2u];
    var acc = 0.0f;
    var comp = 0.0f;
    var sx = 0.0f;
    for (var k = 0u; k < 10u; k++) {
        let f1 = st[k];
        let hh = fmix(h1 ^ (khi ^ k));
        let c = buffer_2_global[hh >> 22u];
        let y = c * f1 - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        sx = sx + f1;
    }
    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[nh >> 22u];
    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * acc + buffer_7_global[0u] * sx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0f;
    if (!(pre > 43.0f)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0f - leak) * buffer_1_global[r] + leak * th;
}
