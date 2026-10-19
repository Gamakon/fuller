@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_a: array<f32, 1024>;

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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    for (var k = 0u; k < 4u; k += 1u) {
        let j = lidx + k * 256u;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);
    let base = r * 10u;

    var s0 = buffer_0_global[base];
    var s1 = buffer_0_global[base + 1u];
    var s2 = buffer_0_global[base + 2u];
    var s3 = buffer_0_global[base + 3u];
    var s4 = buffer_0_global[base + 4u];
    var s5 = buffer_0_global[base + 5u];
    var s6 = buffer_0_global[base + 6u];
    var s7 = buffer_0_global[base + 7u];
    var s8 = buffer_0_global[base + 8u];
    var s9 = buffer_0_global[base + 9u];
    var idx = array<u32, 10>(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9);

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var k = 0u; k < 10u; k += 1u) {
        let x = buffer_1_global[idx[k]];
        let w = wg_a[fmix32(h1 ^ (key_hi ^ k)) >> 22u];
        let y = w * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let step = buffer_6_global[3u];
    let z = wg_a[fmix32(fmix32(nlo ^ r) ^ (nhi ^ step)) >> 22u];

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
