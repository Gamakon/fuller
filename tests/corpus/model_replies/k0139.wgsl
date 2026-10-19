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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 1024u; k += 256u) {
        wg_a[k + lid.x] = buffer_2_global[k + lid.x];
    }
    workgroupBarrier();

    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
    let row = id + buffer_6_global[0u];
    if (row >= buffer_5_global[0u]) {
        return;
    }

    let base = row * 10u;
    let h0 = fmix32(buffer_5_global[1u] ^ row);
    let k2 = buffer_5_global[2u];

    var sum = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let x = buffer_1_global[buffer_0_global[base + s]];
        let h = fmix32(h0 ^ (k2 ^ s));
        let t = wg_a[h >> 22u] * x - comp;
        let n = sum + t;
        comp = (n - sum) - t;
        sum = n;
        acc_x = acc_x + x;
    }

    let nh = fmix32(fmix32(buffer_6_global[1u] ^ row) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[nh >> 22u];

    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * acc_x + buffer_3_global[row] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[row] = (1.0 - leak) * buffer_1_global[row] + leak * th;
}
