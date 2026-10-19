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

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 8u; k = k + 1u) {
        let j = k * 128u + lid.x;
        wg_a[j & 1023u] = buffer_2_global[j];
    }
    workgroupBarrier();

    let id = ((gid.z * nwg.x * 128u * nwg.y) + (gid.y * nwg.x * 128u)) + gid.x;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let base = r * 10u;

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let src = buffer_0_global[base + k];
        let xs = buffer_1_global[src];
        let hh = fmix(h1 ^ (key_hi ^ k));
        let prod = wg_a[(hh >> 22u) & 1023u] * xs;
        let y = prod - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xs;
    }

    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[(hn >> 22u) & 1023u];
    let leak = buffer_7_global[2u];
    let pre = ((buffer_7_global[1u] * sum + buffer_7_global[0u] * accx) + buffer_3_global[r]) + buffer_7_global[4u] * z;
    let th = select(tanh(pre), 1.0, pre > 43.0);
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
