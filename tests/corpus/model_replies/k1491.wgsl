// Workgroup size stays 256 because the harness dispatches ceil(width/256) workgroups. Each workgroup therefore has to cover exactly 256 rows, so I could not give a thread more than one row without leaving rows unprocessed. Instead the kernel unrolls the ten edges and hoists the scalars, so each thread has ten independent loads and hashes in flight.
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
    let k2 = buffer_5_global[2u];
    let h0 = fmix(buffer_5_global[1u] ^ r);

    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        src[s] = buffer_0_global[base + s];
    }
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        xs[s] = buffer_1_global[src[s]];
    }
    var ws: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        ws[s] = buffer_2_global[fmix(h0 ^ (k2 ^ s)) >> 22u];
    }

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = xs[s];
        let p = ws[s] * x - comp;
        let t = sum + p;
        comp = (t - sum) - p;
        sum = t;
        accx = accx + x;
    }

    let hn = fmix(buffer_6_global[1u] ^ r);
    let z = buffer_2_global[fmix(hn ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
