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

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    wg_a[lidx] = buffer_2_global[lidx];
    wg_a[lidx + 256u] = buffer_2_global[lidx + 256u];
    wg_a[lidx + 512u] = buffer_2_global[lidx + 512u];
    wg_a[lidx + 768u] = buffer_2_global[lidx + 768u];
    workgroupBarrier();

    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }

    let base = r * 10u;
    var idx: array<u32, 10>;
    var val: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        idx[s] = buffer_0_global[base + s];
    }
    for (var s = 0u; s < 10u; s++) {
        val[s] = buffer_1_global[idx[s]];
    }

    let k1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let h = fmix(k1 ^ (k2 ^ s));
        let v = val[s];
        let p = wg_a[h >> 22u] * v;
        let y = p - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + v;
    }

    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[hn >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
