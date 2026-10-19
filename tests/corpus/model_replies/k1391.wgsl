// grid 256 1 1 1
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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = 0u; i < 4u; i += 1u) {
        let j = lidx + i * 256u;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let id = gid.x + (gid.y * nwg.x * 256u) + (gid.z * nwg.x * 256u * nwg.y);
    let r = id + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (r >= width) {
        return;
    }
    let base = r * 10u;
    let k0 = fmix(buffer_5_global[1u] ^ r);
    let k1 = buffer_5_global[2u];

    var x: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        x[s] = buffer_1_global[buffer_0_global[base + s]];
    }

    var sum = 0.0f;
    var comp = 0.0f;
    var accx = 0.0f;
    for (var s = 0u; s < 10u; s += 1u) {
        let h = fmix(k0 ^ (k1 ^ s));
        let xv = x[s];
        let y = (cb[h >> 22u] * xv) - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xv;
    }

    let nz = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let leak = buffer_7_global[2u];
    let pre = (((buffer_7_global[1u] * sum) + (buffer_7_global[0u] * accx)) + buffer_3_global[r]) + (buffer_7_global[4u] * cb[nz >> 22u]);
    var th = 1.0f;
    if (pre <= 43.0f) {
        th = tanh(pre);
    }
    buffer_4_global[r] = ((1.0f - leak) * buffer_1_global[r]) + (leak * th);
}
