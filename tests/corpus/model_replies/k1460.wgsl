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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    cb[lidx] = buffer_2_global[lidx];
    cb[lidx + 256u] = buffer_2_global[lidx + 256u];
    cb[lidx + 512u] = buffer_2_global[lidx + 512u];
    cb[lidx + 768u] = buffer_2_global[lidx + 768u];
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    var xs: array<f32, 10>;
    for (var s = 0u; s < 10u; s++) {
        xs[s] = buffer_1_global[src[s]];
    }
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let k2 = buffer_5_global[2u];
    var sum = 0.0;
    var c = 0.0;
    var ax = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let h = fmix(h1 ^ (k2 ^ s));
        let prod = cb[h >> 22u] * xs[s];
        let y = prod - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
        ax = ax + xs[s];
    }
    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[nh >> 22u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * ax + buffer_3_global[r] + buffer_7_global[4u] * z;
    let leak = buffer_7_global[2u];
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
