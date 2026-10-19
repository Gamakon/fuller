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

    let r = (gid.y * nwg.x * 256u) + gid.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (r >= width) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);
    let base = r * 10u;

    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        src[s] = buffer_0_global[base + s];
    }
    var st: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        st[s] = buffer_1_global[src[s]];
    }

    var sum = 0.0;
    var c = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = st[s];
        let h = fmix(h0 ^ (key_hi ^ s));
        let p = cb[h >> 22u] * x;
        let y = p - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
        accx = accx + x;
    }

    let nz = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[nz >> 22u];
    let leak = buffer_7_global[2u];
    let pre = (((buffer_7_global[1u] * sum) + (buffer_7_global[0u] * accx)) + buffer_3_global[r]) + (buffer_7_global[4u] * z);
    let th = select(tanh(pre), 1.0, pre > 43.0);
    buffer_4_global[r] = ((1.0 - leak) * buffer_1_global[r]) + (leak * th);
}
