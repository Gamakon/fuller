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

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 1024u; k += 256u) {
        wg_a[k + lidx] = buffer_2_global[k + lidx];
    }
    workgroupBarrier();

    let id = gid.x + gid.y * nwg.x * 256u;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let key1 = buffer_5_global[1u];
    let key2 = buffer_5_global[2u];
    let base = r * 10u;
    let h0 = fmix(key1 ^ r);

    // fused: source loads, hashing, Kahan sum and plain sum in one pass
    var s = 0.0;
    var c = 0.0;
    var sx = 0.0;
    for (var k = 0u; k < 10u; k += 1u) {
        let x = buffer_1_global[buffer_0_global[base + k]];
        let h = fmix(h0 ^ (key2 ^ k));
        let p = wg_a[(h >> 22u) & 1023u] * x;
        let y = p - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
        sx = sx + x;
    }

    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[(hn >> 22u) & 1023u];
    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * s + buffer_7_global[0u] * sx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
