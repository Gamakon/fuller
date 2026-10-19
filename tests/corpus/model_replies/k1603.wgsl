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

fn fmix(hin: u32) -> u32 {
    var h = hin;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 4u; k += 1u) {
        let j = k * 256u + lidx;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let r = gid.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let base = r * 10u;

    var src: array<u32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        src[s] = buffer_0_global[base + s];
    }
    var st: array<f32, 10>;
    for (var s = 0u; s < 10u; s += 1u) {
        st[s] = buffer_1_global[src[s]];
    }

    let h1 = fmix(key_lo ^ r);
    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let hh = fmix(h1 ^ (key_hi ^ s));
        let v = st[s];
        let y = wg_a[hh >> 22u] * v - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + v;
    }

    let n1 = fmix(buffer_6_global[1u] ^ r);
    let n2 = fmix(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_a[n2 >> 22u];

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
