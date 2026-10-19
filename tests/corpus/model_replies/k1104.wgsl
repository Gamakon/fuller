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

fn fmix(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    for (var i = li; i < 1024u; i += 256u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let id = global_id.x + global_id.y * num_workgroups.x * 256u;
    let r = id + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let k2 = buffer_5_global[2u];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let base = r * 10u;

    var az = 0.0;
    var ax = 0.0;
    var comp = 0.0;
    for (var s = 0u; s < 10u; s++) {
        let src = buffer_0_global[base + s];
        let x = buffer_1_global[src];
        let w = cb[fmix(h1 ^ (k2 ^ s)) >> 22u];
        let y = w * x - comp;
        let t = az + y;
        comp = (t - az) - y;
        az = t;
        ax += x;
    }

    let hn = fmix(buffer_6_global[1u] ^ r);
    let z = cb[fmix(hn ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let pre = buffer_7_global[1u] * az + buffer_7_global[0u] * ax + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
