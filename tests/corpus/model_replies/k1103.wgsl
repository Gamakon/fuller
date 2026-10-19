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
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    for (var i: u32 = li; i < 1024u; i += 256u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    if r >= buffer_5_global[0u] {
        return;
    }
    let base = r * 10u;
    let k1 = fmix(buffer_5_global[1u] ^ r);
    let khi = buffer_5_global[2u];

    var acc = 0f;
    var comp = 0f;
    var accx = 0f;
    for (var s: u32 = 0u; s < 10u; s++) {
        let x = buffer_1_global[buffer_0_global[base + s]];
        let w = cb[fmix(k1 ^ (khi ^ s)) >> 22u];
        let p = w * x - comp;
        let t = acc + p;
        comp = (t - acc) - p;
        acc = t;
        accx += x;
    }

    let zh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = cb[zh >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
