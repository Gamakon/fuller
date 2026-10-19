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

@compute
@workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    cb[li] = buffer_2_global[li];
    cb[li + 256u] = buffer_2_global[li + 256u];
    cb[li + 512u] = buffer_2_global[li + 512u];
    cb[li + 768u] = buffer_2_global[li + 768u];
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if r >= width {
        return;
    }
    let klo = buffer_5_global[1u];
    let khi = buffer_5_global[2u];
    let base = r * 10u;

    var src: array<u32, 10>;
    var st: array<f32, 10>;
    for (var s: u32 = 0u; s < 10u; s++) {
        src[s] = buffer_0_global[base + s];
    }
    for (var s: u32 = 0u; s < 10u; s++) {
        st[s] = buffer_1_global[src[s]];
    }

    let h0 = fmix(klo ^ r);
    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s: u32 = 0u; s < 10u; s++) {
        let idx = fmix(h0 ^ (khi ^ s)) >> 22u;
        let v = st[s];
        let term = cb[idx] * v - comp;
        let t = sum + term;
        comp = (t - sum) - term;
        sum = t;
        accx = accx + v;
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let nt = buffer_6_global[3u];
    let z = cb[fmix(fmix(nlo ^ r) ^ (nhi ^ nt)) >> 22u];

    let leak = buffer_7_global[2u];
    let pre = buffer_7_global[1u] * sum + buffer_7_global[0u] * accx + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
