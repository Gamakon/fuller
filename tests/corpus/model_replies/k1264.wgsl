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

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    for (var i = 0u; i < 4u; i += 1u) {
        let j = lid + i * 256u;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    let n = buffer_5_global[0u];
    if r >= n {
        return;
    }

    let key_hi = buffer_5_global[2u];
    let h1 = fmix(buffer_5_global[1u] ^ r);
    let base = r * 10u;

    var az0 = 0.0;
    var az1 = 0.0;
    var ax0 = 0.0;
    var ax1 = 0.0;
    for (var s = 0u; s < 10u; s += 2u) {
        let src0 = buffer_0_global[base + s];
        let src1 = buffer_0_global[base + s + 1u];
        let x0 = buffer_1_global[src0];
        let x1 = buffer_1_global[src1];
        let c0 = cb[fmix(h1 ^ (key_hi ^ s)) >> 22u];
        let c1 = cb[fmix(h1 ^ (key_hi ^ (s + 1u))) >> 22u];
        az0 = fma(c0, x0, az0);
        az1 = fma(c1, x1, az1);
        ax0 += x0;
        ax1 += x1;
    }

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * (az0 + az1) + loc * (ax0 + ax1) + buffer_3_global[r] + sigma * nz;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
