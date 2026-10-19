@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> s_cb: array<f32, 1024>;
var<workgroup> s_src: array<u32, 2560>;

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
fn generated_step(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    let n = buffer_5_global[0u];
    let base = wid.x * 256u + buffer_6_global[0u];

    for (var i = 0u; i < 4u; i += 1u) {
        let j = i * 256u + lid;
        s_cb[j] = buffer_2_global[j];
    }
    for (var i = 0u; i < 10u; i += 1u) {
        let j = i * 256u + lid;
        let row = base + j / 10u;
        if (row < n) {
            s_src[j] = buffer_0_global[base * 10u + j];
        }
    }
    workgroupBarrier();

    let r = base + lid;
    if (r >= n) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let sb = lid * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let x = buffer_1_global[s_src[sb + s]];
        let h = fmix(h0 ^ (key_hi ^ s));
        let term = s_cb[h >> 22u] * x - comp;
        let t = sum + term;
        comp = (t - sum) - term;
        sum = t;
        accx = accx + x;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let tt = buffer_6_global[3u];
    let hz = fmix(fmix(n_lo ^ r) ^ (n_hi ^ tt));
    let z = s_cb[hz >> 22u];

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
