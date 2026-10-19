var<workgroup> cb: array<f32, 1024>;

@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

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

@compute @workgroup_size(64, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = lid.x; i < 1024u; i = i + 64u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let id = gid.x;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let noise_lo = buffer_6_global[1u];
    let nkey = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let oneml = 1.0 - leak;

    let base = id * 4u + off;
    if base >= width {
        return;
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let r = base + j;
        if r >= width {
            break;
        }
        let h1 = fmix(key_lo ^ r);
        let rb = r * 10u;
        var sum = 0.0;
        var comp = 0.0;
        var accx = 0.0;
        for (var s = 0u; s < 10u; s = s + 1u) {
            let xv = buffer_1_global[buffer_0_global[rb + s]];
            let w = cb[fmix(h1 ^ (key_hi ^ s)) >> 22u];
            let y = w * xv - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
            accx = accx + xv;
        }
        let z = cb[fmix(fmix(noise_lo ^ r) ^ nkey) >> 22u];
        let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
        buffer_4_global[r] = oneml * buffer_1_global[r] + leak * safe_tanh(pre);
    }
}
