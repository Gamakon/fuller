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
    h ^= h >> 16u;
    h *= 2246822507u;
    h ^= h >> 13u;
    h *= 3266489909u;
    h ^= h >> 16u;
    return h;
}

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(128, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let l = lid.x;
    for (var i = 0u; i < 8u; i += 1u) {
        let j = i * 128u + l;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();
    if l >= 32u {
        return;
    }
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_w = buffer_6_global[2u] ^ buffer_6_global[3u];
    let sc = buffer_7_global[1u];
    let loc = buffer_7_global[0u];
    let leak = buffer_7_global[2u];
    let sg = buffer_7_global[4u];
    let base = wid.x * 128u + l * 4u + off;

    for (var k = 0u; k < 4u; k += 1u) {
        let r = base + k;
        if r < width {
            let h0 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var accz = 0f;
            var comp = 0f;
            var accx = 0f;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let x = buffer_1_global[src];
                let hh = fmix(h0 ^ (key_hi ^ s));
                let w = cb[hh >> 22u];
                let y = w * x - comp;
                let t = accz + y;
                comp = (t - accz) - y;
                accz = t;
                accx += x;
            }
            let z = cb[fmix(fmix(n_lo ^ r) ^ n_w) >> 22u];
            let pre = sc * accz + loc * accx + buffer_3_global[r] + sg * z;
            buffer_4_global[r] = (1f - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
