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
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let l = lid.x;
    for (var k = 0u; k < 1024u; k += 256u) {
        wg_a[k + l] = buffer_2_global[k + l];
    }
    workgroupBarrier();

    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let noise_lo = buffer_6_global[1u];
    let nmix = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scl = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let wbase = wid.x * 1024u;
    if (wbase + off >= width) {
        return;
    }

    for (var j = 0u; j < 4u; j += 1u) {
        let r = wbase + j * 256u + l + off;
        if (r < width) {
            let h0 = fmix(key_lo ^ r);
            let b = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let xv = buffer_1_global[buffer_0_global[b + s]];
                let hh = fmix(h0 ^ (key_hi ^ s)) >> 22u;
                let p = wg_a[hh] * xv;
                let y = p - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                ax = ax + xv;
            }
            let z = wg_a[fmix(fmix(noise_lo ^ r) ^ nmix) >> 22u];
            let pre = scl * sum + loc * ax + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
