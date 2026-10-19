var<workgroup> wg_a: array<f32, 1024>;

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

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    for (var k = 0u; k < 4u; k = k + 1u) {
        wg_a[k * 256u + lidx] = buffer_2_global[k * 256u + lidx];
    }
    workgroupBarrier();

    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off0 = buffer_6_global[0u];
    let noise_lo = buffer_6_global[1u];
    let noise_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let omleak = 1.0 - leak;

    let base = wid.x * 1024u + lidx;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let r = base + j * 256u + off0;
        if (r < n) {
            let h1 = fmix(key_lo ^ r);
            let b = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let x = buffer_1_global[buffer_0_global[b + s]];
                let hh = fmix(h1 ^ (key_hi ^ s));
                let t = wg_a[(hh >> 22u) & 1023u] * x - comp;
                let ns = sum + t;
                comp = (ns - sum) - t;
                sum = ns;
                accx = accx + x;
            }
            let hn = fmix(fmix(noise_lo ^ r) ^ noise_tail);
            let z = wg_a[(hn >> 22u) & 1023u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * th;
        }
    }
}
