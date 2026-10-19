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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // Even workgroups handle two workgroups' worth of rows, two adjacent rows per thread;
    // odd workgroups have nothing to do (uniform per workgroup, before any barrier).
    if ((wid.x & 1u) != 0u) {
        return;
    }
    for (var k = 0u; k < 4u; k = k + 1u) {
        let j = lid.x + k * 256u;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nlo = buffer_6_global[1u];
    let nmix = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let base = wid.x * 256u + lid.x * 2u + off;

    for (var q = 0u; q < 2u; q = q + 1u) {
        let r = base + q;
        if (r < n) {
            let h0 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var acc = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let xv = buffer_1_global[buffer_0_global[rb + s]];
                let h = fmix(h0 ^ (key_hi ^ s));
                let t = wg_a[h >> 22u] * xv;
                let y = t - comp;
                let sum = acc + y;
                comp = (sum - acc) - y;
                acc = sum;
                accx = accx + xv;
            }
            let nz = wg_a[fmix(fmix(nlo ^ r) ^ nmix) >> 22u];
            let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * nz;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
