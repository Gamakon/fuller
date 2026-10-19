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
fn generated_step(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var t = 0u; t < 1024u; t += 256u) {
        wg_a[t + lidx] = buffer_2_global[t + lidx];
    }
    workgroupBarrier();

    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let n_lo = buffer_6_global[1u];
    let n_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let wflat = wid.y * nwg.x + wid.x;
    let base = wflat * 1024u;

    for (var j = 0u; j < 4u; j += 1u) {
        let id = base + j * 256u + lidx;
        let r = id + off;
        if (r < width) {
            let hr = fmix(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var ax = 0.0;
            for (var k = 0u; k < 10u; k += 1u) {
                let x = buffer_1_global[buffer_0_global[rb + k]];
                let h = fmix(hr ^ (key_hi ^ k));
                let p = wg_a[h >> 22u] * x - comp;
                let ns = sum + p;
                comp = (ns - sum) - p;
                sum = ns;
                ax += x;
            }
            let z = wg_a[fmix(fmix(n_lo ^ r) ^ n_tail) >> 22u];
            let pre = scale * sum + loc * ax + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (!(pre > 43.0)) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
