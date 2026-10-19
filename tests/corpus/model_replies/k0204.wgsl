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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let offs = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let wgid = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let first = wgid * 1024u;
    // uniform per workgroup
    if (first + offs >= width) {
        return;
    }
    for (var i = 0u; i < 4u; i += 1u) {
        let j = lid.x + i * 256u;
        wg_a[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let base = first + lid.x * 4u;
    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + j + offs;
        if (r < width) {
            let h1 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let x = buffer_1_global[src];
                let hh = fmix(h1 ^ (key_hi ^ s));
                let w = wg_a[hh >> 22u];
                let y = w * x - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                accx = accx + x;
            }
            let zh = fmix(fmix(n_lo ^ r) ^ n_tail);
            let z = wg_a[zh >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
