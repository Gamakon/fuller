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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if (off >= width) {
        return;
    }
    let n = width - off;
    let q = (n + 3u) / 4u;
    let g = gid.x + gid.y * nwg.x * 256u;
    if (g >= q) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let nt = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = nhi ^ nt;

    for (var k = 0u; k < 4u; k++) {
        let r = g + k * q + off;
        if (r < width) {
            let base = r * 10u;
            let h1 = fmix(key_lo ^ r);
            var sum = 0.0;
            var comp = 0.0;
            var acc_x = 0.0;
            for (var s = 0u; s < 10u; s++) {
                let src = buffer_0_global[base + s];
                let st = buffer_1_global[src];
                let hh = fmix(h1 ^ (key_hi ^ s));
                let w = buffer_2_global[hh >> 22u];
                let y = w * st - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                acc_x = acc_x + st;
            }
            let z = buffer_2_global[fmix(fmix(nlo ^ r) ^ nkey) >> 22u];
            let pre = scale * sum + loc * acc_x + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
