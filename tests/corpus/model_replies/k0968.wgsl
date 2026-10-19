// grid 256 1 1 1
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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = 0u; k < 4u; k = k + 1u) {
        let c = lidx + k * 256u;
        wg_a[c] = buffer_2_global[c];
    }
    workgroupBarrier();

    let id = gid.x + gid.y * nwg.x * 256u;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = n_hi ^ n_t;

    let base = id * 4u + off;
    if (id * 4u >= width) {
        return;
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let r = base + j;
        if (r < width) {
            let h0 = fmix(key_lo ^ r);
            var accz = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            let rb = r * 10u;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let x = buffer_1_global[buffer_0_global[rb + s]];
                let hh = fmix(h0 ^ (key_hi ^ s));
                let term = wg_a[(hh >> 22u) & 1023u] * x - comp;
                let t = accz + term;
                comp = (t - accz) - term;
                accz = t;
                accx = accx + x;
            }
            let hn = fmix(fmix(n_lo ^ r) ^ nkey);
            let z = wg_a[(hn >> 22u) & 1023u];
            let pre = scale * accz + loc * accx + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
