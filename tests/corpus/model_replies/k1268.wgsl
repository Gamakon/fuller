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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    for (var i = 0u; i < 4u; i += 1u) {
        let j = i * 256u + lid;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let id = global_id.x + global_id.y * num_workgroups.x * 256u;
    let n = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let base = id * 2u + off;
    if base >= n {
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
    let omleak = 1.0 - leak;
    let nk = nhi ^ nt;

    for (var k = 0u; k < 2u; k += 1u) {
        let r = base + k;
        if r < n {
            let h0 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var s = 0.0;
            var c = 0.0;
            var ax = 0.0;
            for (var e = 0u; e < 10u; e += 1u) {
                let src = buffer_0_global[rb + e];
                let xv = buffer_1_global[src];
                let w = cb[fmix(h0 ^ (key_hi ^ e)) >> 22u];
                let y = w * xv - c;
                let t = s + y;
                c = (t - s) - y;
                s = t;
                ax += xv;
            }
            let z = cb[fmix(fmix(nlo ^ r) ^ nk) >> 22u];
            let pre = scale * s + loc * ax + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
