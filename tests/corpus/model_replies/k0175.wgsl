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

fn fmix32(hin: u32) -> u32 {
    var h = hin;
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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let lid = lid3.x;
    for (var i = 0u; i < 4u; i += 1u) {
        cb[i * 256u + lid] = buffer_2_global[i * 256u + lid];
    }
    workgroupBarrier();

    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
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
    let base = wid.x * 1024u;

    for (var k = 0u; k < 4u; k += 1u) {
        let id = base + k * 256u + lid;
        let r = id + off;
        if (r < n) {
            let h1 = fmix32(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0f;
            var comp = 0f;
            var accx = 0f;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rb + s];
                let xv = buffer_1_global[src];
                let hh = fmix32(h1 ^ (key_hi ^ s));
                let w = cb[hh >> 22u];
                let y = w * xv - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                accx = accx + xv;
            }
            let z = cb[fmix32(fmix32(nlo ^ r) ^ nkey) >> 22u];
            let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
            let xr = buffer_1_global[r];
            buffer_4_global[r] = (1f - leak) * xr + leak * safe_tanh(pre);
        }
    }
}
