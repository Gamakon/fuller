var<workgroup> cb: array<f32, 1024>;

@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

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
    if (x > 43.0) {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let wgi = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    let base = wgi * 1024u;
    // workgroup-uniform early exit
    if (base + off >= n) {
        return;
    }
    for (var k = 0u; k < 1024u; k += 256u) {
        let j = k + lid.x;
        cb[j] = buffer_2_global[j];
    }
    workgroupBarrier();

    let k1 = buffer_5_global[1u];
    let k2 = buffer_5_global[2u];
    let nlo = buffer_6_global[1u];
    let nn = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let omleak = 1.0 - leak;

    for (var j = 0u; j < 4u; j += 1u) {
        let r = base + lid.x * 4u + j + off;
        if (r < n) {
            let h1 = fmix32(k1 ^ r);
            let sb = r * 10u;
            var acc = 0.0;
            var comp = 0.0;
            var accx = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[sb + s];
                let xv = buffer_1_global[src];
                let hh = fmix32(h1 ^ (k2 ^ s));
                let y = cb[hh >> 22u] * xv - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                accx = accx + xv;
            }
            let zh = fmix32(fmix32(nlo ^ r) ^ nn);
            let z = cb[zh >> 22u];
            let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
