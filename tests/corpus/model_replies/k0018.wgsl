// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<vec4<f32>, 256>;
var<workgroup> sB: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    let ld_t = lid >> 4u;
    let ld_q = (lid & 15u) * 4u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ld_t;
        let tv = t < m;
        let base = t * d;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (tv) {
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                let lr = row0 + ld_q + k;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va[k] = phi[base + gr];
                }
                let c = col0 + ld_q + k;
                if (c < d) {
                    vb[k] = phi[base + c];
                }
            }
        }
        sA[lid & 255u] = va;
        sB[lid & 255u] = vb;
        workgroupBarrier();

        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let a = sA[(tt * 16u + ty) & 255u];
            let b = sB[(tt * 16u + tx) & 255u];
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let cb = col0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let rb = lr * d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + j;
                if (c < d) {
                    h[rb + c] = h[rb + c] + v[j];
                }
            }
        }
    }
}
