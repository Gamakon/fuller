// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<vec4<f32>, 256>;
var<workgroup> sB: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let lk = tid >> 4u;
    let lq = tid & 15u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + lk;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let lr = rowBase + lq * 4u + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va[j] = phi[base + gr];
                }
                let c = colBase + lq * 4u + j;
                if (c < d) {
                    vb[j] = phi[base + c];
                }
            }
        }
        sA[tid & 255u] = va;
        sB[tid & 255u] = vb;
        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sA[(k * 16u + ty) & 255u];
            let b = sB[(k * 16u + tx) & 255u];
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let ro = lr * d;
            if (c0 + 3u < d) {
                h[ro + c0] = h[ro + c0] + v.x;
                h[ro + c0 + 1u] = h[ro + c0 + 1u] + v.y;
                h[ro + c0 + 2u] = h[ro + c0 + 2u] + v.z;
                h[ro + c0 + 3u] = h[ro + c0 + 3u] + v.w;
            } else {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (c0 + j < d) {
                        h[ro + c0 + j] = h[ro + c0 + j] + v[j];
                    }
                }
            }
        }
    }
}
