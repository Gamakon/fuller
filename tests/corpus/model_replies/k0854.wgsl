// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 512>;
var<workgroup> sB: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let oa = ty * 4u;
    let ob = tx * 4u;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = lin + e * 256u;
            let tl = i >> 6u;
            let k = i & 63u;
            let t = t0 + tl;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = rowBase + k;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colBase + k;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sA[i] = va;
            sB[i] = vb;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let pa = k * 64u + oa;
            let pb = k * 64u + ob;
            let a = vec4<f32>(sA[pa], sA[pa + 1u], sA[pa + 2u], sA[pa + 3u]);
            let b = vec4<f32>(sB[pb], sB[pb + 1u], sB[pb + 2u], sB[pb + 3u]);
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = acc0;
            if (i == 1u) { v = acc1; }
            if (i == 2u) { v = acc2; }
            if (i == 3u) { v = acc3; }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}