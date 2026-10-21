// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) { acc[i] = vec4<f32>(0.0); }
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let d = cfg.d;
    let m = cfg.m;
    for (var t = 0u; t < m; t = t + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = idx >> 6u;
            let cc = idx & 63u;
            var va = 0.0;
            var vb = 0.0;
            if (t + tr < m) {
                let base = (t + tr) * d;
                if (colbase + cc < d) { va = phi[base + colbase + cc]; }
                let lr = rowbase + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) { vb = phi[base + cfg.r_off + lr]; }
            }
            let q = (tr * 64u + cc) >> 2u;
            let comp = cc & 3u;
            // write scalar into vec4 slot: use per-component stores via separate arrays packing
            var ua = wa[q];
            var ub = wb[q];
            ua[comp] = va;
            ub[comp] = vb;
            wa[q] = ua;
            wb[q] = ub;
        }
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let a = wa[kk * 16u + lid.x];
            let b = wb[kk * 16u + lid.y];
            p0 = p0 + b.x * a;
            p1 = p1 + b.y * a;
            p2 = p2 + b.z * a;
            p3 = p3 + b.w * a;
        }
        acc[0] = acc[0] + p0;
        acc[1] = acc[1] + p1;
        acc[2] = acc[2] + p2;
        acc[3] = acc[3] + p3;
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[i];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = colbase + lid.x * 4u + j;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + v[j];
                }
            }
        }
    }
}
