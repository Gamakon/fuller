// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let li = lidx >> 4u;
    let q = lidx & 15u;
    let lc = colBase + q * 4u;
    let lrw = rowBase + q * 4u;
    var tot: array<vec4<f32>, 4>;
    tot[0] = vec4<f32>(0.0);
    tot[1] = vec4<f32>(0.0);
    tot[2] = vec4<f32>(0.0);
    tot[3] = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + li;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                if (lc + e < d) {
                    a[e] = phi[base + lc + e];
                }
                if (lrw + e < cfg.r_rows && cfg.r_off + lrw + e < d) {
                    b[e] = phi[base + cfg.r_off + lrw + e];
                }
            }
        }
        wa[lidx] = a;
        wb[lidx] = b;
        workgroupBarrier();
        var tmp: array<vec4<f32>, 4>;
        tmp[0] = vec4<f32>(0.0);
        tmp[1] = vec4<f32>(0.0);
        tmp[2] = vec4<f32>(0.0);
        tmp[3] = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = wa[k * 16u + tx];
            let bv = wb[k * 16u + ty];
            tmp[0] = tmp[0] + vec4<f32>(bv.x) * av;
            tmp[1] = tmp[1] + vec4<f32>(bv.y) * av;
            tmp[2] = tmp[2] + vec4<f32>(bv.z) * av;
            tmp[3] = tmp[3] + vec4<f32>(bv.w) * av;
        }
        tot[0] = tot[0] + tmp[0];
        tot[1] = tot[1] + tmp[1];
        tot[2] = tot[2] + tmp[2];
        tot[3] = tot[3] + tmp[3];
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + tot[i][j];
                }
            }
        }
    }
}
