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
    var acc: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }
    let lk = lidx >> 4u;
    let lj = (lidx & 15u) * 4u;
    let colb = wid.x * 64u + lj;
    let rowl = wid.y * 64u + lj;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + lk;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = colb + i;
                if (c < d) {
                    a[i] = phi[base + c];
                }
                let lr = rowl + i;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    b[i] = phi[base + gr];
                }
            }
        }
        wa[lidx] = a;
        wb[lidx] = b;
        workgroupBarrier();
        var tl: array<vec4<f32>, 4>;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            tl[i] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = wa[k * 16u + lid.x];
            let bv = wb[k * 16u + lid.y];
            tl[0] = tl[0] + av * bv.x;
            tl[1] = tl[1] + av * bv.y;
            tl[2] = tl[2] + av * bv.z;
            tl[3] = tl[3] + av * bv.w;
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            acc[i] = acc[i] + tl[i];
        }
        workgroupBarrier();
    }
    let c0 = (wid.x * 16u + lid.x) * 4u;
    let r0 = (wid.y * 16u + lid.y) * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            let v = acc[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = c0 + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
