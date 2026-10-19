// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let c0 = wid.x * 32u;
    let r0 = wid.y * 32u;
    var p00: f32 = 0.0; var p01: f32 = 0.0; var p10: f32 = 0.0; var p11: f32 = 0.0;
    var q00: f32 = 0.0; var q01: f32 = 0.0; var q10: f32 = 0.0; var q11: f32 = 0.0;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let e = lidx + 256u * j;
            let it = t + (e >> 5u);
            let cc = e & 31u;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (it < cfg.m) {
                let col = c0 + cc;
                if (col < cfg.d) { a = phi[it * cfg.d + col]; }
                let lr = r0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) { b = phi[it * cfg.d + gr]; }
            }
            wa[e] = a;
            wb[e] = b;
        }
        workgroupBarrier();
        var e00: f32 = 0.0; var e01: f32 = 0.0; var e10: f32 = 0.0; var e11: f32 = 0.0;
        var o00: f32 = 0.0; var o01: f32 = 0.0; var o10: f32 = 0.0; var o11: f32 = 0.0;
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            let bs = k * 32u;
            let x0 = wa[bs + lid.x];
            let x1 = wa[bs + lid.x + 16u];
            let y0 = wb[bs + lid.y];
            let y1 = wb[bs + lid.y + 16u];
            e00 = e00 + y0 * x0;
            e01 = e01 + y0 * x1;
            e10 = e10 + y1 * x0;
            e11 = e11 + y1 * x1;
            let bs2 = bs + 32u;
            let u0 = wa[bs2 + lid.x];
            let u1 = wa[bs2 + lid.x + 16u];
            let w0 = wb[bs2 + lid.y];
            let w1 = wb[bs2 + lid.y + 16u];
            o00 = o00 + w0 * u0;
            o01 = o01 + w0 * u1;
            o10 = o10 + w1 * u0;
            o11 = o11 + w1 * u1;
        }
        if (((t >> 5u) & 1u) == 0u) {
            p00 = p00 + (e00 + o00);
            p01 = p01 + (e01 + o01);
            p10 = p10 + (e10 + o10);
            p11 = p11 + (e11 + o11);
        } else {
            q00 = q00 + (e00 + o00);
            q01 = q01 + (e01 + o01);
            q10 = q10 + (e10 + o10);
            q11 = q11 + (e11 + o11);
        }
        workgroupBarrier();
    }
    let lr0 = r0 + lid.y;
    let lr1 = lr0 + 16u;
    let cA = c0 + lid.x;
    let cB = cA + 16u;
    if (lr0 < cfg.r_rows && cfg.r_off + lr0 < cfg.d) {
        if (cA < cfg.d) { let idx = lr0 * cfg.d + cA; h[idx] = h[idx] + (p00 + q00); }
        if (cB < cfg.d) { let idx = lr0 * cfg.d + cB; h[idx] = h[idx] + (p01 + q01); }
    }
    if (lr1 < cfg.r_rows && cfg.r_off + lr1 < cfg.d) {
        if (cA < cfg.d) { let idx = lr1 * cfg.d + cA; h[idx] = h[idx] + (p10 + q10); }
        if (cB < cfg.d) { let idx = lr1 * cfg.d + cB; h[idx] = h[idx] + (p11 + q11); }
    }
}
