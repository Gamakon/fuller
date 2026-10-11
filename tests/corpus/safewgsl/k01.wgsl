// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ra: array<f32, 512>;
var<workgroup> ca: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let rbase = wg.y * 32u;
    let cbase = wg.x * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = tid + 256u * k;
            let it = e / 32u;
            let col = e % 32u;
            let t = t0 + it;
            var rv: f32 = 0.0;
            var cv: f32 = 0.0;
            if (t < m) {
                let lrr = rbase + col;
                let grr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && grr < d) {
                    rv = phi[t * d + grr];
                }
                let cc = cbase + col;
                if (cc < d) {
                    cv = phi[t * d + cc];
                }
            }
            ra[e] = rv;
            ca[e] = cv;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let r0 = ra[i * 32u + lid.y];
            let r1 = ra[i * 32u + lid.y + 16u];
            let c0 = ca[i * 32u + lid.x];
            let c1 = ca[i * 32u + lid.x + 16u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
    }

    let lr0 = rbase + lid.y;
    let lr1 = lr0 + 16u;
    let cc0 = cbase + lid.x;
    let cc1 = cc0 + 16u;
    let ok0 = lr0 < cfg.r_rows && (cfg.r_off + lr0) < d;
    let ok1 = lr1 < cfg.r_rows && (cfg.r_off + lr1) < d;
    if (ok0 && cc0 < d) {
        let idx = lr0 * d + cc0;
        h[idx] = h[idx] + a00;
    }
    if (ok0 && cc1 < d) {
        let idx = lr0 * d + cc1;
        h[idx] = h[idx] + a01;
    }
    if (ok1 && cc0 < d) {
        let idx = lr1 * d + cc0;
        h[idx] = h[idx] + a10;
    }
    if (ok1 && cc1 < d) {
        let idx = lr1 * d + cc1;
        h[idx] = h[idx] + a11;
    }
}
