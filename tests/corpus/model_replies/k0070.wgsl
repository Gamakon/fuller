// grid 32 8 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 1024>;

@compute @workgroup_size(32, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let cx0 = wid.x * 32u;
    let ry0 = wid.y * 8u;
    let lx = lid.x;
    let ly = lid.y;
    let l = ly * 32u + lx;
    let lr = ry0 + ly;
    let c = cx0 + lx;
    let valid = (lr < cfg.r_rows) && (c < d) && ((cfg.r_off + lr) < d);
    var acc: f32 = 0.0;

    let a_rr = l & 7u;
    let a_tt = l >> 3u;
    let a_lr = ry0 + a_rr;
    let a_ok = (a_lr < cfg.r_rows) && ((cfg.r_off + a_lr) < d);
    let a_gr = cfg.r_off + a_lr;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // load row-side tile
        {
            let t = t0 + a_tt;
            var v: f32 = 0.0;
            if (t < m && a_ok) { v = phi[t * d + a_gr]; }
            wg_a[l & 255u] = v;
        }
        // load column-side tile
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = l + k * 256u;
            let tt = idx >> 5u;
            let col = cx0 + (idx & 31u);
            let t = t0 + tt;
            var v: f32 = 0.0;
            if (t < m && col < d) { v = phi[t * d + col]; }
            wg_b[idx & 1023u] = v;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 32u; tt = tt + 1u) {
            acc = acc + wg_a[(tt * 8u + ly) & 255u] * wg_b[(tt * 32u + lx) & 1023u];
        }
        workgroupBarrier();
        t0 = t0 + 32u;
    }
    if (valid) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
