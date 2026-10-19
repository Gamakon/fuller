// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let lr_load = wid.y * 16u + lx;
    let gr_load = cfg.r_off + lr_load;
    let row_ok = (lr_load < cfg.r_rows) && (gr_load < d);
    let c_load = wid.x * 16u + lx;
    let col_ok = c_load < d;
    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var hh: u32 = 0u; hh < 2u; hh = hh + 1u) {
            let tt = ly + hh * 16u;
            let t = t0 + tt;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (row_ok) { av = phi[base + gr_load]; }
                if (col_ok) { bv = phi[base + c_load]; }
            }
            sa[(tt * 16u + lx) & 511u] = av;
            sb[(tt * 16u + lx) & 511u] = bv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = acc + sa[(k * 16u + ly) & 511u] * sb[(k * 16u + lx) & 511u];
        }
        workgroupBarrier();
        t0 = t0 + 32u;
    }
    let lr = wid.y * 16u + ly;
    let c = wid.x * 16u + lx;
    if (lr < cfg.r_rows && c < d && (cfg.r_off + lr) < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
