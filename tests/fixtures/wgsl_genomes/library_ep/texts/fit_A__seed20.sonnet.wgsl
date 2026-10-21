// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ta: array<f32, 512>;
var<workgroup> tb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let col = wg.x * 16u + lx;
    let lr = wg.y * 16u + ly;
    let gr = cfg.r_off + lr;
    // row index used for the cooperative load of the A tile
    let lrow = wg.y * 16u + lx;
    let grow = cfg.r_off + lrow;
    let a_ok = (lrow < cfg.r_rows) && (grow < d);
    let b_ok = col < d;
    var acc: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let k = ly + 16u * j;
            let t = t0 + k;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (a_ok) { av = phi[base + grow]; }
                if (b_ok) { bv = phi[base + col]; }
            }
            ta[k * 16u + lx] = av;
            tb[k * 16u + lx] = bv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = fma(ta[k * 16u + ly], tb[k * 16u + lx], acc);
        }
        workgroupBarrier();
    }
    if (lr < cfg.r_rows && col < d && gr < d) {
        let idx = lr * d + col;
        h[idx] = h[idx] + acc;
    }
}
