// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    for (var z: u32 = 0u; z < 64u; z = z + 1u) {
        pa[z] = 0.0;
    }
    let tx = lid.x;
    let ty = lid.y;
    let nt = (cfg.m + 7u) / 8u;
    for (var t0: u32 = 0u; t0 < nt; t0 = t0 + 1u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 128u;
            let kk = idx / 64u;
            let cc = idx % 64u;
            let item = t0 * 8u + kk;
            let col = wid.x * 64u + cc;
            let lrow = wid.y * 64u + cc;
            let grow = cfg.r_off + lrow;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (item < cfg.m) {
                if (col < cfg.d) {
                    vb = phi[item * cfg.d + col];
                }
                if (lrow < cfg.r_rows && grow < cfg.d) {
                    va = phi[item * cfg.d + grow];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let a = wg_a[kk * 64u + ty * 8u + j];
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let b = wg_b[kk * 64u + tx * 4u + i];
                    pa[j * 4u + i] = pa[j * 4u + i] + a * b;
                }
            }
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let lr = wid.y * 64u + ty * 8u + j;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let c = wid.x * 64u + tx * 4u + i;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[j * 4u + i];
            }
        }
    }
}
