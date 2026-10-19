// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16> = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let nt = (cfg.m + 15u) / 16u;
    for (var tt: u32 = 0u; tt < nt; tt = tt + 1u) {
        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let idx = lidx + l * 256u;
            let kk = idx / 64u;
            let cc = idx % 64u;
            let t = tt * 16u + kk;
            let col = col0 + cc;
            let lrow = row0 + cc;
            let grow = cfg.r_off + lrow;
            var bv: f32 = 0.0;
            var av: f32 = 0.0;
            if (t < cfg.m && col < cfg.d) {
                bv = phi[t * cfg.d + col];
            }
            if (t < cfg.m && lrow < cfg.r_rows && grow < cfg.d) {
                av = phi[t * cfg.d + grow];
            }
            wg_b[idx] = bv;
            wg_a[idx] = av;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[k * 64u + lid.y * 4u + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b = wg_b[k * 64u + lid.x * 4u + j];
                    acc[i * 4u + j] = acc[i * 4u + j] + a * b;
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + lid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + acc[i * 4u + j];
            }
        }
    }
}
