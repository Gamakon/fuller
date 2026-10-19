// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }
    let rbase = wid.y * 64u;
    let cbase = wid.x * 64u;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        for (var q = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let kk = idx / 64u;
            let cc = idx % 64u;
            let item = t * 16u + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < cfg.m) {
                let lrow = rbase + cc;
                if (lrow < cfg.r_rows && cfg.r_off + lrow < cfg.d) {
                    va = phi[item * cfg.d + cfg.r_off + lrow];
                }
                if (cbase + cc < cfg.d) {
                    vb = phi[item * cfg.d + cbase + cc];
                }
            }
            wa[idx] = va;
            wb[idx] = vb;
        }
        workgroupBarrier();
        var tmp: array<f32, 16>;
        for (var q = 0u; q < 16u; q = q + 1u) { tmp[q] = 0.0; }
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i] = wa[o + lid.y + 16u * i];
                b[i] = wb[o + lid.x + 16u * i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    tmp[i * 4u + j] = tmp[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        for (var q = 0u; q < 16u; q = q + 1u) { acc[q] = acc[q] + tmp[q]; }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + lid.y + 16u * i;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let c = cbase + lid.x + 16u * j;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                let ix = lr * cfg.d + c;
                h[ix] = h[ix] + acc[i * 4u + j];
            }
        }
    }
}
