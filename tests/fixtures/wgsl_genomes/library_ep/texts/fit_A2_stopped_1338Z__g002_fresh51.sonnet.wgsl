// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i] = 0.0;
    }
    for (var t = 0u; t < m; t = t + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 256u * k;
            let tt = e >> 6u;
            let cc = e & 63u;
            let item = t + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let gc = col0 + cc;
                if (gc < d) {
                    va = phi[item * d + gc];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[item * d + gr];
                }
            }
            sa[e] = va;
            sb[e] = vb;
        }
        workgroupBarrier();
        var tmp: array<f32, 16>;
        for (var i = 0u; i < 16u; i = i + 1u) {
            tmp[i] = 0.0;
        }
        for (var k = 0u; k < 16u; k = k + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                a[j] = sa[k * 64u + lid.x + 16u * j];
                b[j] = sb[k * 64u + lid.y + 16u * j];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    tmp[i * 4u + j] = tmp[i * 4u + j] + b[i] * a[j];
                }
            }
        }
        for (var i = 0u; i < 16u; i = i + 1u) {
            acc[i] = acc[i] + tmp[i];
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
