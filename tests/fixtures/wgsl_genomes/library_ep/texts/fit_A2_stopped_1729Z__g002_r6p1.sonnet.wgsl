// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        acc[z] = 0.0;
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tt: u32 = 0u; tt < ntiles; tt = tt + 1u) {
        let t0 = tt * 16u;
        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
            let idx = lidx + s * 256u;
            let kk = idx / 64u;
            let i = idx % 64u;
            let t = t0 + kk;
            let lr = row0 + i;
            let gr = cfg.r_off + lr;
            let c = col0 + i;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m && lr < cfg.r_rows && gr < cfg.d) {
                va = phi[t * cfg.d + gr];
            }
            if (t < cfg.m && c < cfg.d) {
                vb = phi[t * cfg.d + c];
            }
            sa[idx] = va;
            sb[idx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                ra[j] = sa[k * 64u + lid.y * 4u + j];
                rb[j] = sb[k * 64u + lid.x * 4u + j];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    acc[j * 4u + i] = acc[j * 4u + i] + ra[j] * rb[i];
                }
            }
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let lr = row0 + lid.y * 4u + j;
            let c = col0 + lid.x * 4u + i;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + acc[j * 4u + i];
            }
        }
    }
}
