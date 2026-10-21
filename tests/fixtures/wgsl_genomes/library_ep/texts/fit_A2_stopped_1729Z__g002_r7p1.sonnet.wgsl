// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        acc[z] = 0.0;
    }
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let nt = (cfg.m + 15u) / 16u;
    for (var tt: u32 = 0u; tt < nt; tt = tt + 1u) {
        let t0 = tt * 16u;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let k = idx / 64u;
            let p = idx % 64u;
            let t = t0 + k;
            let col = colBase + p;
            let lrow = rowBase + p;
            let grow = cfg.r_off + lrow;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < cfg.m && col < cfg.d) {
                vb = phi[t * cfg.d + col];
            }
            if (t < cfg.m && lrow < cfg.r_rows && grow < cfg.d) {
                va = phi[t * cfg.d + grow];
            }
            wg_b[idx] = vb;
            wg_a[idx] = va;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i] = wg_a[k * 64u + lid.y * 4u + i];
                b[i] = wg_b[k * 64u + lid.x * 4u + i];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = colBase + lid.x * 4u + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + acc[i * 4u + j];
            }
        }
    }
}
