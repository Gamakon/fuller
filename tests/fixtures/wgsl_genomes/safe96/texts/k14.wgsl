// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> colT: array<vec4<f32>, 256>;
var<workgroup> rowT: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let li = ly * 16u + lx;
    let colBase = wg.x * 64u;
    let rowBase = wg.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // load column tile: 16 items x 16 vec4 (64 columns)
        {
            let tt = li / 16u;          // 0..15
            let cv = li % 16u;          // vec4 index
            let t = t0 + tt;
            let c = colBase + cv * 4u;
            var v = vec4<f32>(0.0);
            if (t < m) {
                let base = t * d;
                if (c < d) { v.x = phi[base + c]; }
                if (c + 1u < d) { v.y = phi[base + c + 1u]; }
                if (c + 2u < d) { v.z = phi[base + c + 2u]; }
                if (c + 3u < d) { v.w = phi[base + c + 3u]; }
            }
            colT[li] = v;
        }
        // load row tile: 16 items x 64 rows (4 per thread)
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = li + k * 256u;
            let tt = idx / 64u;
            let rr = idx % 64u;
            let t = t0 + tt;
            let lr = rowBase + rr;
            let gr = cfg.r_off + lr;
            var x: f32 = 0.0;
            if (t < m && lr < cfg.r_rows && gr < d) {
                x = phi[t * d + gr];
            }
            rowT[idx] = x;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let b = colT[tt * 16u + lx];
            let rb = tt * 64u + ly * 4u;
            a0 = a0 + rowT[rb] * b;
            a1 = a1 + rowT[rb + 1u] * b;
            a2 = a2 + rowT[rb + 2u] * b;
            a3 = a3 + rowT[rb + 3u] * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let c0 = colBase + lx * 4u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + ly * 4u + j;
        if (lr < cfg.r_rows) {
            var a = a0;
            if (j == 1u) { a = a1; }
            if (j == 2u) { a = a2; }
            if (j == 3u) { a = a3; }
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + a.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + a.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + a.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + a.w; }
        }
    }
}
