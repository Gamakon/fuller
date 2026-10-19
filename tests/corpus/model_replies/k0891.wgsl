// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = t + (idx >> 6u);
            let cl = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (it < m) {
                let c = colBase + cl;
                if (c < d) {
                    fb = phi[it * d + c];
                }
                let lr = rowBase + cl;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    fa = phi[it * d + gr];
                }
            }
            wg_a[idx & 1023u] = fa;
            wg_b[idx & 1023u] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let oa = kk * 64u + lid.y * 4u;
            let ob = kk * 64u + lid.x * 4u;
            let r0 = wg_a[oa & 1023u];
            let r1 = wg_a[(oa + 1u) & 1023u];
            let r2 = wg_a[(oa + 2u) & 1023u];
            let r3 = wg_a[(oa + 3u) & 1023u];
            let cb = vec4<f32>(wg_b[ob & 1023u], wg_b[(ob + 1u) & 1023u], wg_b[(ob + 2u) & 1023u], wg_b[(ob + 3u) & 1023u]);
            a0 = a0 + r0 * cb;
            a1 = a1 + r1 * cb;
            a2 = a2 + r2 * cb;
            a3 = a3 + r3 * cb;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            var row = a0;
            if (i == 1u) { row = a1; }
            if (i == 2u) { row = a2; }
            if (i == 3u) { row = a3; }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x * 4u + j;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + row[j];
                }
            }
        }
    }
}
