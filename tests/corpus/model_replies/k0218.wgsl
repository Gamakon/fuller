// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 2048>;
var<workgroup> wb: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var r = 0u; r < 8u; r = r + 1u) {
        acc[r & 7u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t = 0u; t < m; t = t + 32u) {
        // coalesced cooperative load: consecutive lanes read consecutive floats
        for (var j = 0u; j < 16u; j = j + 1u) {
            let e = lidx + 128u * j;
            let it = e >> 6u;
            let col = e & 63u;
            let item = t + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let rowOff = item * d;
                let gc = colBase + col;
                if (gc < d) {
                    va = phi[rowOff + gc];
                }
                let lrow = rowBase + col;
                if (lrow < cfg.r_rows && (cfg.r_off + lrow) < d) {
                    vb = phi[rowOff + cfg.r_off + lrow];
                }
            }
            wa[e & 2047u] = va;
            wb[e & 2047u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 32u; k = k + 1u) {
            let base = k * 64u;
            let ab = base + lid.x * 4u;
            let a = vec4<f32>(wa[ab & 2047u], wa[(ab + 1u) & 2047u], wa[(ab + 2u) & 2047u], wa[(ab + 3u) & 2047u]);
            let bb = base + lid.y * 8u;
            for (var r = 0u; r < 8u; r = r + 1u) {
                let bv = wb[(bb + r) & 2047u];
                acc[r & 7u] = acc[r & 7u] + vec4<f32>(bv) * a;
            }
        }
        workgroupBarrier();
    }
    let col0 = colBase + lid.x * 4u;
    for (var r = 0u; r < 8u; r = r + 1u) {
        let row = rowBase + lid.y * 8u + r;
        if (row < cfg.r_rows && (cfg.r_off + row) < d) {
            let v = acc[r & 7u];
            let o = row * d + col0;
            if (col0 < d) { h[o] = h[o] + v.x; }
            if (col0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (col0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (col0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
