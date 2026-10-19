// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    let tid = lid.y * 16u + lid.x;
    let tl = tid >> 4u;
    let c4 = tid & 15u;
    let lc0 = cbase + c4 * 4u;
    let lr0 = rbase + c4 * 4u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + tl;
        let tv = t < m;
        let base = t * d;
        var vb = vec4<f32>(0.0);
        var va = vec4<f32>(0.0);
        if (tv) {
            if (lc0 + 3u < d) {
                vb = vec4<f32>(phi[base + lc0], phi[base + lc0 + 1u], phi[base + lc0 + 2u], phi[base + lc0 + 3u]);
            } else {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    if (lc0 + j < d) { vb[j & 3u] = phi[base + lc0 + j]; }
                }
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                let lr = lr0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va[j & 3u] = phi[base + gr]; }
            }
        }
        sb[tid & 255u] = vb;
        sa[tid & 255u] = va;
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let b = sb[(k * 16u + lid.x) & 255u];
            let a = sa[(k * 16u + lid.y) & 255u];
            acc[0u] = acc[0u] + a.x * b;
            acc[1u] = acc[1u] + a.y * b;
            acc[2u] = acc[2u] + a.z * b;
            acc[3u] = acc[3u] + a.w * b;
        }
        workgroupBarrier();
    }

    let colb = cbase + lid.x * 4u;
    let rowb = rbase + lid.y * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowb + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = colb + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j & 3u];
                }
            }
        }
    }
}
