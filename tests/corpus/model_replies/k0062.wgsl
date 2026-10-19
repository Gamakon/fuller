// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sc: array<f32, 1024>;
var<workgroup> sr: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let li = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load of 16 items x 64 columns and 16 items x 64 rows
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = li + k * 256u;
            let tt = idx >> 6u;
            let q = idx & 63u;
            let t = t0 + tt;
            var cv = 0.0;
            var rv = 0.0;
            if (t < m) {
                let base = t * d;
                let c = col0 + q;
                if (c < d) {
                    cv = phi[base + c];
                }
                let lr = row0 + q;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    rv = phi[base + gr];
                }
            }
            sc[idx & 1023u] = cv;
            sr[idx & 1023u] = rv;
        }
        workgroupBarrier();

        for (var tt = 0u; tt < 16u; tt = tt + 1u) {
            let cb = tt * 64u + tx * 4u;
            let rb = tt * 64u + ty * 4u;
            let cv = vec4<f32>(
                sc[cb & 1023u],
                sc[(cb + 1u) & 1023u],
                sc[(cb + 2u) & 1023u],
                sc[(cb + 3u) & 1023u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                let rv = sr[(rb + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + rv * cv;
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let a = acc[i & 3u];
            let c = col0 + tx * 4u;
            let ro = lr * d;
            if (c < d) { h[ro + c] = h[ro + c] + a.x; }
            if (c + 1u < d) { h[ro + c + 1u] = h[ro + c + 1u] + a.y; }
            if (c + 2u < d) { h[ro + c + 2u] = h[ro + c + 2u] + a.z; }
            if (c + 3u < d) { h[ro + c + 3u] = h[ro + c + 3u] + a.w; }
        }
    }
}
