// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = tid + 256u * q;
            let k = e >> 6u;
            let n = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let gc = colBase + n;
                if (gc < d) {
                    vb = phi[t * d + gc];
                }
                let lr = rowBase + n;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ro = k * 64u + ty * 4u;
            let co = k * 64u + tx * 4u;
            let r0 = sa[ro & 1023u];
            let r1 = sa[(ro + 1u) & 1023u];
            let r2 = sa[(ro + 2u) & 1023u];
            let r3 = sa[(ro + 3u) & 1023u];
            let b = vec4<f32>(sb[co & 1023u], sb[(co + 1u) & 1023u], sb[(co + 2u) & 1023u], sb[(co + 3u) & 1023u]);
            a0 = a0 + r0 * b;
            a1 = a1 + r1 * b;
            a2 = a2 + r2 * b;
            a3 = a3 + r3 * b;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let c = colBase + tx * 4u;
            let base = lr * d + c;
            if (c < d) { h[base] = h[base] + v.x; }
            if (c + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
