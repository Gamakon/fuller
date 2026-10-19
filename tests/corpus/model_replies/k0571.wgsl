// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pe: array<vec4<f32>, 4>;
    var po: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        pe[i] = vec4<f32>(0.0);
        po[i] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 6u);
            let col = idx & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let gc = wid.x * 64u + col;
                if (gc < d) {
                    va = phi[item * d + gc];
                }
                let lr = wid.y * 64u + col;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    vb = phi[item * d + cfg.r_off + lr];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        var p0: vec4<f32> = vec4<f32>(0.0);
        var p1: vec4<f32> = vec4<f32>(0.0);
        var p2: vec4<f32> = vec4<f32>(0.0);
        var p3: vec4<f32> = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let base = kk * 64u;
            let av = vec4<f32>(wg_a[base + lid.x], wg_a[base + lid.x + 16u], wg_a[base + lid.x + 32u], wg_a[base + lid.x + 48u]);
            let b0 = wg_b[base + lid.y];
            let b1 = wg_b[base + lid.y + 16u];
            let b2 = wg_b[base + lid.y + 32u];
            let b3 = wg_b[base + lid.y + 48u];
            p0 = p0 + vec4<f32>(b0) * av;
            p1 = p1 + vec4<f32>(b1) * av;
            p2 = p2 + vec4<f32>(b2) * av;
            p3 = p3 + vec4<f32>(b3) * av;
        }
        let wo = f32((t >> 5u) & 1u);
        let we = 1.0 - wo;
        pe[0] = pe[0] + p0 * we;
        pe[1] = pe[1] + p1 * we;
        pe[2] = pe[2] + p2 * we;
        pe[3] = pe[3] + p3 * we;
        po[0] = po[0] + p0 * wo;
        po[1] = po[1] + p1 * wo;
        po[2] = po[2] + p2 * wo;
        po[3] = po[3] + p3 * wo;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            let s = pe[i] + po[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 64u + lid.x + 16u * j;
                if (col < d) {
                    let idx = lr * d + col;
                    h[idx] = h[idx] + s[j];
                }
            }
        }
    }
}
