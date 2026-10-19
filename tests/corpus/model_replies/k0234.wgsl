// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t = 0u; t < m; t = t + 16u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let e = lidx + 256u * j;
            let it = e >> 6u;
            let cc = e & 63u;
            let item = t + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let c = col0 + cc;
                if (c < d) {
                    va = phi[item * d + c];
                }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    vb = phi[item * d + cfg.r_off + lr];
                }
            }
            wa[e & 1023u] = va;
            wb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ba = kk * 64u + lid.x * 4u;
            let bb = kk * 64u + lid.y * 4u;
            let a0 = wa[ba & 1023u];
            let a1 = wa[(ba + 1u) & 1023u];
            let a2 = wa[(ba + 2u) & 1023u];
            let a3 = wa[(ba + 3u) & 1023u];
            for (var i = 0u; i < 4u; i = i + 1u) {
                let b = wb[(bb + i) & 1023u];
                acc[(i * 4u) & 15u] = acc[(i * 4u) & 15u] + b * a0;
                acc[(i * 4u + 1u) & 15u] = acc[(i * 4u + 1u) & 15u] + b * a1;
                acc[(i * 4u + 2u) & 15u] = acc[(i * 4u + 2u) & 15u] + b * a2;
                acc[(i * 4u + 3u) & 15u] = acc[(i * 4u + 3u) & 15u] + b * a3;
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
