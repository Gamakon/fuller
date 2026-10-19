// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    let ntiles = (cfg.m + 31u) / 32u;
    for (var tt: u32 = 0u; tt < ntiles; tt = tt + 1u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let idx = lidx + k * 128u;
            let ti = idx / 64u;
            let cc = idx % 64u;
            let t = tt * 32u + ti;
            let col = wid.x * 64u + cc;
            var va: f32 = 0.0;
            if (t < cfg.m && col < cfg.d) { va = phi[t * cfg.d + col]; }
            wg_a[idx] = va;
            let lr = wid.y * 64u + cc;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            if (t < cfg.m && lr < cfg.r_rows && gr < cfg.d) { vb = phi[t * cfg.d + gr]; }
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let ab = k * 64u + lid.x * 4u;
            let a = vec4<f32>(wg_a[ab], wg_a[ab + 1u], wg_a[ab + 2u], wg_a[ab + 3u]);
            let bb = k * 64u + lid.y * 8u;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                pv[i] = pv[i] + wg_b[bb + i] * a;
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = gid.y * 8u + i;
        let gr = cfg.r_off + lr;
        let acc = pv[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = gid.x * 4u + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[j];
            }
        }
    }
}
