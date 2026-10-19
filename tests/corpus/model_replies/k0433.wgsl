// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    let r0 = gid.y * 8u;
    let c0 = gid.x * 4u;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) {
        pa[q] = 0.0;
        pb[q] = 0.0;
        pc[q] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let row = r0 + j;
            let gr = cfg.r_off + row;
            var rv: f32 = 0.0;
            if (row < cfg.r_rows && gr < cfg.d) {
                rv = phi[base + gr];
            }
            pb[j] = rv;
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let c = c0 + i;
            var cv: f32 = 0.0;
            if (c < cfg.d) {
                cv = phi[base + c];
            }
            pc[i] = cv;
        }
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                pa[j * 4u + i] = pa[j * 4u + i] + pb[j] * pc[i];
            }
        }
    }
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let row = r0 + j;
            let c = c0 + i;
            if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d && c < cfg.d) {
                let idx = row * cfg.d + c;
                h[idx] = h[idx] + pa[j * 4u + i];
            }
        }
    }
}
