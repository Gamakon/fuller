// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 8>;
    var pw: array<vec4<f32>, 8>;
    var px: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
        pw[k] = vec4<f32>(0.0);
    }
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 128u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        // load A tile: 16 items x 64 columns
        {
            let item = lidx >> 4u;
            let cv = (lidx & 15u) * 4u;
            let ti = t + item;
            var w = vec4<f32>(0.0);
            if (ti < cfg.m) {
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    let c = colbase + cv + e;
                    if (c < cfg.d) {
                        w[e] = phi[ti * cfg.d + c];
                    }
                }
            }
            wg_va[lidx] = w;
        }
        // load B tile: 16 items x 128 rows
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let idx = lidx + 256u * j;
            let item = idx >> 5u;
            let rv = (idx & 31u) * 4u;
            let ti = t + item;
            var w = vec4<f32>(0.0);
            if (ti < cfg.m) {
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    let lr = rowbase + rv + e;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        w[e] = phi[ti * cfg.d + cfg.r_off + lr];
                    }
                }
            }
            wg_vb[idx] = w;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            px[k] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.x];
            let b0 = wg_vb[k * 32u + lid.y];
            let b1 = wg_vb[k * 32u + 16u + lid.y];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                px[j] = px[j] + vec4<f32>(b0[j]) * a;
                px[j + 4u] = px[j + 4u] + vec4<f32>(b1[j]) * a;
            }
        }
        if (((t >> 4u) & 1u) == 0u) {
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                pv[k] = pv[k] + px[k];
            }
        } else {
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                pw[k] = pw[k] + px[k];
            }
        }
        workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
        let lr = rowbase + (r >> 2u) * 64u + lid.y * 4u + (r & 3u);
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            let acc = pv[r] + pw[r];
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                let col = colbase + lid.x * 4u + c;
                if (col < cfg.d) {
                    let idx = lr * cfg.d + col;
                    h[idx] = h[idx] + acc[c];
                }
            }
        }
    }
}
