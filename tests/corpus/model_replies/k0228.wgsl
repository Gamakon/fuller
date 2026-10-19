// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;

fn ld_col(item: u32, c: u32) -> f32 {
    if (item < cfg.m && c < cfg.d) {
        return phi[item * cfg.d + c];
    }
    return 0.0;
}

fn ld_row(item: u32, r: u32) -> f32 {
    if (item < cfg.m && r < cfg.r_rows && cfg.r_off + r < cfg.d) {
        return phi[item * cfg.d + cfg.r_off + r];
    }
    return 0.0;
}

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);

    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let ch = (idx & 15u) * 4u;
            let c = col0 + ch;
            let r = row0 + ch;
            wg_va[idx & 511u] = vec4<f32>(ld_col(item, c), ld_col(item, c + 1u), ld_col(item, c + 2u), ld_col(item, c + 3u));
            wg_vb[idx & 511u] = vec4<f32>(ld_row(item, r), ld_row(item, r + 1u), ld_row(item, r + 2u), ld_row(item, r + 3u));
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b0 = wg_vb[(k * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(k * 16u + lid.y * 2u + 1u) & 511u];
            a0 = a0 + vec4<f32>(b0.x) * a;
            a1 = a1 + vec4<f32>(b0.y) * a;
            a2 = a2 + vec4<f32>(b0.z) * a;
            a3 = a3 + vec4<f32>(b0.w) * a;
            a4 = a4 + vec4<f32>(b1.x) * a;
            a5 = a5 + vec4<f32>(b1.y) * a;
            a6 = a6 + vec4<f32>(b1.z) * a;
            a7 = a7 + vec4<f32>(b1.w) * a;
        }
        workgroupBarrier();
    }

    var pv: array<vec4<f32>, 8>;
    pv[0] = a0; pv[1] = a1; pv[2] = a2; pv[3] = a3;
    pv[4] = a4; pv[5] = a5; pv[6] = a6; pv[7] = a7;

    let cb = col0 + lid.x * 4u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = row0 + lid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let v = pv[k & 7u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + v[j & 3u];
                }
            }
        }
    }
}
