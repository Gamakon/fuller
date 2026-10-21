// grid 8 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 16>;
    var s: array<vec4<f32>, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
        s[i] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let q = lidx + 64u * j;
            let item = t + (q >> 4u);
            let cv = (q & 15u) * 4u;
            let col = wid.x * 64u + cv;
            let lrow = wid.y * 64u + cv;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                if (col < cfg.d) { a.x = phi[base + col]; }
                if (col + 1u < cfg.d) { a.y = phi[base + col + 1u]; }
                if (col + 2u < cfg.d) { a.z = phi[base + col + 2u]; }
                if (col + 3u < cfg.d) { a.w = phi[base + col + 3u]; }
                let g = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && g < cfg.d) { b.x = phi[base + g]; }
                if (lrow + 1u < cfg.r_rows && g + 1u < cfg.d) { b.y = phi[base + g + 1u]; }
                if (lrow + 2u < cfg.r_rows && g + 2u < cfg.d) { b.z = phi[base + g + 2u]; }
                if (lrow + 3u < cfg.r_rows && g + 3u < cfg.d) { b.w = phi[base + g + 3u]; }
            }
            wg_va[q] = a;
            wg_vb[q] = b;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            s[i] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_va[k * 16u + lid.x * 2u];
            let a1 = wg_va[k * 16u + lid.x * 2u + 1u];
            let b0 = wg_vb[k * 16u + lid.y * 2u];
            let b1 = wg_vb[k * 16u + lid.y * 2u + 1u];
            s[0] = s[0] + vec4<f32>(b0.x) * a0;
            s[1] = s[1] + vec4<f32>(b0.x) * a1;
            s[2] = s[2] + vec4<f32>(b0.y) * a0;
            s[3] = s[3] + vec4<f32>(b0.y) * a1;
            s[4] = s[4] + vec4<f32>(b0.z) * a0;
            s[5] = s[5] + vec4<f32>(b0.z) * a1;
            s[6] = s[6] + vec4<f32>(b0.w) * a0;
            s[7] = s[7] + vec4<f32>(b0.w) * a1;
            s[8] = s[8] + vec4<f32>(b1.x) * a0;
            s[9] = s[9] + vec4<f32>(b1.x) * a1;
            s[10] = s[10] + vec4<f32>(b1.y) * a0;
            s[11] = s[11] + vec4<f32>(b1.y) * a1;
            s[12] = s[12] + vec4<f32>(b1.z) * a0;
            s[13] = s[13] + vec4<f32>(b1.z) * a1;
            s[14] = s[14] + vec4<f32>(b1.w) * a0;
            s[15] = s[15] + vec4<f32>(b1.w) * a1;
        }
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc[i] = acc[i] + s[i];
        }
        workgroupBarrier();
    }
    for (var ri: u32 = 0u; ri < 8u; ri = ri + 1u) {
        let lr = wid.y * 64u + lid.y * 8u + ri;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var c: u32 = 0u; c < 8u; c = c + 1u) {
                let col = wid.x * 64u + lid.x * 8u + c;
                if (col < cfg.d) {
                    let idx = lr * cfg.d + col;
                    h[idx] = h[idx] + acc[ri * 2u + (c >> 2u)][c & 3u];
                }
            }
        }
    }
}
