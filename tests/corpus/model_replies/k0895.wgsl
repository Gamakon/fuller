// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    // prologue: load items 0..7 into half 0
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let idx = lidx + k * 256u;
        let it = idx >> 6u;
        let cc = idx & 63u;
        var f0: f32 = 0.0;
        var f1: f32 = 0.0;
        if (it < m) {
            if (colBase + cc < d) {
                f0 = phi[it * d + colBase + cc];
            }
            if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < d) {
                f1 = phi[it * d + cfg.r_off + rowBase + cc];
            }
        }
        wg_a[idx & 2047u] = f1;
        wg_b[idx & 2047u] = f0;
    }
    workgroupBarrier();
    for (var t: u32 = 0u; t < m; t = t + 8u) {
        let cur = 1024u * ((t >> 3u) & 1u);
        let nxt = 1024u * (((t >> 3u) + 1u) & 1u);
        // prefetch next item tile into the other half
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = t + 8u + (idx >> 6u);
            let cc = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (it < m) {
                if (colBase + cc < d) {
                    f0 = phi[it * d + colBase + cc];
                }
                if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < d) {
                    f1 = phi[it * d + cfg.r_off + rowBase + cc];
                }
            }
            wg_a[(idx + nxt) & 2047u] = f1;
            wg_b[(idx + nxt) & 2047u] = f0;
        }
        // compute on current half: load b vectors once per item, reuse across rows
        let ao = cur + lid.y * 4u;
        let bo = cur + lid.x * 4u;
        for (var k: u32 = 0u; k < 8u; k = k + 2u) {
            let o0 = k * 64u;
            let o1 = o0 + 64u;
            let b00 = wg_b[(bo + o0) & 2047u];
            let b01 = wg_b[(bo + o0 + 1u) & 2047u];
            let b02 = wg_b[(bo + o0 + 2u) & 2047u];
            let b03 = wg_b[(bo + o0 + 3u) & 2047u];
            let b10 = wg_b[(bo + o1) & 2047u];
            let b11 = wg_b[(bo + o1 + 1u) & 2047u];
            let b12 = wg_b[(bo + o1 + 2u) & 2047u];
            let b13 = wg_b[(bo + o1 + 3u) & 2047u];
            for (var r: u32 = 0u; r < 4u; r = r + 1u) {
                let a0 = wg_a[(ao + o0 + r) & 2047u];
                let a1 = wg_a[(ao + o1 + r) & 2047u];
                let q = r * 4u;
                pa[(q) & 15u] = fma(a0, b00, pa[(q) & 15u]);
                pa[(q + 1u) & 15u] = fma(a0, b01, pa[(q + 1u) & 15u]);
                pa[(q + 2u) & 15u] = fma(a0, b02, pa[(q + 2u) & 15u]);
                pa[(q + 3u) & 15u] = fma(a0, b03, pa[(q + 3u) & 15u]);
                pb[(q) & 15u] = fma(a1, b10, pb[(q) & 15u]);
                pb[(q + 1u) & 15u] = fma(a1, b11, pb[(q + 1u) & 15u]);
                pb[(q + 2u) & 15u] = fma(a1, b12, pb[(q + 2u) & 15u]);
                pb[(q + 3u) & 15u] = fma(a1, b13, pb[(q + 3u) & 15u]);
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = rowBase + lid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = colBase + lid.x * 4u + k2;
                if (col < d) {
                    let q = (k * 4u + k2) & 15u;
                    h[row * d + col] = h[row * d + col] + (pa[q] + pb[q]);
                }
            }
        }
    }
}
