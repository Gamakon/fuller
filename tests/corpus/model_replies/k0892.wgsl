// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q & 15u] = 0.0; }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    // stage 0 preload
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let idx = lidx + k * 256u;
        let it = idx >> 6u;
        let cc = idx & 63u;
        var fb: f32 = 0.0;
        var fa: f32 = 0.0;
        if (it < m) {
            if (col0 + cc < d) { fb = phi[it * d + col0 + cc]; }
            if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) { fa = phi[it * d + cfg.r_off + row0 + cc]; }
        }
        wg_a[idx & 2047u] = fa;
        wg_b[idx & 2047u] = fb;
    }
    workgroupBarrier();
    for (var t: u32 = 0u; t < m; t = t + 8u) {
        let cur = ((t >> 3u) & 1u) * 1024u;
        let nxt = 1024u - cur;
        // prefetch next tile into the other buffer
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = t + 8u + (idx >> 6u);
            let cc = idx & 63u;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (it < m) {
                if (col0 + cc < d) { fb = phi[it * d + col0 + cc]; }
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) { fa = phi[it * d + cfg.r_off + row0 + cc]; }
            }
            wg_a[(nxt + idx) & 2047u] = fa;
            wg_b[(nxt + idx) & 2047u] = fb;
        }
        // compute from current buffer
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let ba = cur + s * 64u + lid.y * 4u;
            let bb = cur + s * 64u + lid.x * 4u;
            let a0 = wg_a[ba & 2047u];
            let a1 = wg_a[(ba + 1u) & 2047u];
            let a2 = wg_a[(ba + 2u) & 2047u];
            let a3 = wg_a[(ba + 3u) & 2047u];
            let b0 = wg_b[bb & 2047u];
            let b1 = wg_b[(bb + 1u) & 2047u];
            let b2 = wg_b[(bb + 2u) & 2047u];
            let b3 = wg_b[(bb + 3u) & 2047u];
            acc[0u] = fma(a0, b0, acc[0u]);
            acc[1u] = fma(a0, b1, acc[1u]);
            acc[2u] = fma(a0, b2, acc[2u]);
            acc[3u] = fma(a0, b3, acc[3u]);
            acc[4u] = fma(a1, b0, acc[4u]);
            acc[5u] = fma(a1, b1, acc[5u]);
            acc[6u] = fma(a1, b2, acc[6u]);
            acc[7u] = fma(a1, b3, acc[7u]);
            acc[8u] = fma(a2, b0, acc[8u]);
            acc[9u] = fma(a2, b1, acc[9u]);
            acc[10u] = fma(a2, b2, acc[10u]);
            acc[11u] = fma(a2, b3, acc[11u]);
            acc[12u] = fma(a3, b0, acc[12u]);
            acc[13u] = fma(a3, b1, acc[13u]);
            acc[14u] = fma(a3, b2, acc[14u]);
            acc[15u] = fma(a3, b3, acc[15u]);
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
