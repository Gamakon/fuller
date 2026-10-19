// grid 16 16 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 64>;
    var ra: array<f32, 8>;
    var rb: array<f32, 8>;
    for (var q = 0u; q < 64u; q = q + 1u) { acc[q & 63u] = 0.0; }
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 128u;
    let rowbase = wid.y * 128u;

    // prologue: load item tile 0 into buffer 0
    for (var k = 0u; k < 4u; k = k + 1u) {
        let e = lidx + k * 256u;
        let it = e >> 7u;
        let cc = e & 127u;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (it < m) {
            let gc = colbase + cc;
            if (gc < d) { vb = phi[it * d + gc]; }
            let lr = rowbase + cc;
            if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { va = phi[it * d + cfg.r_off + lr]; }
        }
        wg_a[e & 2047u] = va;
        wg_b[e & 2047u] = vb;
    }
    workgroupBarrier();

    for (var t = 0u; t < m; t = t + 8u) {
        let cur = (t >> 3u) & 1u;
        let nxt = 1u - cur;
        // prefetch next tile into the other buffer
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = e >> 7u;
            let cc = e & 127u;
            let tt = t + 8u + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tt < m) {
                let gc = colbase + cc;
                if (gc < d) { vb = phi[tt * d + gc]; }
                let lr = rowbase + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { va = phi[tt * d + cfg.r_off + lr]; }
            }
            wg_a[(nxt * 1024u + e) & 2047u] = va;
            wg_b[(nxt * 1024u + e) & 2047u] = vb;
        }
        // compute from the current buffer
        let base = cur * 1024u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            for (var i = 0u; i < 8u; i = i + 1u) {
                ra[i & 7u] = wg_a[(base + j * 128u + i * 16u + lid.y) & 2047u];
                rb[i & 7u] = wg_b[(base + j * 128u + i * 16u + lid.x) & 2047u];
            }
            for (var i = 0u; i < 8u; i = i + 1u) {
                let a = ra[i & 7u];
                for (var q = 0u; q < 8u; q = q + 1u) {
                    let ix = (i * 8u + q) & 63u;
                    acc[ix] = fma(a, rb[q & 7u], acc[ix]);
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 8u; i = i + 1u) {
        let lr = rowbase + i * 16u + lid.y;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var q = 0u; q < 8u; q = q + 1u) {
                let c = colbase + q * 16u + lid.x;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 8u + q) & 63u];
                }
            }
        }
    }
}
