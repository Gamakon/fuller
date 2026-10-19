// grid 16 16 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 2048>;
var<workgroup> wb: array<f32, 2048>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    let col0 = wid.x * 128u;
    let row0 = wid.y * 128u;

    var acc: array<f32, 64>;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) {
        acc[q & 63u] = 0.0;
    }
    var av: array<f32, 8>;
    var bv: array<f32, 8>;

    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let e = lidx + 256u * q;
            let item = e >> 7u;
            let cc = e & 127u;
            let ti = t + item;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (ti < m) {
                let base = ti * d;
                let gc = col0 + cc;
                if (gc < d) {
                    va = phi[base + gc];
                }
                let lrw = row0 + cc;
                let grw = r_off + lrw;
                if (lrw < r_rows && grw < d) {
                    vb = phi[base + grw];
                }
            }
            wa[e & 2047u] = va;
            wb[e & 2047u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 128u;
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                av[j & 7u] = wa[(o + lid.x + 16u * j) & 2047u];
                bv[j & 7u] = wb[(o + lid.y + 16u * j) & 2047u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let bb = bv[i & 7u];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    let ix = (i * 8u + j) & 63u;
                    acc[ix] = acc[ix] + bb * av[j & 7u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < r_rows && (r_off + lr) < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
