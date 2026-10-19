// grid 16 16 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

fn load_tile(t0: u32, buf: u32, lidx: u32, wx: u32, wy: u32) {
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        let e = lidx + q * 256u;
        let it = e >> 7u;
        let cl = e & 127u;
        let item = t0 + it;
        var fb: f32 = 0.0;
        var fa: f32 = 0.0;
        if (item < cfg.m) {
            let cc = wx * 128u + cl;
            if (cc < cfg.d) {
                fb = phi[item * cfg.d + cc];
            }
            let rl = wy * 128u + cl;
            if (rl < cfg.r_rows && (cfg.r_off + rl) < cfg.d) {
                fa = phi[item * cfg.d + cfg.r_off + rl];
            }
        }
        let si = (buf * 1024u + e) & 2047u;
        wg_a[si] = fa;
        wg_b[si] = fb;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = vec4<f32>(0.0);
    }
    load_tile(0u, 0u, lidx, wid.x, wid.y);
    workgroupBarrier();
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        let buf = (t >> 3u) & 1u;
        load_tile(t + 8u, 1u - buf, lidx, wid.x, wid.y);
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let base = buf * 1024u + kk * 128u;
            let bi0 = base + lid.x * 4u;
            let bi1 = base + 64u + lid.x * 4u;
            let bv0 = vec4<f32>(wg_b[bi0 & 2047u], wg_b[(bi0 + 1u) & 2047u], wg_b[(bi0 + 2u) & 2047u], wg_b[(bi0 + 3u) & 2047u]);
            let bv1 = vec4<f32>(wg_b[bi1 & 2047u], wg_b[(bi1 + 1u) & 2047u], wg_b[(bi1 + 2u) & 2047u], wg_b[(bi1 + 3u) & 2047u]);
            for (var r: u32 = 0u; r < 8u; r = r + 1u) {
                let ai = base + (r >> 2u) * 64u + lid.y * 4u + (r & 3u);
                let av = wg_a[ai & 2047u];
                let a4 = vec4<f32>(av);
                acc[(r * 2u) & 15u] = fma(a4, bv0, acc[(r * 2u) & 15u]);
                acc[(r * 2u + 1u) & 15u] = fma(a4, bv1, acc[(r * 2u + 1u) & 15u]);
            }
        }
        workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
        let lr = wid.y * 128u + (r >> 2u) * 64u + lid.y * 4u + (r & 3u);
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var hh: u32 = 0u; hh < 2u; hh = hh + 1u) {
                let v = acc[(r * 2u + hh) & 15u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = wid.x * 128u + hh * 64u + lid.x * 4u + j;
                    if (c < cfg.d) {
                        let idx = lr * cfg.d + c;
                        h[idx] = h[idx] + v[j];
                    }
                }
            }
        }
    }
}
