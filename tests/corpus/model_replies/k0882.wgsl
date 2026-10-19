// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

fn store_row(lr: u32, c0: u32, v: vec4<f32>) {
    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
        let base = lr * cfg.d + c0;
        if (c0 < cfg.d) { h[base] = h[base] + v.x; }
        if (c0 + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
        if (c0 + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
        if (c0 + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let tx = lid.x;
    let ty = lid.y;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let d = cfg.d;
    let m = cfg.m;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    // initial tile into buffer 0
    for (var k = 0u; k < 4u; k = k + 1u) {
        let e = lidx + k * 256u;
        let it = e >> 6u;
        let cc = e & 63u;
        var fb = 0.0;
        var fa = 0.0;
        if (it < m) {
            if (colbase + cc < d) { fb = phi[it * d + colbase + cc]; }
            let lr = rowbase + cc;
            if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { fa = phi[it * d + cfg.r_off + lr]; }
        }
        wg_a[e & 2047u] = fa;
        wg_b[e & 2047u] = fb;
    }
    workgroupBarrier();

    for (var t = 0u; t < m; t = t + 16u) {
        let cur = (t >> 4u) & 1u;
        let nxt = cur ^ 1u;
        let tn = t + 16u;

        var ra0 = 0.0; var ra1 = 0.0; var ra2 = 0.0; var ra3 = 0.0;
        var rb0 = 0.0; var rb1 = 0.0; var rb2 = 0.0; var rb3 = 0.0;
        // prefetch next tile into registers
        {
            let e0 = lidx;
            let it0 = tn + (e0 >> 6u);
            let cc0 = e0 & 63u;
            if (it0 < m) {
                if (colbase + cc0 < d) { rb0 = phi[it0 * d + colbase + cc0]; }
                let lr = rowbase + cc0;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { ra0 = phi[it0 * d + cfg.r_off + lr]; }
            }
            let e1 = lidx + 256u;
            let it1 = tn + (e1 >> 6u);
            let cc1 = e1 & 63u;
            if (it1 < m) {
                if (colbase + cc1 < d) { rb1 = phi[it1 * d + colbase + cc1]; }
                let lr = rowbase + cc1;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { ra1 = phi[it1 * d + cfg.r_off + lr]; }
            }
            let e2 = lidx + 512u;
            let it2 = tn + (e2 >> 6u);
            let cc2 = e2 & 63u;
            if (it2 < m) {
                if (colbase + cc2 < d) { rb2 = phi[it2 * d + colbase + cc2]; }
                let lr = rowbase + cc2;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { ra2 = phi[it2 * d + cfg.r_off + lr]; }
            }
            let e3 = lidx + 768u;
            let it3 = tn + (e3 >> 6u);
            let cc3 = e3 & 63u;
            if (it3 < m) {
                if (colbase + cc3 < d) { rb3 = phi[it3 * d + colbase + cc3]; }
                let lr = rowbase + cc3;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { ra3 = phi[it3 * d + cfg.r_off + lr]; }
            }
        }

        // compute on current buffer
        for (var it = 0u; it < 16u; it = it + 1u) {
            let base = cur * 1024u + it * 64u;
            let ai = base + ty * 4u;
            let bi = base + tx * 4u;
            let a = vec4<f32>(wg_a[ai & 2047u], wg_a[(ai + 1u) & 2047u], wg_a[(ai + 2u) & 2047u], wg_a[(ai + 3u) & 2047u]);
            let b = vec4<f32>(wg_b[bi & 2047u], wg_b[(bi + 1u) & 2047u], wg_b[(bi + 2u) & 2047u], wg_b[(bi + 3u) & 2047u]);
            acc0 = fma(vec4<f32>(a.x), b, acc0);
            acc1 = fma(vec4<f32>(a.y), b, acc1);
            acc2 = fma(vec4<f32>(a.z), b, acc2);
            acc3 = fma(vec4<f32>(a.w), b, acc3);
        }

        // store prefetched tile into other buffer
        let o = nxt * 1024u;
        wg_a[(o + lidx) & 2047u] = ra0;
        wg_b[(o + lidx) & 2047u] = rb0;
        wg_a[(o + lidx + 256u) & 2047u] = ra1;
        wg_b[(o + lidx + 256u) & 2047u] = rb1;
        wg_a[(o + lidx + 512u) & 2047u] = ra2;
        wg_b[(o + lidx + 512u) & 2047u] = rb2;
        wg_a[(o + lidx + 768u) & 2047u] = ra3;
        wg_b[(o + lidx + 768u) & 2047u] = rb3;
        workgroupBarrier();
    }

    let r0 = rowbase + ty * 4u;
    let c0 = colbase + tx * 4u;
    store_row(r0, c0, acc0);
    store_row(r0 + 1u, c0, acc1);
    store_row(r0 + 2u, c0, acc2);
    store_row(r0 + 3u, c0, acc3);
}
