// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pacc: array<vec4<f32>, 16>;
    var px: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pacc[k] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        px[k] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let lidx = lid.y * 16u + lid.x;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 128u;
    let rlim = min(cfg.r_rows, select(0u, cfg.d - cfg.r_off, cfg.r_off < cfg.d));
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        let aitem = lidx >> 4u;
        let acv = colbase + (lidx & 15u) * 4u;
        let ati = t + aitem;
        let abase = ati * cfg.d + acv;
        let oka = ati < cfg.m;
        let ia0 = select(0u, abase, acv < cfg.d && oka);
        let ia1 = select(0u, abase + 1u, acv + 1u < cfg.d && oka);
        let ia2 = select(0u, abase + 2u, acv + 2u < cfg.d && oka);
        let ia3 = select(0u, abase + 3u, acv + 3u < cfg.d && oka);
        let va0 = select(0.0, phi[ia0], acv < cfg.d && oka);
        let va1 = select(0.0, phi[ia1], acv + 1u < cfg.d && oka);
        let va2 = select(0.0, phi[ia2], acv + 2u < cfg.d && oka);
        let va3 = select(0.0, phi[ia3], acv + 3u < cfg.d && oka);
        wg_va[lidx] = vec4<f32>(va0, va1, va2, va3);
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let bidx = lidx + 256u * j;
            let bitem = bidx >> 5u;
            let lr0 = rowbase + (bidx & 31u) * 4u;
            let bti = t + bitem;
            let okb = bti < cfg.m;
            let bbase = bti * cfg.d + cfg.r_off + lr0;
            let ib0 = select(0u, bbase, lr0 < rlim && okb);
            let ib1 = select(0u, bbase + 1u, lr0 + 1u < rlim && okb);
            let ib2 = select(0u, bbase + 2u, lr0 + 2u < rlim && okb);
            let ib3 = select(0u, bbase + 3u, lr0 + 3u < rlim && okb);
            let vb0 = select(0.0, phi[ib0], lr0 < rlim && okb);
            let vb1 = select(0.0, phi[ib1], lr0 + 1u < rlim && okb);
            let vb2 = select(0.0, phi[ib2], lr0 + 2u < rlim && okb);
            let vb3 = select(0.0, phi[ib3], lr0 + 3u < rlim && okb);
            wg_vb[bidx] = vec4<f32>(vb0, vb1, vb2, vb3);
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            px[k] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.x];
            let b0 = wg_vb[k * 32u + lid.y];
            let b1 = wg_vb[k * 32u + 16u + lid.y];
            px[0] = px[0] + vec4<f32>(b0.x, b0.x, b0.x, b0.x) * a;
            px[1] = px[1] + vec4<f32>(b0.y, b0.y, b0.y, b0.y) * a;
            px[2] = px[2] + vec4<f32>(b0.z, b0.z, b0.z, b0.z) * a;
            px[3] = px[3] + vec4<f32>(b0.w, b0.w, b0.w, b0.w) * a;
            px[4] = px[4] + vec4<f32>(b1.x, b1.x, b1.x, b1.x) * a;
            px[5] = px[5] + vec4<f32>(b1.y, b1.y, b1.y, b1.y) * a;
            px[6] = px[6] + vec4<f32>(b1.z, b1.z, b1.z, b1.z) * a;
            px[7] = px[7] + vec4<f32>(b1.w, b1.w, b1.w, b1.w) * a;
        }
        let sel = ((t >> 4u) & 1u) * 8u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pacc[sel + k] = pacc[sel + k] + px[k];
        }
        workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
        let lr = rowbase + (r >> 2u) * 64u + lid.y * 4u + (r & 3u);
        let gr = cfg.r_off + lr;
        let acc = pacc[r] + pacc[r + 8u];
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let col = colbase + lid.x * 4u + c;
            if (lr < cfg.r_rows && gr < cfg.d && col < cfg.d) {
                let idx = lr * cfg.d + col;
                h[idx] = h[idx] + acc[c];
            }
        }
    }
}
