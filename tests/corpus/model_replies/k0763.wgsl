// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pacc: array<vec4<f32>, 16>;
    var px: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pacc[k] = vec4<f32>(0.0);
    }
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 128u;
    let rlim = min(cfg.r_rows, select(0u, cfg.d - cfg.r_off, cfg.r_off < cfg.d));
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        let aitem = lidx >> 4u;
        let acv = colbase + (lidx & 15u) * 4u;
        let ati = t + aitem;
        var wa = vec4<f32>(0.0);
        if (ati < cfg.m) {
            let abase = ati * cfg.d + acv;
            if (acv < cfg.d) { wa.x = phi[abase]; }
            if (acv + 1u < cfg.d) { wa.y = phi[abase + 1u]; }
            if (acv + 2u < cfg.d) { wa.z = phi[abase + 2u]; }
            if (acv + 3u < cfg.d) { wa.w = phi[abase + 3u]; }
        }
        wg_va[lidx] = wa;
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let bidx = lidx + 256u * j;
            let bitem = bidx >> 5u;
            let lr0 = rowbase + (bidx & 31u) * 4u;
            let bti = t + bitem;
            var wb = vec4<f32>(0.0);
            if (bti < cfg.m) {
                let bbase = bti * cfg.d + cfg.r_off + lr0;
                if (lr0 < rlim) { wb.x = phi[bbase]; }
                if (lr0 + 1u < rlim) { wb.y = phi[bbase + 1u]; }
                if (lr0 + 2u < rlim) { wb.z = phi[bbase + 2u]; }
                if (lr0 + 3u < rlim) { wb.w = phi[bbase + 3u]; }
            }
            wg_vb[bidx] = wb;
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
        let sel = ((t >> 4u) & 1u) * 8u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pacc[sel + k] = pacc[sel + k] + px[k];
        }
        workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
        let lr = rowbase + (r >> 2u) * 64u + lid.y * 4u + (r & 3u);
        if (lr < rlim) {
            let acc = pacc[r] + pacc[r + 8u];
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
