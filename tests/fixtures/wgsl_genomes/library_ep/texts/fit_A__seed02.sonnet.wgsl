// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc00 = vec4<f32>(0.0);
    var acc01 = vec4<f32>(0.0);
    var acc02 = vec4<f32>(0.0);
    var acc03 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i = tid + k * 256u;
            let tt = i >> 6u;
            let cc = i & 63u;
            let t = t0 + tt;
            let col = colBase + cc;
            let lr = rowBase + cc;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                if (col < d) { vb = phi[t * d + col]; }
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sb[i] = vb;
            sa[i] = va;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 8u; tt = tt + 1u) {
            let ob = tt * 64u + lid.x * 4u;
            let oa = tt * 64u + lid.y * 4u;
            let b = vec4<f32>(sb[ob], sb[ob + 1u], sb[ob + 2u], sb[ob + 3u]);
            acc00 = acc00 + sa[oa] * b;
            acc01 = acc01 + sa[oa + 1u] * b;
            acc02 = acc02 + sa[oa + 2u] * b;
            acc03 = acc03 + sa[oa + 3u] * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows) {
            var a = acc00;
            if (i == 1u) { a = acc01; }
            if (i == 2u) { a = acc02; }
            if (i == 3u) { a = acc03; }
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + a.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + a.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + a.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + a.w; }
        }
    }
}
