// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let item = idx >> 6u;
            let col = idx & 63u;
            let t = t0 + item;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let c = cb + col;
                if (c < d) { vb = phi[t * d + c]; }
                let lr = rb + col;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sb[idx] = vb;
            sa[idx] = va;
        }
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ob = k * 64u + lid.x * 4u;
            let oa = k * 64u + lid.y * 4u;
            let b = vec4<f32>(sb[ob], sb[ob + 1u], sb[ob + 2u], sb[ob + 3u]);
            p0 = p0 + sa[oa] * b;
            p1 = p1 + sa[oa + 1u] * b;
            p2 = p2 + sa[oa + 2u] * b;
            p3 = p3 + sa[oa + 3u] * b;
        }
        a0 = a0 + p0;
        a1 = a1 + p1;
        a2 = a2 + p2;
        a3 = a3 + p3;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rb + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var row = a0;
            if (i == 1u) { row = a1; }
            if (i == 2u) { row = a2; }
            if (i == 3u) { row = a3; }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + row[j];
                }
            }
        }
    }
}
