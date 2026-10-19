// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s0: vec4<f32> = vec4<f32>(0.0);
    var s1: vec4<f32> = vec4<f32>(0.0);
    var s2: vec4<f32> = vec4<f32>(0.0);
    var s3: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    let ld_item = lidx >> 4u;
    let ld_c = (lidx & 15u) * 4u;
    let colbase = wid.x * 64u + ld_c;
    let rowloc = wid.y * 64u + ld_c;
    let rowg = cfg.r_off + rowloc;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        let it = t + ld_item;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (it < cfg.m) {
            let base = it * d;
            if (colbase < d) { a.x = phi[base + colbase]; }
            if (colbase + 1u < d) { a.y = phi[base + colbase + 1u]; }
            if (colbase + 2u < d) { a.z = phi[base + colbase + 2u]; }
            if (colbase + 3u < d) { a.w = phi[base + colbase + 3u]; }
            if (rowloc < cfg.r_rows && rowg < d) { b.x = phi[base + rowg]; }
            if (rowloc + 1u < cfg.r_rows && rowg + 1u < d) { b.y = phi[base + rowg + 1u]; }
            if (rowloc + 2u < cfg.r_rows && rowg + 2u < d) { b.z = phi[base + rowg + 2u]; }
            if (rowloc + 3u < cfg.r_rows && rowg + 3u < d) { b.w = phi[base + rowg + 3u]; }
        }
        wg_va[lidx] = a;
        wg_vb[lidx] = b;
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            p0 = p0 + vec4<f32>(vb.x) * va;
            p1 = p1 + vec4<f32>(vb.y) * va;
            p2 = p2 + vec4<f32>(vb.z) * va;
            p3 = p3 + vec4<f32>(vb.w) * va;
        }
        s0 = s0 + p0;
        s1 = s1 + p1;
        s2 = s2 + p2;
        s3 = s3 + p3;
        workgroupBarrier();
    }
    let c0 = wid.x * 64u + lid.x * 4u;
    let r0 = wid.y * 64u + lid.y * 4u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = r0 + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var s = s0;
            if (j == 1u) { s = s1; }
            if (j == 2u) { s = s2; }
            if (j == 3u) { s = s3; }
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let c = c0 + q;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + s[q];
                }
            }
        }
    }
}
