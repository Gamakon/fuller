// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var tot0 = vec4<f32>(0.0);
    var tot1 = vec4<f32>(0.0);
    var tot2 = vec4<f32>(0.0);
    var tot3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        // load: 4 vec4-groups... each thread loads 4 scalars per tile via (item, col) mapping
        // layout: slot s = item*16 + (col>>2), component col&3
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + 256u * q;
            let item = e >> 6u;
            let cc = e & 63u;
            let ti = t + item;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (ti < m) {
                let gc = col0 + cc;
                if (gc < d) {
                    vb = phi[ti * d + gc];
                }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    va = phi[ti * d + cfg.r_off + lr];
                }
            }
            let slot = item * 16u + (cc >> 2u);
            let comp = cc & 3u;
            // component-wise write; each (slot, comp) is unique per thread
            var cur_b = sb[slot];
            cur_b[comp] = vb;
            sb[slot] = cur_b;
            var cur_a = sa[slot];
            cur_a[comp] = va;
            sa[slot] = cur_a;
        }
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sa[k * 16u + lid.y];
            let b = sb[k * 16u + lid.x];
            p0 = p0 + vec4<f32>(a.x) * b;
            p1 = p1 + vec4<f32>(a.y) * b;
            p2 = p2 + vec4<f32>(a.z) * b;
            p3 = p3 + vec4<f32>(a.w) * b;
        }
        tot0 = tot0 + p0;
        tot1 = tot1 + p1;
        tot2 = tot2 + p2;
        tot3 = tot3 + p3;
        workgroupBarrier();
    }
    let cbase = col0 + lid.x * 4u;
    let rbase = row0 + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = tot0;
            if (i == 1u) { v = tot1; }
            if (i == 2u) { v = tot2; }
            if (i == 3u) { v = tot3; }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cbase + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
