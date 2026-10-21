// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let ld_item = li >> 4u;
    let ld_c4 = (li & 15u) * 4u;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        let item = t + ld_item;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (item < m) {
            let base = item * d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colbase + ld_c4 + j;
                if (c < d) {
                    a[j] = phi[base + c];
                }
                let lr = rowbase + ld_c4 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    b[j] = phi[base + gr];
                }
            }
        }
        sa[li] = a;
        sb[li] = b;
        workgroupBarrier();
        var t0 = vec4<f32>(0.0);
        var t1 = vec4<f32>(0.0);
        var t2 = vec4<f32>(0.0);
        var t3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = sa[k * 16u + lid.x];
            let bv = sb[k * 16u + lid.y];
            t0 = t0 + bv.x * av;
            t1 = t1 + bv.y * av;
            t2 = t2 + bv.z * av;
            t3 = t3 + bv.w * av;
        }
        acc0 = acc0 + t0;
        acc1 = acc1 + t1;
        acc2 = acc2 + t2;
        acc3 = acc3 + t3;
        workgroupBarrier();
    }
    var res: array<vec4<f32>, 4>;
    res[0] = acc0;
    res[1] = acc1;
    res[2] = acc2;
    res[3] = acc3;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colbase + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + res[i][j];
                }
            }
        }
    }
}
