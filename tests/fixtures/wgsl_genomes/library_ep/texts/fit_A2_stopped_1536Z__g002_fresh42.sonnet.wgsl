// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 256>;
var<workgroup> wg_b: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    let d = cfg.d;
    let lk = tid >> 4u;
    let lc = (tid & 15u) * 4u;
    let colBase = wid.x * 64u + lc;
    let rowBase = wid.y * 64u + lc;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        let item = t + lk;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (item < cfg.m) {
            let base = item * d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + j;
                if (c < d) { a[j] = phi[base + c]; }
                let lr = rowBase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { b[j] = phi[base + gr]; }
            }
        }
        wg_a[tid] = a;
        wg_b[tid] = b;
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = wg_a[k * 16u + lid.x];
            let bv = wg_b[k * 16u + lid.y];
            p0 = p0 + bv.x * av;
            p1 = p1 + bv.y * av;
            p2 = p2 + bv.z * av;
            p3 = p3 + bv.w * av;
        }
        o0 = o0 + p0;
        o1 = o1 + p1;
        o2 = o2 + p2;
        o3 = o3 + p3;
        workgroupBarrier();
    }
    var accs = array<vec4<f32>, 4>(o0, o1, o2, o3);
    let c0 = wid.x * 64u + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            let v = accs[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = c0 + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
