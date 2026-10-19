// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let litem = tid >> 4u;
    let lq = tid & 15u;
    let lc = col0 + lq * 4u;
    let lrw = row0 + lq * 4u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + litem;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = lc + j;
                if (c < d) { vb[j] = phi[base + c]; }
                let lr = lrw + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va[j] = phi[base + gr]; }
            }
        }
        sa[tid & 255u] = va;
        sb[tid & 255u] = vb;
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let a = sa[(i * 16u + ty) & 255u];
            let b = sb[(i * 16u + tx) & 255u];
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let c = col0 + tx * 4u;
    let r = row0 + ty * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r + i;
        if (lr < cfg.r_rows) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let o = lr * d + c;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                if (c + j < d) { h[o + j] = h[o + j] + v[j]; }
            }
        }
    }
}
