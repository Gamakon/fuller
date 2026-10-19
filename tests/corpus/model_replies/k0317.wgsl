// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<vec4<f32>, 256>;
var<workgroup> sB: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let ldt = l / 16u;
    let ldj = (l % 16u) * 4u;
    let colBase = wg.x * 64u;
    let rowBase = wg.y * 64u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ldt;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = colBase + ldj + i;
                if (c < d) { vb[i] = phi[base + c]; }
                let lr = rowBase + ldj + i;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va[i] = phi[base + gr]; }
            }
        }
        sA[l] = va;
        sB[l] = vb;
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a = sA[kk * 16u + ty];
            let b = sB[kk * 16u + tx];
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let col0 = colBase + tx * 4u;
    let row0 = rowBase + ty * 4u;
    var accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i;
        if (lr < cfg.r_rows) {
            let v = accs[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
