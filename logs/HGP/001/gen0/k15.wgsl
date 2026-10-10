// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid3.x;
    let ly = lid3.y;
    let lid = ly * 16u + lx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var a0: array<f32, 16>;
    var a1: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        a0[i] = 0.0;
        a1[i] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lid + k * 256u;
            let tt = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrow = row0 + cc;
                let grow = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && grow < d) {
                    va = phi[t * d + grow];
                }
                let c = col0 + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            As[idx] = va;
            Bs[idx] = vb;
        }
        workgroupBarrier();
        for (var tt = 0u; tt < 16u; tt = tt + 2u) {
            let o0 = tt * 64u;
            let o1 = o0 + 64u;
            var ra0: array<f32, 4>;
            var rb0: array<f32, 4>;
            var ra1: array<f32, 4>;
            var rb1: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                ra0[i] = As[o0 + ly + 16u * i];
                rb0[i] = Bs[o0 + lx + 16u * i];
                ra1[i] = As[o1 + ly + 16u * i];
                rb1[i] = Bs[o1 + lx + 16u * i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    a0[i * 4u + j] = a0[i * 4u + j] + ra0[i] * rb0[j];
                    a1[i * 4u + j] = a1[i * 4u + j] + ra1[i] * rb1[j];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ly + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (a0[i * 4u + j] + a1[i * 4u + j]);
                }
            }
        }
    }
}
