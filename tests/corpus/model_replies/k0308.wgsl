// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sr: array<f32, 1024>;
var<workgroup> sc: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let d = cfg.d;
    let m = cfg.m;

    var acc: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k = 0u; k < 4u; k = k + 1u) {
            let i = lin + k * 256u;
            let tt = i >> 6u;
            let cc = i & 63u;
            let t = t0 + tt;
            var vr: f32 = 0.0;
            var vc: f32 = 0.0;
            if (t < m) {
                let lr = rowBase + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vr = phi[t * d + gr];
                }
                let c = colBase + cc;
                if (c < d) {
                    vc = phi[t * d + c];
                }
            }
            sr[i] = vr;
            sc[i] = vc;
        }
        workgroupBarrier();
        for (var tt = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                a[j] = sr[o + ty + j * 16u];
                b[j] = sc[o + tx + j * 16u];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                for (var i = 0u; i < 4u; i = i + 1u) {
                    acc[j * 4u + i] = acc[j * 4u + i] + a[j] * b[i];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + ty + j * 16u;
        if (lr < cfg.r_rows) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = colBase + tx + i * 16u;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[j * 4u + i];
                }
            }
        }
    }
}
