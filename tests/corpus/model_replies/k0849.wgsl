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
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = l + 256u * e;
            let k = idx / 64u;
            let j = idx % 64u;
            let t = t0 + k;
            var rv = 0.0;
            var cv = 0.0;
            if (t < m) {
                let lrow = row0 + j;
                let grow = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && grow < d) {
                    rv = phi[t * d + grow];
                }
                let c = col0 + j;
                if (c < d) {
                    cv = phi[t * d + c];
                }
            }
            sr[idx] = rv;
            sc[idx] = cv;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            var rr: array<f32, 4>;
            var cc: array<f32, 4>;
            for (var a = 0u; a < 4u; a = a + 1u) {
                rr[a] = sr[k * 64u + ty + 16u * a];
                cc[a] = sc[k * 64u + tx + 16u * a];
            }
            for (var a = 0u; a < 4u; a = a + 1u) {
                for (var b = 0u; b < 4u; b = b + 1u) {
                    acc[a * 4u + b] = acc[a * 4u + b] + rr[a] * cc[b];
                }
            }
        }
        workgroupBarrier();
    }

    for (var a = 0u; a < 4u; a = a + 1u) {
        let lr = row0 + ty + 16u * a;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var b = 0u; b < 4u; b = b + 1u) {
                let c = col0 + tx + 16u * b;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[a * 4u + b];
                }
            }
        }
    }
}