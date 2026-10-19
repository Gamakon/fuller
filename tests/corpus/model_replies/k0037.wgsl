// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lin = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let rbase = wid.y * 64u;
    let cbase = wid.x * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lin + 256u * q;
            let k = e >> 6u;
            let x = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = rbase + x;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = cbase + x;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = sa[(o + ty + 16u * i) & 1023u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b = sb[(o + tx + 16u * j) & 1023u];
                    let ai = (i * 4u + j) & 15u;
                    acc[ai] = acc[ai] + a * b;
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cbase + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
