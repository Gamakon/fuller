// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 1024>;
var<workgroup> sB: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i = tid + k * 256u;
            let tt = i >> 6u;
            let j = i & 63u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrow = r0 + j;
                let grow = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && grow < d) {
                    va = phi[t * d + grow];
                }
                let col = c0 + j;
                if (col < d) {
                    vb = phi[t * d + col];
                }
            }
            sA[i] = va;
            sB[i] = vb;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let base = kk * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                a[q] = sA[base + ty + 16u * q];
                b[q] = sB[base + tx + 16u * q];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = c0 + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
