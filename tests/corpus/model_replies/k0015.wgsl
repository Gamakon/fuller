// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc0: array<f32, 16>;
    var acc1: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc0[q & 15u] = 0.0;
        acc1[q & 15u] = 0.0;
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();

        for (var k = 0u; k < 16u; k = k + 2u) {
            let k1 = k + 1u;
            var a0: array<f32, 4>;
            var b0: array<f32, 4>;
            var a1: array<f32, 4>;
            var b1: array<f32, 4>;
            for (var q = 0u; q < 4u; q = q + 1u) {
                a0[q & 3u] = sa[(k * 64u + ty + 16u * q) & 1023u];
                b0[q & 3u] = sb[(k * 64u + tx + 16u * q) & 1023u];
                a1[q & 3u] = sa[(k1 * 64u + ty + 16u * q) & 1023u];
                b1[q & 3u] = sb[(k1 * 64u + tx + 16u * q) & 1023u];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let p = (j * 4u + i) & 15u;
                    acc0[p] = acc0[p] + a0[j & 3u] * b0[i & 3u];
                    acc1[p] = acc1[p] + a1[j & 3u] * b1[i & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + ty + 16u * j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = col0 + tx + 16u * i;
                if (c < d) {
                    let p = (j * 4u + i) & 15u;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (acc0[p] + acc1[p]);
                }
            }
        }
    }
}
