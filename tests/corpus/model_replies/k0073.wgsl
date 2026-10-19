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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let it = e >> 6u;
            let j = e & 63u;
            let t = t0 + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
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
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let base = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                a[p & 3u] = sa[(base + ty + 16u * p) & 1023u];
                b[p & 3u] = sb[(base + tx + 16u * p) & 1023u];
            }
            for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    acc[(p * 4u + q) & 15u] = acc[(p * 4u + q) & 15u] + a[p & 3u] * b[q & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var p: u32 = 0u; p < 4u; p = p + 1u) {
        let lr = row0 + ty + 16u * p;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let c = col0 + tx + 16u * q;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(p * 4u + q) & 15u];
                }
            }
        }
    }
}
