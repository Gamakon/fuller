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

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }

    var ra: array<f32, 4>;
    var rb: array<f32, 4>;

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let it = e >> 6u;
            let j = e & 63u;
            let t = t0 + it;
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
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            for (var i = 0u; i < 4u; i = i + 1u) {
                ra[i & 3u] = sa[(o + ty + 16u * i) & 1023u];
                rb[i & 3u] = sb[(o + tx + 16u * i) & 1023u];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = acc[q] + ra[i & 3u] * rb[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
