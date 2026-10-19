// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid & 15u;
    let ty = lid >> 4u;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lid + 256u * q;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
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
            let o = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sa[(o + ty + 16u * i) & 1023u];
                b[i & 3u] = sb[(o + tx + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let ai = a[i & 3u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let idx = (i * 4u + j) & 15u;
                    acc[idx] = acc[idx] + ai * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
