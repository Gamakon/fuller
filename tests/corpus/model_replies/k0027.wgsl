// grid 16 16 4 4
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + 256u * k;
            let it = e >> 6u;
            let off = e & 63u;
            let t = t0 + it;
            let lr = row0 + off;
            let gr = cfg.r_off + lr;
            let c = col0 + off;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lr < r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
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
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sa[(base + ty + 16u * i) & 1023u];
                b[i & 3u] = sb[(base + tx + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let ai = a[i & 3u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let qi = (i * 4u + j) & 15u;
                    acc[qi] = acc[qi] + ai * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < r_rows && gr < d) {
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
