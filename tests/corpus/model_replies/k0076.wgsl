// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // cooperative load of 8 items x 64 rows and 8 items x 64 cols (zero-padded remainder)
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let idx = l + s * 256u;
            let k = idx >> 6u;
            let p = idx & 63u;
            let t = t0 + k;
            let lr = rowbase + p;
            let gr = cfg.r_off + lr;
            let c = colbase + p;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[idx & 511u] = va;
            sb[idx & 511u] = vb;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let o = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sa[(o + ty + 16u * i) & 511u];
                b[i & 3u] = sb[(o + tx + 16u * i) & 511u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = acc[q] + a[i & 3u] * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colbase + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
