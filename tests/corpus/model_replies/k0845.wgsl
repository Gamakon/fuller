// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    var t0 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let gc = col0 + j;
                if (gc < d) { vb = phi[t * d + gc]; }
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sa[e] = va;
            sb[e] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i] = sa[o + lid.y + 16u * i];
                b[i] = sb[o + lid.x + 16u * i];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                for (var i = 0u; i < 4u; i = i + 1u) {
                    acc[j * 4u + i] = fma(a[j], b[i], acc[j * 4u + i]);
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y + 16u * j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = col0 + lid.x + 16u * i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[j * 4u + i];
                }
            }
        }
    }
}