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
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var q = 0u; q < 2u; q = q + 1u) {
            let e = tid + q * 256u;
            let it = e >> 6u;
            let cc = e & 63u;
            let t = t0 + it;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let base = t * d;
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[base + gr];
                }
                let c = col0 + cc;
                if (c < d) {
                    vb = phi[base + c];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 8u; k = k + 1u) {
            let ao = k * 64u + ty * 4u;
            let bo = k * 64u + tx * 4u;
            let a = vec4<f32>(sa[ao & 511u], sa[(ao + 1u) & 511u], sa[(ao + 2u) & 511u], sa[(ao + 3u) & 511u]);
            let b = vec4<f32>(sb[bo & 511u], sb[(bo + 1u) & 511u], sb[(bo + 2u) & 511u], sb[(bo + 3u) & 511u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
