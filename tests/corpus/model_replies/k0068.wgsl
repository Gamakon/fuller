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
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative tile load
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = tid + 256u * q;
            let k = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + k;
            let lr = row0 + cc;
            let gr = cfg.r_off + lr;
            let c = col0 + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[base + gr];
                }
                if (c < d) {
                    vb = phi[base + c];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ao = (k << 6u) + ty * 4u;
            let bo = (k << 6u) + tx * 4u;
            let a = vec4<f32>(sa[ao & 1023u], sa[(ao + 1u) & 1023u], sa[(ao + 2u) & 1023u], sa[(ao + 3u) & 1023u]);
            let b = vec4<f32>(sb[bo & 1023u], sb[(bo + 1u) & 1023u], sb[(bo + 2u) & 1023u], sb[(bo + 3u) & 1023u]);
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let ai = a[i & 3u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let n = i * 4u + j;
                    acc[n & 15u] = acc[n & 15u] + ai * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx * 4u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
