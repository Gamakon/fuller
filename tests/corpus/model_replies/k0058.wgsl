// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let tx = li.x;
    let ty = li.y;
    let lid = ty * 16u + tx;
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    let d = cfg.d;
    let m = cfg.m;
    var acc: array<vec4<f32>, 4>;
    for (var z: u32 = 0u; z < 4u; z = z + 1u) {
        acc[z & 3u] = vec4<f32>(0.0);
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lid + 256u * q;
            let k = e >> 6u;
            let i = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = row0 + i;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    va = phi[t * d + cfg.r_off + lr];
                }
                let c = col0 + i;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            let si = ((k << 4u) | (i >> 2u)) & 255u;
            sa[si][i & 3u] = va;
            sb[si][i & 3u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sa[((k << 4u) + ty) & 255u];
            let b = sb[((k << 4u) + tx) & 255u];
            acc[0u] = acc[0u] + a.x * b;
            acc[1u] = acc[1u] + a.y * b;
            acc[2u] = acc[2u] + a.z * b;
            acc[3u] = acc[3u] + a.w * b;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
