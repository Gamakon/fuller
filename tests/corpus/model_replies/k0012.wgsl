// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let col0 = wg.x * 64u;
    let row0 = wg.y * 64u;
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let e = l + j * 256u;
            let k = e >> 6u;
            let cc = e & 63u;
            let t = t0 + k;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                let gc = col0 + cc;
                if (gc < d) {
                    bv = phi[t * d + gc];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
            }
            As[e & 1023u] = av;
            Bs[e & 1023u] = bv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let bb = k * 64u + tx * 4u;
            let ab = k * 64u + ty * 4u;
            let b = vec4<f32>(Bs[bb & 1023u], Bs[(bb + 1u) & 1023u], Bs[(bb + 2u) & 1023u], Bs[(bb + 3u) & 1023u]);
            let a = vec4<f32>(As[ab & 1023u], As[(ab + 1u) & 1023u], As[(ab + 2u) & 1023u], As[(ab + 3u) & 1023u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let base = lr * d;
            let c0 = col0 + tx * 4u;
            if (c0 < d) { h[base + c0] = h[base + c0] + v.x; }
            if (c0 + 1u < d) { h[base + c0 + 1u] = h[base + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + c0 + 2u] = h[base + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + c0 + 3u] = h[base + c0 + 3u] + v.w; }
        }
    }
}
