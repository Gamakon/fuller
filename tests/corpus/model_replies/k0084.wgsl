// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> bs: array<f32, 1024>;
var<workgroup> as_: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let e = l + j * 256u;
            let k = e >> 6u;
            let cc = e & 63u;
            let t = t0 + k;
            var vb = 0.0;
            var va = 0.0;
            if (t < m) {
                let c = c0 + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
                let lr = r0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            bs[e & 1023u] = vb;
            as_[e & 1023u] = va;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let bb = k * 64u + tx * 4u;
            let ab = k * 64u + ty * 4u;
            let b = vec4<f32>(bs[bb & 1023u], bs[(bb + 1u) & 1023u],
                              bs[(bb + 2u) & 1023u], bs[(bb + 3u) & 1023u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a = as_[(ab + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + a * b;
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = c0 + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
