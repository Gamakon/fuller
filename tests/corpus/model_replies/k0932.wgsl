// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load
        for (var q = 0u; q < 4u; q = q + 1u) {
            let idx = tid + q * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            As[idx & 1023u] = av;
            Bs[idx & 1023u] = bv;
        }
        workgroupBarrier();

        for (var k = 0u; k < 16u; k = k + 1u) {
            let ab = k * 64u + ty * 4u;
            let bb = k * 64u + tx * 4u;
            let b = vec4<f32>(
                Bs[bb & 1023u],
                Bs[(bb + 1u) & 1023u],
                Bs[(bb + 2u) & 1023u],
                Bs[(bb + 3u) & 1023u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a = As[(ab + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + a * b;
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
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
