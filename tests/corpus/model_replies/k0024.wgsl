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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let kk = e >> 6u;
            let j = e & 63u;
            let t = t0 + kk;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let c = col0 + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let bb = kk * 64u + tx * 4u;
            let b = vec4<f32>(sb[bb & 1023u], sb[(bb + 1u) & 1023u], sb[(bb + 2u) & 1023u], sb[(bb + 3u) & 1023u]);
            let ab = kk * 64u + ty * 4u;
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a = sa[(ab + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + vec4<f32>(a) * b;
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let c = col0 + tx * 4u;
            let base = lr * d + c;
            if (c < d) { h[base] = h[base] + v.x; }
            if (c + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
