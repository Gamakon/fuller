// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 1024>;
var<workgroup> sB: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var e: u32 = 0u; e < 4u; e = e + 1u) {
            let idx = tid + e * 256u;
            let kk = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = rowbase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let col = colbase + j;
                if (col < d) {
                    vb = phi[t * d + col];
                }
            }
            sA[idx & 1023u] = va;
            sB[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ba = k * 64u + ty * 4u;
            let bb = k * 64u + tx * 4u;
            let a = vec4<f32>(sA[ba & 1023u], sA[(ba + 1u) & 1023u], sA[(ba + 2u) & 1023u], sA[(ba + 3u) & 1023u]);
            let b = vec4<f32>(sB[bb & 1023u], sB[(bb + 1u) & 1023u], sB[(bb + 2u) & 1023u], sB[(bb + 3u) & 1023u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let c0 = colbase + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
