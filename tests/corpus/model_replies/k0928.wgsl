// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 1024>;
var<workgroup> sB: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let colBase = wg.x * 64u;
    let rowBase = wg.y * 64u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let i = e >> 6u;
            let j = e & 63u;
            let t = t0 + i;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = rowBase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colBase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sA[e & 1023u] = va;
            sB[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let ra = i * 64u + lid.y * 4u;
            let rb = i * 64u + lid.x * 4u;
            let a = vec4<f32>(sA[ra & 1023u], sA[(ra + 1u) & 1023u], sA[(ra + 2u) & 1023u], sA[(ra + 3u) & 1023u]);
            let b = vec4<f32>(sB[rb & 1023u], sB[(rb + 1u) & 1023u], sB[(rb + 2u) & 1023u], sB[(rb + 3u) & 1023u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        let lr = rowBase + lid.y * 4u + q;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            var v = a0;
            if (q == 1u) { v = a1; }
            if (q == 2u) { v = a2; }
            if (q == 3u) { v = a3; }
            for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                let c = colBase + lid.x * 4u + p;
                if (c < d) {
                    var x = v.x;
                    if (p == 1u) { x = v.y; }
                    if (p == 2u) { x = v.z; }
                    if (p == 3u) { x = v.w; }
                    let idx = lr * d + c;
                    h[idx] = h[idx] + x;
                }
            }
        }
    }
}
