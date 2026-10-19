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
    let tid = lid.y * 16u + lid.x;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let it = e >> 6u;
            let cc = e & 63u;
            let t = t0 + it;
            let lrow = rowBase + cc;
            let gr = cfg.r_off + lrow;
            let c = colBase + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lrow < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
                if (c < d) { vb = phi[t * d + c]; }
            }
            sA[e & 1023u] = va;
            sB[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let ao = kk * 64u + lid.y * 4u;
            let bo = kk * 64u + lid.x * 4u;
            let a = vec4<f32>(sA[ao & 1023u], sA[(ao + 1u) & 1023u], sA[(ao + 2u) & 1023u], sA[(ao + 3u) & 1023u]);
            let b = vec4<f32>(sB[bo & 1023u], sB[(bo + 1u) & 1023u], sB[(bo + 2u) & 1023u], sB[(bo + 3u) & 1023u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let c0 = colBase + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
