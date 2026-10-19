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
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        acc[q & 3u] = vec4<f32>(0.0);
    }

    // per-thread load coordinates (i constant across n since 256 % 64 == 0)
    let li = tid & 63u;
    let lk0 = tid >> 6u;
    let lr_ = rowBase + li;
    let grr = cfg.r_off + lr_;
    let rowOk = (lr_ < cfg.r_rows) && (grr < d);
    let cc = colBase + li;
    let colOk = cc < d;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var n: u32 = 0u; n < 4u; n = n + 1u) {
            let k = lk0 + n * 4u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowOk) { va = phi[base + grr]; }
                if (colOk) { vb = phi[base + cc]; }
            }
            let si = (k * 64u + li) & 1023u;
            sa[si] = va;
            sb[si] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let ao = o + ty * 4u;
            let bo = o + tx * 4u;
            let a = vec4<f32>(sa[ao & 1023u], sa[(ao + 1u) & 1023u], sa[(ao + 2u) & 1023u], sa[(ao + 3u) & 1023u]);
            let b = vec4<f32>(sb[bo & 1023u], sb[(bo + 1u) & 1023u], sb[(bo + 2u) & 1023u], sb[(bo + 3u) & 1023u]);
            acc[0u] = acc[0u] + a.x * b;
            acc[1u] = acc[1u] + a.y * b;
            acc[2u] = acc[2u] + a.z * b;
            acc[3u] = acc[3u] + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + tx * 4u;
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        let lr = rowBase + ty * 4u + q;
        if (lr < cfg.r_rows) {
            let v = acc[q & 3u];
            let idx = lr * d + c0;
            if (c0 < d) { h[idx] = h[idx] + v.x; }
            if (c0 + 1u < d) { h[idx + 1u] = h[idx + 1u] + v.y; }
            if (c0 + 2u < d) { h[idx + 2u] = h[idx + 2u] + v.z; }
            if (c0 + 3u < d) { h[idx + 3u] = h[idx + 3u] + v.w; }
        }
    }
}
