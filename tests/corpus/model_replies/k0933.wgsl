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
    let tid = lid.y * 16u + lid.x;
    let col_base = wid.x * 64u;
    let row_base = wid.y * 64u;
    let grow_base = cfg.r_off + row_base;

    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);

    let ax = lid.y * 4u;
    let bx = lid.x * 4u;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i = tid + k * 256u;
            let item = i >> 6u;
            let cc = i & 63u;
            let t = t0 + item;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                let gr = grow_base + cc;
                let gc = col_base + cc;
                if (gr < d && (row_base + cc) < cfg.r_rows) {
                    va = phi[base + gr];
                }
                if (gc < d) {
                    vb = phi[base + gc];
                }
            }
            sa[i & 1023u] = va;
            sb[i & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a = vec4<f32>(
                sa[(o + ax) & 1023u],
                sa[(o + ax + 1u) & 1023u],
                sa[(o + ax + 2u) & 1023u],
                sa[(o + ax + 3u) & 1023u]);
            let b = vec4<f32>(
                sb[(o + bx) & 1023u],
                sb[(o + bx + 1u) & 1023u],
                sb[(o + bx + 2u) & 1023u],
                sb[(o + bx + 3u) & 1023u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = col_base + bx;
    let r0 = row_base + ax;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + v.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
