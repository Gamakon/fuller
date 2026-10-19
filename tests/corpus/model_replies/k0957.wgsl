// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

fn store_row(lr: u32, c0: u32, v: vec4<f32>) {
    if (lr >= cfg.r_rows) { return; }
    let base = lr * cfg.d + c0;
    if (c0 + 0u < cfg.d) { h[base + 0u] = h[base + 0u] + v.x; }
    if (c0 + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
    if (c0 + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
    if (c0 + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        // cooperative load: 2 elements of A and B per thread
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let i = tid + 256u * j;
            let item = i >> 6u;
            let cc = i & 63u;
            let t = t0 + item;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrow = row0 + cc;
                let grow = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && grow < d) {
                    va = phi[t * d + grow];
                }
                let gc = col0 + cc;
                if (gc < d) {
                    vb = phi[t * d + gc];
                }
            }
            sa[i & 511u] = va;
            sb[i & 511u] = vb;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let ra = k * 64u + ty * 4u;
            let rb = k * 64u + tx * 4u;
            let a = vec4<f32>(sa[(ra + 0u) & 511u], sa[(ra + 1u) & 511u],
                              sa[(ra + 2u) & 511u], sa[(ra + 3u) & 511u]);
            let b = vec4<f32>(sb[(rb + 0u) & 511u], sb[(rb + 1u) & 511u],
                              sb[(rb + 2u) & 511u], sb[(rb + 3u) & 511u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = col0 + tx * 4u;
    let r0 = row0 + ty * 4u;
    store_row(r0 + 0u, c0, a0);
    store_row(r0 + 1u, c0, a1);
    store_row(r0 + 2u, c0, a2);
    store_row(r0 + 3u, c0, a3);
}
