// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let flat = ty * 16u + tx;
    let lk = flat >> 4u;          // item within tile 0..15
    let lq = flat & 15u;          // column quad 0..15
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    let cbase = col0 + lq * 4u;
    let rbase = row0 + lq * 4u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + lk;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = cbase + j;
                if (c < d) {
                    vb[j] = phi[base + c];
                }
                let lr = rbase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va[j] = phi[base + gr];
                }
            }
        }
        sa[flat & 255u] = va;
        sb[flat & 255u] = vb;
        workgroupBarrier();

        for (var k = 0u; k < 16u; k = k + 1u) {
            let a = sa[(k * 16u + ty) & 255u];
            let b = sb[(k * 16u + tx) & 255u];
            acc[0u] = acc[0u] + a.x * b;
            acc[1u] = acc[1u] + a.y * b;
            acc[2u] = acc[2u] + a.z * b;
            acc[3u] = acc[3u] + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            let rb = lr * d;
            let c0 = col0 + tx * 4u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = c0 + j;
                if (c < d) {
                    h[rb + c] = h[rb + c] + v[j];
                }
            }
        }
    }
}
