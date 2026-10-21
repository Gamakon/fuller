// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let lr0 = wid.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i = tid + k * 256u;
            let t = i >> 6u;
            let j = i & 63u;
            let tt = t0 + t;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tt < m) {
                let base = tt * d;
                let lr = lr0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[base + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    vb = phi[base + c];
                }
            }
            let si = t * 16u + (j >> 2u);
            let comp = j & 3u;
            sa[si][comp] = va;
            sb[si][comp] = vb;
        }
        workgroupBarrier();
        for (var t: u32 = 0u; t < 16u; t = t + 1u) {
            let a = sa[t * 16u + ty];
            let b = sb[t * 16u + tx];
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    var accs = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = lr0 + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = accs[i];
            let rb = lr * d;
            let c = col0 + tx * 4u;
            if (c < d) { h[rb + c] = h[rb + c] + v.x; }
            if (c + 1u < d) { h[rb + c + 1u] = h[rb + c + 1u] + v.y; }
            if (c + 2u < d) { h[rb + c + 2u] = h[rb + c + 2u] + v.z; }
            if (c + 3u < d) { h[rb + c + 3u] = h[rb + c + 3u] + v.w; }
        }
    }
}
