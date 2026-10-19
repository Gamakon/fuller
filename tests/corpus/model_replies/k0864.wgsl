// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load of 16 items x 64 rows and 16 items x 64 cols
        for (var p: u32 = 0u; p < 4u; p = p + 1u) {
            let idx = lid + p * 256u;
            let k = idx >> 6u;
            let col = idx & 63u;
            let t = t0 + k;
            let lr = rowBase + col;
            let gr = cfg.r_off + lr;
            let c = colBase + col;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            As[idx] = av;
            Bs[idx] = bv;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ao = k * 64u + ty * 4u;
            let bo = k * 64u + tx * 4u;
            let a = vec4<f32>(As[ao], As[ao + 1u], As[ao + 2u], As[ao + 3u]);
            let b = vec4<f32>(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        let gr = cfg.r_off + lr;
        var v = acc0;
        if (i == 1u) { v = acc1; }
        if (i == 2u) { v = acc2; }
        if (i == 3u) { v = acc3; }
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx * 4u + j;
                if (c < d) {
                    var x = v.x;
                    if (j == 1u) { x = v.y; }
                    if (j == 2u) { x = v.z; }
                    if (j == 3u) { x = v.w; }
                    let o = lr * d + c;
                    h[o] = h[o] + x;
                }
            }
        }
    }
}