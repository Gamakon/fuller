// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<vec4<f32>, 256>;
var<workgroup> Bs: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load: 16 items x 64 values, for rows and columns
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = tid + k * 256u;
            let tt = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                let lr = rowbase + cc;
                if (lr < cfg.r_rows) {
                    let gr = cfg.r_off + lr;
                    if (gr < d) { va = phi[base + gr]; }
                }
                let c = colbase + cc;
                if (c < d) { vb = phi[base + c]; }
            }
            // scatter into vec4 layout: element tt*16 + (cc>>2), lane cc&3
            let e = tt * 16u + (cc >> 2u);
            let lane = cc & 3u;
            var ea = As[e];
            ea[lane] = va;
            As[e] = ea;
            var eb = Bs[e];
            eb[lane] = vb;
            Bs[e] = eb;
        }
        workgroupBarrier();

        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let a = As[tt * 16u + ty];
            let b = Bs[tt * 16u + tx];
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = colbase + tx * 4u;
    var accs = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let rb = lr * d;
            let v = accs[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = c0 + j;
                if (c < d) {
                    h[rb + c] = h[rb + c] + v[j];
                }
            }
        }
    }
}