// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };

@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let c = wid.x * 16u + lid.x;
    let lr = wid.y * 16u + lid.y;
    let valid_c = c < cfg.d;
    let valid_r = lr < cfg.r_rows;
    let gr = cfg.r_off + lr;
    let valid_out = valid_c && valid_r && (gr < cfg.d);

    var p0: f32 = 0.0;
    var p1: f32 = 0.0;
    var p2: f32 = 0.0;
    var p3: f32 = 0.0;
    var p4: f32 = 0.0;
    var p5: f32 = 0.0;
    var p6: f32 = 0.0;
    var p7: f32 = 0.0;

    for (var tb: u32 = 0u; tb < cfg.m; tb = tb + 128u) {
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let tk = lid.y + q * 16u;
            let t = tb + tk;
            let wi = tk * 16u + lid.x;

            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < cfg.m) {
                if (valid_c) {
                    av = phi[t * cfg.d + c];
                }
                let load_lr = wid.y * 16u + lid.x;
                let load_gr = cfg.r_off + load_lr;
                if (load_lr < cfg.r_rows && load_gr < cfg.d) {
                    bv = phi[t * cfg.d + load_gr];
                }
            }
            wg_a[wi] = av;
            wg_b[wi] = bv;
        }

        workgroupBarrier();

        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        var s4: f32 = 0.0;
        var s5: f32 = 0.0;
        var s6: f32 = 0.0;
        var s7: f32 = 0.0;

        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let k = j * 8u;
            s0 = s0 + wg_b[(k + 0u) * 16u + lid.y] * wg_a[(k + 0u) * 16u + lid.x];
            s1 = s1 + wg_b[(k + 1u) * 16u + lid.y] * wg_a[(k + 1u) * 16u + lid.x];
            s2 = s2 + wg_b[(k + 2u) * 16u + lid.y] * wg_a[(k + 2u) * 16u + lid.x];
            s3 = s3 + wg_b[(k + 3u) * 16u + lid.y] * wg_a[(k + 3u) * 16u + lid.x];
            s4 = s4 + wg_b[(k + 4u) * 16u + lid.y] * wg_a[(k + 4u) * 16u + lid.x];
            s5 = s5 + wg_b[(k + 5u) * 16u + lid.y] * wg_a[(k + 5u) * 16u + lid.x];
            s6 = s6 + wg_b[(k + 6u) * 16u + lid.y] * wg_a[(k + 6u) * 16u + lid.x];
            s7 = s7 + wg_b[(k + 7u) * 16u + lid.y] * wg_a[(k + 7u) * 16u + lid.x];
        }

        p0 = p0 + s0;
        p1 = p1 + s1;
        p2 = p2 + s2;
        p3 = p3 + s3;
        p4 = p4 + s4;
        p5 = p5 + s5;
        p6 = p6 + s6;
        p7 = p7 + s7;

        workgroupBarrier();
    }

    if (valid_out) {
        let acc = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
        let idx = lr * cfg.d + c;
        h[idx] = h[idx] + acc;
    }
}
