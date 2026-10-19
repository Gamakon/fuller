// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sr: array<f32, 1024>;
var<workgroup> sc: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative, coalesced load of 16 items x 64 rows and 16 items x 64 columns
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = tid + j * 256u;
            let tt = k >> 6u;
            let pos = k & 63u;
            let t = t0 + tt;
            let lrl = rowBase + pos;
            let gr = cfg.r_off + lrl;
            let c = colBase + pos;
            var rv: f32 = 0.0;
            var cv: f32 = 0.0;
            if (t < m) {
                if (lrl < cfg.r_rows && gr < d) {
                    rv = phi[t * d + gr];
                }
                if (c < d) {
                    cv = phi[t * d + c];
                }
            }
            sr[k] = rv;
            sc[k] = cv;
        }
        workgroupBarrier();

        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 64u;
            let a0 = sr[o + ly];
            let a1 = sr[o + ly + 16u];
            let a2 = sr[o + ly + 32u];
            let a3 = sr[o + ly + 48u];
            let b0 = sc[o + lx];
            let b1 = sc[o + lx + 16u];
            let b2 = sc[o + lx + 32u];
            let b3 = sc[o + lx + 48u];
            acc[0] = acc[0] + a0 * b0;
            acc[1] = acc[1] + a0 * b1;
            acc[2] = acc[2] + a0 * b2;
            acc[3] = acc[3] + a0 * b3;
            acc[4] = acc[4] + a1 * b0;
            acc[5] = acc[5] + a1 * b1;
            acc[6] = acc[6] + a1 * b2;
            acc[7] = acc[7] + a1 * b3;
            acc[8] = acc[8] + a2 * b0;
            acc[9] = acc[9] + a2 * b1;
            acc[10] = acc[10] + a2 * b2;
            acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0;
            acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2;
            acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
    }

    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + ly + j * 16u;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = colBase + lx + i * 16u;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[j * 4u + i];
                }
            }
        }
    }
}
