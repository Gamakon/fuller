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
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;
    var acc0: array<f32, 16>;
    var acc1: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc0[q & 15u] = 0.0;
        acc1[q & 15u] = 0.0;
    }
    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            let c = colBase + j;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 2u) {
            let k0 = k * 64u;
            let k1 = k0 + 64u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let a0 = sa[(k0 + lid.y + 16u * j) & 1023u];
                let a1 = sa[(k1 + lid.y + 16u * j) & 1023u];
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let b0 = sb[(k0 + lid.x + 16u * i) & 1023u];
                    let b1 = sb[(k1 + lid.x + 16u * i) & 1023u];
                    let q = (j * 4u + i) & 15u;
                    acc0[q] = acc0[q] + a0 * b0;
                    acc1[q] = acc1[q] + a1 * b1;
                }
            }
        }
        workgroupBarrier();
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + lid.y + 16u * j;
        if (lr < cfg.r_rows) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = colBase + lid.x + 16u * i;
                if (c < d) {
                    let q = (j * 4u + i) & 15u;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (acc0[q] + acc1[q]);
                }
            }
        }
    }
}
