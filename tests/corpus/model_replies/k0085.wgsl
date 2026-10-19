// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let cc = tid & 63u;
    let kq = tid >> 6u;
    let colBase = wg.x * 64u;
    let rowBase = wg.y * 64u;
    let myc = colBase + cc;
    let mylr = rowBase + cc;
    let mygr = cfg.r_off + mylr;
    let cOk = myc < d;
    let rOk = (mylr < cfg.r_rows) && (mygr < d);

    var acc0: array<f32, 16>;
    var acc1: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc0[i & 15u] = 0.0;
        acc1[i & 15u] = 0.0;
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q = 0u; q < 4u; q = q + 1u) {
            let k = kq + 4u * q;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let base = t * d;
                if (rOk) { va = phi[base + mygr]; }
                if (cOk) { vb = phi[base + myc]; }
            }
            sa[(k * 64u + cc) & 1023u] = va;
            sb[(k * 64u + cc) & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 8u; kk = kk + 1u) {
            let k0 = kk * 2u;
            let k1 = k0 + 1u;
            var a0: array<f32, 4>;
            var b0: array<f32, 4>;
            var a1: array<f32, 4>;
            var b1: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a0[i & 3u] = sa[(k0 * 64u + lid.y + 16u * i) & 1023u];
                b0[i & 3u] = sb[(k0 * 64u + lid.x + 16u * i) & 1023u];
                a1[i & 3u] = sa[(k1 * 64u + lid.y + 16u * i) & 1023u];
                b1[i & 3u] = sb[(k1 * 64u + lid.x + 16u * i) & 1023u];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let idx = (i * 4u + j) & 15u;
                    acc0[idx] = acc0[idx] + a0[i & 3u] * b0[j & 3u];
                    acc1[idx] = acc1[idx] + a1[i & 3u] * b1[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x + 16u * j;
                if (c < d) {
                    let idx = (i * 4u + j) & 15u;
                    let o = lr * d + c;
                    h[o] = h[o] + (acc0[idx] + acc1[idx]);
                }
            }
        }
    }
}
