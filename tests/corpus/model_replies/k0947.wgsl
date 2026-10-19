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
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    // per-thread staging column / row (fixed across tiles)
    let cc = tid & 63u;
    let kb = tid >> 6u;
    let gcol = c0 + cc;
    let lrow = r0 + cc;
    let grow = cfg.r_off + lrow;
    let colok = gcol < d;
    let rowok = (lrow < cfg.r_rows) && (grow < d);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = kb + j * 4u;
            let k2 = k;
            let t = t0 + k2;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowok) { va = phi[base + grow]; }
                if (colok) { vb = phi[base + gcol]; }
            }
            // k ranges 0..15 : kb in 0..3, j in 0..3 -> k = kb + 4j
            sa[(k2 * 64u + cc) & 1023u] = va;
            sb[(k2 * 64u + cc) & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ia = k * 64u + ty * 4u;
            let ib = k * 64u + tx * 4u;
            let a = vec4<f32>(sa[ia & 1023u], sa[(ia + 1u) & 1023u], sa[(ia + 2u) & 1023u], sa[(ia + 3u) & 1023u]);
            let b = vec4<f32>(sb[ib & 1023u], sb[(ib + 1u) & 1023u], sb[(ib + 2u) & 1023u], sb[(ib + 3u) & 1023u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let cb = c0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let rb = lr * d;
            if (cb < d) { h[rb + cb] = h[rb + cb] + v.x; }
            if (cb + 1u < d) { h[rb + cb + 1u] = h[rb + cb + 1u] + v.y; }
            if (cb + 2u < d) { h[rb + cb + 2u] = h[rb + cb + 2u] + v.z; }
            if (cb + 3u < d) { h[rb + cb + 3u] = h[rb + cb + 3u] + v.w; }
        }
    }
}
