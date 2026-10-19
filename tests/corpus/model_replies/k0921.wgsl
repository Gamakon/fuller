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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            let c = col0 + j;
            let lr = row0 + j;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                if (c < d) { vb = phi[t * d + c]; }
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sb[e & 1023u] = vb;
            sa[e & 1023u] = va;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let bb = k * 64u + tx * 4u;
            let ab = k * 64u + ty * 4u;
            let b = vec4<f32>(sb[bb & 1023u], sb[(bb + 1u) & 1023u], sb[(bb + 2u) & 1023u], sb[(bb + 3u) & 1023u]);
            let a = vec4<f32>(sa[ab & 1023u], sa[(ab + 1u) & 1023u], sa[(ab + 2u) & 1023u], sa[(ab + 3u) & 1023u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let cb = col0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let base = lr * d;
            if (cb < d) { h[base + cb] = h[base + cb] + v.x; }
            if (cb + 1u < d) { h[base + cb + 1u] = h[base + cb + 1u] + v.y; }
            if (cb + 2u < d) { h[base + cb + 2u] = h[base + cb + 2u] + v.z; }
            if (cb + 3u < d) { h[base + cb + 3u] = h[base + cb + 3u] + v.w; }
        }
    }
}
