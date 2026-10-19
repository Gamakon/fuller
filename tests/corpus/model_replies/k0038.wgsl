// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    let tx = lid & 15u;
    let ty = lid >> 4u;
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let i = lid + q * 256u;
            let k = i >> 6u;
            let j = i & 63u;
            let t = t0 + k;
            let lr = rb + j;
            let gr = ro + lr;
            let c = cb + j;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (t < m) {
                if (lr < rr && gr < d) { a = phi[t * d + gr]; }
                if (c < d) { b = phi[t * d + c]; }
            }
            sa[i & 1023u] = a;
            sb[i & 1023u] = b;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ab = k * 64u + ty * 4u;
            let bb = k * 64u + tx * 4u;
            let a0 = sa[(ab) & 1023u];
            let a1 = sa[(ab + 1u) & 1023u];
            let a2 = sa[(ab + 2u) & 1023u];
            let a3 = sa[(ab + 3u) & 1023u];
            let bv = vec4<f32>(sb[(bb) & 1023u], sb[(bb + 1u) & 1023u],
                               sb[(bb + 2u) & 1023u], sb[(bb + 3u) & 1023u]);
            acc[0] = acc[0] + a0 * bv;
            acc[1] = acc[1] + a1 * bv;
            acc[2] = acc[2] + a2 * bv;
            acc[3] = acc[3] + a3 * bv;
        }
        workgroupBarrier();
    }

    for (var ri: u32 = 0u; ri < 4u; ri = ri + 1u) {
        let lr = rb + ty * 4u + ri;
        if (lr < rr && ro + lr < d) {
            let v = acc[ri & 3u];
            let base = lr * d;
            let c0 = cb + tx * 4u;
            if (c0 < d) { h[base + c0] = h[base + c0] + v.x; }
            if (c0 + 1u < d) { h[base + c0 + 1u] = h[base + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + c0 + 2u] = h[base + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + c0 + 3u] = h[base + c0 + 3u] + v.w; }
        }
    }
}
