// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // load tiles: 16 items x 64 values each; thread loads 4 scalars per tile
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let k = (tid >> 6u) + i * 4u;
            let j = tid & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = r0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = c0 + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            let idx = (k * 16u + (j >> 2u)) & 255u;
            let comp = j & 3u;
            var ca = sa[idx];
            var cb = sb[idx];
            ca[comp] = va;
            cb[comp] = vb;
            sa[idx] = ca;
            sb[idx] = cb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sa[(k * 16u + ty) & 255u];
            let b = sb[(k * 16u + tx) & 255u];
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let cb0 = c0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let base = lr * d + cb0;
            if (cb0 < d) { h[base] = h[base] + v.x; }
            if (cb0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (cb0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (cb0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
