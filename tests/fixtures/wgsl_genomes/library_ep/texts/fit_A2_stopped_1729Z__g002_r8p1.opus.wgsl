// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let lx = lid.x;
    let ly = lid.y;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tb: u32 = 0u; tb < ntiles; tb = tb + 1u) {
        let t0 = tb * 16u;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let k = idx / 64u;
            let rr = idx % 64u;
            let t = t0 + k;
            let lr = row0 + rr;
            let gr = cfg.r_off + lr;
            var va: f32 = 0.0;
            if (t < cfg.m && lr < cfg.r_rows && gr < cfg.d) {
                va = phi[t * cfg.d + gr];
            }
            wg_a[idx] = va;
            let c = col0 + rr;
            var vb: f32 = 0.0;
            if (t < cfg.m && c < cfg.d) {
                vb = phi[t * cfg.d + c];
            }
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        var p0: vec4<f32> = vec4<f32>(0.0);
        var p1: vec4<f32> = vec4<f32>(0.0);
        var p2: vec4<f32> = vec4<f32>(0.0);
        var p3: vec4<f32> = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let base = k * 64u;
            let a = vec4<f32>(wg_a[base + ly], wg_a[base + ly + 16u], wg_a[base + ly + 32u], wg_a[base + ly + 48u]);
            let b = vec4<f32>(wg_b[base + lx], wg_b[base + lx + 16u], wg_b[base + lx + 32u], wg_b[base + lx + 48u]);
            p0 = p0 + a.x * b;
            p1 = p1 + a.y * b;
            p2 = p2 + a.z * b;
            p3 = p3 + a.w * b;
        }
        acc[0] = acc[0] + p0;
        acc[1] = acc[1] + p1;
        acc[2] = acc[2] + p2;
        acc[3] = acc[3] + p3;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ly + 16u * i;
        let gr = cfg.r_off + lr;
        let v = acc[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + lx + 16u * j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + v[j];
            }
        }
    }
}
