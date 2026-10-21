// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<vec4<f32>, 512>;
var<workgroup> wb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let tx = lid.x;
    let ty = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;

    var e0 = vec4<f32>(0.0); var e1 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0); var e3 = vec4<f32>(0.0);
    var o0 = vec4<f32>(0.0); var o1 = vec4<f32>(0.0); var o2 = vec4<f32>(0.0); var o3 = vec4<f32>(0.0);

    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let idx = lidx + 256u * j;
            let it = t + (idx >> 4u);
            let q = (idx & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (it < m) {
                let base = it * d;
                let c = c0 + q;
                if (c < d) { va.x = phi[base + c]; }
                if (c + 1u < d) { va.y = phi[base + c + 1u]; }
                if (c + 2u < d) { va.z = phi[base + c + 2u]; }
                if (c + 3u < d) { va.w = phi[base + c + 3u]; }
                let lr = r0 + q;
                let g = ro + lr;
                if (lr < rr && g < d) { vb.x = phi[base + g]; }
                if (lr + 1u < rr && g + 1u < d) { vb.y = phi[base + g + 1u]; }
                if (lr + 2u < rr && g + 2u < d) { vb.z = phi[base + g + 2u]; }
                if (lr + 3u < rr && g + 3u < d) { vb.w = phi[base + g + 3u]; }
            }
            wa[idx] = va;
            wb[idx] = vb;
        }
        workgroupBarrier();

        var l0 = vec4<f32>(0.0); var l1 = vec4<f32>(0.0); var l2 = vec4<f32>(0.0); var l3 = vec4<f32>(0.0);
        var p0 = vec4<f32>(0.0); var p1 = vec4<f32>(0.0); var p2 = vec4<f32>(0.0); var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wa[k * 16u + tx];
            let b = wb[k * 16u + ty];
            let a2 = wa[(k + 16u) * 16u + tx];
            let b2 = wb[(k + 16u) * 16u + ty];
            l0 = fma(vec4<f32>(b.x), a, l0);
            l1 = fma(vec4<f32>(b.y), a, l1);
            l2 = fma(vec4<f32>(b.z), a, l2);
            l3 = fma(vec4<f32>(b.w), a, l3);
            p0 = fma(vec4<f32>(b2.x), a2, p0);
            p1 = fma(vec4<f32>(b2.y), a2, p1);
            p2 = fma(vec4<f32>(b2.z), a2, p2);
            p3 = fma(vec4<f32>(b2.w), a2, p3);
        }
        if (((t >> 5u) & 1u) == 0u) {
            e0 = e0 + (l0 + p0); e1 = e1 + (l1 + p1); e2 = e2 + (l2 + p2); e3 = e3 + (l3 + p3);
        } else {
            o0 = o0 + (l0 + p0); o1 = o1 + (l1 + p1); o2 = o2 + (l2 + p2); o3 = o3 + (l3 + p3);
        }
        workgroupBarrier();
    }

    var acc: array<vec4<f32>, 4>;
    acc[0] = e0 + o0;
    acc[1] = e1 + o1;
    acc[2] = e2 + o2;
    acc[3] = e3 + o3;
    let cb = c0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + ty * 4u + i;
        if (lr < rr && ro + lr < d) {
            let v = acc[i];
            let rowb = lr * d;
            for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
                let c = cb + jj;
                if (c < d) {
                    h[rowb + c] = h[rowb + c] + v[jj];
                }
            }
        }
    }
}
