// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

fn st(lr: u32, col: u32, v: f32) {
    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d && col < cfg.d) {
        let idx = lr * cfg.d + col;
        h[idx] = h[idx] + v;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let tx = lid.x;
    let ty = lid.y;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e >> 6u;
            let cc = e & 63u;
            let t = t0 + kk;
            let col = wid.x * 64u + cc;
            let lr = wid.y * 64u + cc;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                if (col < d) { vb = phi[t * d + col]; }
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sb[e] = vb;
            sa[e] = va;
        }
        workgroupBarrier();
        var t0v = vec4<f32>(0.0);
        var t1v = vec4<f32>(0.0);
        var t2v = vec4<f32>(0.0);
        var t3v = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a = vec4<f32>(sa[o + ty], sa[o + ty + 16u], sa[o + ty + 32u], sa[o + ty + 48u]);
            let b = vec4<f32>(sb[o + tx], sb[o + tx + 16u], sb[o + tx + 32u], sb[o + tx + 48u]);
            t0v = t0v + a.x * b;
            t1v = t1v + a.y * b;
            t2v = t2v + a.z * b;
            t3v = t3v + a.w * b;
        }
        acc0 = acc0 + t0v;
        acc1 = acc1 + t1v;
        acc2 = acc2 + t2v;
        acc3 = acc3 + t3v;
        workgroupBarrier();
    }
    let c0 = wid.x * 64u + tx;
    let r0 = wid.y * 64u + ty;
    st(r0, c0, acc0.x);
    st(r0, c0 + 16u, acc0.y);
    st(r0, c0 + 32u, acc0.z);
    st(r0, c0 + 48u, acc0.w);
    st(r0 + 16u, c0, acc1.x);
    st(r0 + 16u, c0 + 16u, acc1.y);
    st(r0 + 16u, c0 + 32u, acc1.z);
    st(r0 + 16u, c0 + 48u, acc1.w);
    st(r0 + 32u, c0, acc2.x);
    st(r0 + 32u, c0 + 16u, acc2.y);
    st(r0 + 32u, c0 + 32u, acc2.z);
    st(r0 + 32u, c0 + 48u, acc2.w);
    st(r0 + 48u, c0, acc3.x);
    st(r0 + 48u, c0 + 16u, acc3.y);
    st(r0 + 48u, c0 + 32u, acc3.z);
    st(r0 + 48u, c0 + 48u, acc3.w);
}
