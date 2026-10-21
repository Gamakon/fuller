// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let item = lidx / 16u;
    let q = lidx % 16u;
    let rowb = wid.y * 64u + q * 4u;
    let colb = wid.x * 64u + q * 4u;
    var c0: vec4<f32> = vec4<f32>(0.0);
    var c1: vec4<f32> = vec4<f32>(0.0);
    var c2: vec4<f32> = vec4<f32>(0.0);
    var c3: vec4<f32> = vec4<f32>(0.0);
    var e0: vec4<f32> = vec4<f32>(0.0);
    var e1: vec4<f32> = vec4<f32>(0.0);
    var e2: vec4<f32> = vec4<f32>(0.0);
    var e3: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0);
    var o1: vec4<f32> = vec4<f32>(0.0);
    var o2: vec4<f32> = vec4<f32>(0.0);
    var o3: vec4<f32> = vec4<f32>(0.0);
    var va: vec4<f32> = vec4<f32>(0.0);
    var vb: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        va = vec4<f32>(0.0);
        vb = vec4<f32>(0.0);
        let ti = t + item;
        if (ti < cfg.m) {
            let base = ti * cfg.d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let lr = rowb + j;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    va[j] = phi[base + cfg.r_off + lr];
                }
                let c = colb + j;
                if (c < cfg.d) {
                    vb[j] = phi[base + c];
                }
            }
        }
        wg_va[lidx] = va;
        wg_vb[lidx] = vb;
        workgroupBarrier();
        e0 = vec4<f32>(0.0); e1 = vec4<f32>(0.0); e2 = vec4<f32>(0.0); e3 = vec4<f32>(0.0);
        o0 = vec4<f32>(0.0); o1 = vec4<f32>(0.0); o2 = vec4<f32>(0.0); o3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 2u) {
            let a = wg_va[k * 16u + lid.y];
            let b = wg_vb[k * 16u + lid.x];
            e0 = e0 + a.x * b;
            e1 = e1 + a.y * b;
            e2 = e2 + a.z * b;
            e3 = e3 + a.w * b;
            let a2 = wg_va[(k + 1u) * 16u + lid.y];
            let b2 = wg_vb[(k + 1u) * 16u + lid.x];
            o0 = o0 + a2.x * b2;
            o1 = o1 + a2.y * b2;
            o2 = o2 + a2.z * b2;
            o3 = o3 + a2.w * b2;
        }
        c0 = c0 + (e0 + o0);
        c1 = c1 + (e1 + o1);
        c2 = c2 + (e2 + o2);
        c3 = c3 + (e3 + o3);
        workgroupBarrier();
    }
    pv[0] = c0;
    pv[1] = c1;
    pv[2] = c2;
    pv[3] = c3;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let row = gid.y * 4u + i;
            let col = gid.x * 4u + j;
            if (row < cfg.r_rows && col < cfg.d && (cfg.r_off + row) < cfg.d) {
                let idx = row * cfg.d + col;
                h[idx] = h[idx] + pv[i][j];
            }
        }
    }
}
