// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

fn lb(it: u32, cc: u32, wx: u32) -> f32 {
    let c = wx * 64u + cc;
    if (it < cfg.m && c < cfg.d) {
        return phi[it * cfg.d + c];
    }
    return 0.0;
}

fn la(it: u32, cc: u32, wy: u32) -> f32 {
    let r = wy * 64u + cc;
    if (it < cfg.m && r < cfg.r_rows && (cfg.r_off + r) < cfg.d) {
        return phi[it * cfg.d + cfg.r_off + r];
    }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }
    let tid = lid.y * 16u + lid.x;
    let e0 = tid;
    let e1 = tid + 256u;
    let it0 = e0 >> 6u;
    let it1 = e1 >> 6u;
    let c0 = e0 & 63u;
    let c1 = e1 & 63u;

    // prologue: tile 0 into buffer 0
    wg_a[e0 & 2047u] = la(it0, c0, wid.y);
    wg_a[e1 & 2047u] = la(it1, c1, wid.y);
    wg_b[e0 & 2047u] = lb(it0, c0, wid.x);
    wg_b[e1 & 2047u] = lb(it1, c1, wid.x);
    workgroupBarrier();

    let ra = lid.y * 4u;
    let rb = lid.x * 4u;
    var t: u32 = 0u;
    for (; t < cfg.m; t = t + 8u) {
        let cur = ((t >> 3u) & 1u) * 1024u;
        let nxt = 1024u - cur;
        // issue prefetch of next tile into registers
        let na0 = la(t + 8u + it0, c0, wid.y);
        let na1 = la(t + 8u + it1, c1, wid.y);
        let nb0 = lb(t + 8u + it0, c0, wid.x);
        let nb1 = lb(t + 8u + it1, c1, wid.x);
        // compute on current tile
        for (var it = 0u; it < 8u; it = it + 1u) {
            let o = cur + it * 64u;
            let a = vec4<f32>(
                wg_a[(o + ra) & 2047u],
                wg_a[(o + ra + 1u) & 2047u],
                wg_a[(o + ra + 2u) & 2047u],
                wg_a[(o + ra + 3u) & 2047u]);
            let b = vec4<f32>(
                wg_b[(o + rb) & 2047u],
                wg_b[(o + rb + 1u) & 2047u],
                wg_b[(o + rb + 2u) & 2047u],
                wg_b[(o + rb + 3u) & 2047u]);
            acc[0u] = fma(vec4<f32>(a.x), b, acc[0u]);
            acc[1u] = fma(vec4<f32>(a.y), b, acc[1u]);
            acc[2u] = fma(vec4<f32>(a.z), b, acc[2u]);
            acc[3u] = fma(vec4<f32>(a.w), b, acc[3u]);
        }
        // store next tile into the other buffer
        wg_a[(nxt + e0) & 2047u] = na0;
        wg_a[(nxt + e1) & 2047u] = na1;
        wg_b[(nxt + e0) & 2047u] = nb0;
        wg_b[(nxt + e1) & 2047u] = nb1;
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let row = gid.y * 4u + i;
        if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
            let v = acc[i & 3u];
            let base = row * cfg.d + gid.x * 4u;
            let col = gid.x * 4u;
            if (col < cfg.d) { h[base] = h[base] + v.x; }
            if (col + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (col + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (col + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
