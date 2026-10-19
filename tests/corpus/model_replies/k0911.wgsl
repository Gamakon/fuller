// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

fn tile_load(t0: u32, buf: u32, tid: u32, wx: u32, wy: u32) {
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let idx = tid + k * 256u;
        let it = t0 + (idx >> 6u);
        let c = idx & 63u;
        var fa: f32 = 0.0;
        var fb: f32 = 0.0;
        if (it < cfg.m) {
            let cx = wx * 64u + c;
            if (cx < cfg.d) {
                fb = phi[it * cfg.d + cx];
            }
            let rr = wy * 64u + c;
            if (rr < cfg.r_rows && (cfg.r_off + rr) < cfg.d) {
                fa = phi[it * cfg.d + cfg.r_off + rr];
            }
        }
        wg_a[(idx + 1024u * buf) & 2047u] = fa;
        wg_b[(idx + 1024u * buf) & 2047u] = fb;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<f32, 16>;
    let tid = lid.y * 16u + lid.x;
    tile_load(0u, 0u, tid, wid.x, wid.y);
    workgroupBarrier();
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        tile_load(t + 8u, ((t >> 3u) + 1u) & 1u, tid, wid.x, wid.y);
        let cur = 1024u * ((t >> 3u) & 1u);
        for (var it: u32 = 0u; it < 8u; it = it + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[(cur + it * 64u + lid.y * 4u + i) & 2047u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b = wg_b[(cur + it * 64u + lid.x * 4u + j) & 2047u];
                    acc[(i * 4u + j) & 15u] = fma(a, b, acc[(i * 4u + j) & 15u]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let row = gid.y * 4u + i;
        if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let col = gid.x * 4u + j;
                if (col < cfg.d) {
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
