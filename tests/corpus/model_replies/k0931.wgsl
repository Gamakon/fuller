// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 8u + tx;
    let col0 = wid.x * 32u;
    let row0 = wid.y * 32u;
    var acc: array<vec4<f32>, 4>;
    for (var q = 0u; q < 4u; q = q + 1u) {
        acc[q & 3u] = vec4<f32>(0.0);
    }
    let cl = tid & 31u;
    let lrow = row0 + cl;
    let grow = cfg.r_off + lrow;
    let rowok = (lrow < cfg.r_rows) && (grow < d);
    let ccol = col0 + cl;
    let colok = ccol < d;

    var t0 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var j = 0u; j < 8u; j = j + 1u) {
            let item = (tid >> 5u) + j * 2u;
            let t = t0 + item;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowok) { va = phi[base + grow]; }
                if (colok) { vb = phi[base + ccol]; }
            }
            let si = item * 32u + cl;
            sa[si & 511u] = va;
            sb[si & 511u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let ra = ty * 4u + o;
            let rb = tx * 4u + o;
            let a = vec4<f32>(sa[ra & 511u], sa[(ra + 1u) & 511u], sa[(ra + 2u) & 511u], sa[(ra + 3u) & 511u]);
            let b = vec4<f32>(sb[rb & 511u], sb[(rb + 1u) & 511u], sb[(rb + 2u) & 511u], sb[(rb + 3u) & 511u]);
            acc[0u] = acc[0u] + a.x * b;
            acc[1u] = acc[1u] + a.y * b;
            acc[2u] = acc[2u] + a.z * b;
            acc[3u] = acc[3u] + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let c = col0 + tx * 4u;
            let base = lr * d + c;
            if (c < d) { h[base] = h[base] + v.x; }
            if (c + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
