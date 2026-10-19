@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> ptile: array<vec4<f32>, 256>;
var<workgroup> ctile: array<vec4<f32>, 256>;
var<workgroup> rv: array<f32, 1024>;
var<workgroup> ri: array<u32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let tx = t % 16u;
    let ty = t / 16u;

    for (var pt = 0u; pt < 4u; pt += 1u) {
        let p0 = wid.x * 256u + pt * 64u;
        var bv = array<f32, 4>(-3.4e38, -3.4e38, -3.4e38, -3.4e38);
        var bi = array<u32, 4>(0u, 0u, 0u, 0u);

        for (var cb = 0u; cb < c; cb += 64u) {
            var acc = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for (var d = 0u; d < dp; d += 16u) {
                for (var m = 0u; m < 4u; m += 1u) {
                    let e = t + 256u * m;
                    let row = e / 16u;
                    let k = e % 16u;
                    var pv = 0.0;
                    var cv = 0.0;
                    let pr = p0 + row;
                    let cr = cb + row;
                    if (pr < n && d + k < dp) { pv = point_keys[pr * dp + d + k]; }
                    if (cr < c && d + k < dp) { cv = centroid_keys[cr * dp + d + k]; }
                    ptile[k * 16u + row / 4u][row % 4u] = pv;
                    ctile[k * 16u + row / 4u][row % 4u] = cv;
                }
                workgroupBarrier();
                for (var k = 0u; k < 16u; k += 1u) {
                    let a = ptile[k * 16u + ty];
                    let b = ctile[k * 16u + tx];
                    for (var i = 0u; i < 4u; i += 1u) {
                        acc[i * 4u + 0u] += a[i] * b.x;
                        acc[i * 4u + 1u] += a[i] * b.y;
                        acc[i * 4u + 2u] += a[i] * b.z;
                        acc[i * 4u + 3u] += a[i] * b.w;
                    }
                }
                workgroupBarrier();
            }
            for (var i = 0u; i < 4u; i += 1u) {
                for (var j = 0u; j < 4u; j += 1u) {
                    let ci = cb + tx * 4u + j;
                    let s = acc[i * 4u + j];
                    if (ci < c && s > bv[i]) {
                        bv[i] = s;
                        bi[i] = ci;
                    }
                }
            }
        }

        for (var i = 0u; i < 4u; i += 1u) {
            let slot = (ty * 4u + i) * 16u + tx;
            rv[slot] = bv[i];
            ri[slot] = bi[i];
        }
        workgroupBarrier();
        if (t < 64u) {
            var v = -3.4e38;
            var ix = 0xFFFFFFFFu;
            for (var q = 0u; q < 16u; q += 1u) {
                let s = t * 16u + q;
                let cv = rv[s];
                let cix = ri[s];
                if (cv > v || (cv == v && cix < ix)) {
                    v = cv;
                    ix = cix;
                }
            }
            let gp = p0 + t;
            if (gp < n) {
                assignments[row_off + gp] = ix;
            }
        }
        workgroupBarrier();
    }
}
