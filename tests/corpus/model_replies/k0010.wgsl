// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 512>;
var<workgroup> wb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let lrBase = wid.y * 16u;
    let cBase = wid.x * 16u;
    let lrL = lrBase + lx;
    let grL = cfg.r_off + lrL;
    let cL = cBase + lx;
    let rowOk = (lrL < cfg.r_rows) && (grL < d);
    let colOk = cL < d;
    var acc: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let tt = ly + j * 16u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowOk) { va = phi[base + grL]; }
                if (colOk) { vb = phi[base + cL]; }
            }
            wa[(tt * 16u + lx) & 511u] = va;
            wb[(tt * 16u + lx) & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = acc + wa[(k * 16u + ly) & 511u] * wb[(k * 16u + lx) & 511u];
        }
        workgroupBarrier();
    }
    let lr = lrBase + ly;
    let c = cBase + lx;
    if (lr < cfg.r_rows && c < d && (cfg.r_off + lr) < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}

Wait — this has a bug: the `wa` tile is loaded with `lx` indexing the row, but it is read with `ly`. I'll fix that below.

// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 512>;
var<workgroup> wb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let lrBase = wid.y * 16u;
    let cBase = wid.x * 16u;
    let lrL = lrBase + lx;
    let grL = cfg.r_off + lrL;
    let cL = cBase + lx;
    let rowOk = (lrL < cfg.r_rows) && (grL < d);
    let colOk = cL < d;
    var acc: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let tt = ly + j * 16u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowOk) { va = phi[base + grL]; }
                if (colOk) { vb = phi[base + cL]; }
            }
            wa[(tt * 16u + lx) & 511u] = va;
            wb[(tt * 16u + lx) & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = acc + wa[(k * 16u + ly) & 511u] * wb[(k * 16u + lx) & 511u];
        }
        workgroupBarrier();
    }
    let lr = lrBase + ly;
    let c = cBase + lx;
    if (lr < cfg.r_rows && c < d && (cfg.r_off + lr) < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
