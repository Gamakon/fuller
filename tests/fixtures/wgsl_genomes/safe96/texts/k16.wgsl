// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lr = gid.y;
    let c = gid.x;
    let d = cfg.d;
    let gr = cfg.r_off + lr;
    if (lr >= cfg.r_rows || c >= d || gr >= d) {
        return;
    }
    let m = cfg.m;
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
    var a3: f32 = 0.0;
    var t: u32 = 0u;
    var base: u32 = gr;
    var basec: u32 = c;
    let m4 = m & ~3u;
    loop {
        if (t >= m4) { break; }
        let b0 = t * d;
        let b1 = b0 + d;
        let b2 = b1 + d;
        let b3 = b2 + d;
        let r0 = phi[b0 + gr];
        let r1 = phi[b1 + gr];
        let r2 = phi[b2 + gr];
        let r3 = phi[b3 + gr];
        let c0 = phi[b0 + c];
        let c1 = phi[b1 + c];
        let c2 = phi[b2 + c];
        let c3 = phi[b3 + c];
        a0 = fma(r0, c0, a0);
        a1 = fma(r1, c1, a1);
        a2 = fma(r2, c2, a2);
        a3 = fma(r3, c3, a3);
        t = t + 4u;
    }
    loop {
        if (t >= m) { break; }
        let b = t * d;
        a0 = fma(phi[b + gr], phi[b + c], a0);
        t = t + 1u;
    }
    let idx = lr * d + c;
    h[idx] = h[idx] + ((a0 + a1) + (a2 + a3));
}
