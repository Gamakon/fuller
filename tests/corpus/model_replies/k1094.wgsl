@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> cb: array<f32, 1024>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

// Kahan step: sc = (sum, comp)
fn kstep(sc: vec2<f32>, term: f32) -> vec2<f32> {
    let y = term - sc.y;
    let t = sc.x + y;
    let c = (t - sc.x) - y;
    return vec2<f32>(t, c);
}

fn edge(base: u32, s: u32, h1: u32, k2: u32) -> vec2<f32> {
    let src = buffer_0_global[base + s];
    let x = buffer_1_global[src];
    let w = cb[fmix(h1 ^ (k2 ^ s)) >> 22u];
    return vec2<f32>(w * x, x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    for (var i = li; i < 1024u; i += 256u) {
        cb[i] = buffer_2_global[i];
    }
    workgroupBarrier();

    let r = global_id.x + buffer_6_global[0u];
    let width = buffer_5_global[0u];
    if r >= width {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let k2 = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);
    let base = r * 10u;

    var sc = vec2<f32>(0.0, 0.0);
    var ax = 0.0;
    var e: vec2<f32>;
    e = edge(base, 0u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 1u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 2u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 3u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 4u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 5u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 6u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 7u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 8u, h1, k2); sc = kstep(sc, e.x); ax += e.y;
    e = edge(base, 9u, h1, k2); sc = kstep(sc, e.x); ax += e.y;

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let t = buffer_6_global[3u];
    let z = cb[fmix(fmix(n_lo ^ r) ^ (n_hi ^ t)) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sc.x + loc * ax + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
