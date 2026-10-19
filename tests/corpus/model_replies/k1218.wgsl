@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_cb: array<f32, 1024>;
var<workgroup> wg_src: array<u32, 2560>;

fn fmix32(h_in: u32) -> u32 {
    var h = h_in;
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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];

    let base = wid.x * 256u + off;

    // cooperative, coalesced staging of the codebook and this tile's source indices
    for (var k: u32 = 0u; k < 4u; k++) {
        let i = lidx + k * 256u;
        wg_cb[i] = buffer_2_global[i];
    }
    for (var k: u32 = 0u; k < 10u; k++) {
        let i = lidx + k * 256u;
        let row = base + i / 10u;
        if row < n {
            wg_src[i] = buffer_0_global[base * 10u + i];
        }
    }
    workgroupBarrier();

    let r = base + lidx;
    if r >= n {
        return;
    }

    let noise_lo = buffer_6_global[1u];
    let noise_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let h1 = fmix32(key_lo ^ r);
    var acc_z: f32 = 0.0;
    var comp: f32 = 0.0;
    var acc_x: f32 = 0.0;
    let sbase = lidx * 10u;

    for (var s: u32 = 0u; s < 10u; s++) {
        let src = wg_src[sbase + s];
        let xv = buffer_1_global[src];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = wg_cb[hh >> 22u];
        let term = w * xv;
        let y = term - comp;
        let tt = acc_z + y;
        comp = (tt - acc_z) - y;
        acc_z = tt;
        acc_x = acc_x + xv;
    }

    let zh = fmix32(fmix32(noise_lo ^ r) ^ (noise_hi ^ step_t));
    let z = wg_cb[zh >> 22u];

    let pre = scale * acc_z + loc * acc_x + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
