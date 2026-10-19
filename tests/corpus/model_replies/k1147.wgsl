@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

var<workgroup> wg_code: array<f32, 1024>;
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
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let l = lid.x;

    // stage codebook (coalesced)
    for (var k: u32 = 0u; k < 4u; k++) {
        let i = l + k * 256u;
        wg_code[i] = buffer_2_global[i];
    }
    // stage this workgroup's contiguous block of sources (coalesced)
    let first_row = wid.x * 256u + off;
    let base = first_row * 10u;
    let limit = width * 10u;
    for (var k: u32 = 0u; k < 10u; k++) {
        let i = l + k * 256u;
        let g = base + i;
        var v: u32 = 0u;
        if g < limit {
            v = buffer_0_global[g];
        }
        wg_src[i] = v;
    }
    workgroupBarrier();

    let r = gid.x + off;
    if r >= width {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix32(key_lo ^ r);

    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    var acc_x: f32 = 0.0;
    let lb = l * 10u;

    for (var s: u32 = 0u; s < 10u; s++) {
        let src = wg_src[lb + s];
        let x = buffer_1_global[src];
        let hh = fmix32(h0 ^ (key_hi ^ s));
        let w = wg_code[hh >> 22u];
        let term = w * x;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        acc_x = acc_x + x;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let n_t = buffer_6_global[3u];
    let z = wg_code[fmix32(fmix32(n_lo ^ r) ^ (n_hi ^ n_t)) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * acc_x + buffer_3_global[r] + sigma * z;
    let xs = buffer_1_global[r];
    buffer_4_global[r] = (1.0 - leak) * xs + leak * safe_tanh(pre);
}
