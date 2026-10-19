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

fn fmix32(h0: u32) -> u32 {
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

@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let l = lid.x;
    let wg_lin = wid.y * nwg.x + wid.x;
    let offs = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let base = wg_lin * 256u + offs;
    let r = base + l;

    // stage codebook cooperatively
    for (var k = 0u; k < 4u; k++) {
        let i = l + k * 256u;
        wg_code[i] = buffer_2_global[i];
    }
    // stage sources cooperatively (coalesced)
    let sbase = base * 10u;
    let slimit = width * 10u;
    for (var k = 0u; k < 10u; k++) {
        let i = l + k * 256u;
        let g = sbase + i;
        var v = 0u;
        if g < slimit {
            v = buffer_0_global[g];
        }
        wg_src[i] = v;
    }
    workgroupBarrier();

    if r >= width {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);

    var acc_z = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    let sb = l * 10u;
    for (var s = 0u; s < 10u; s++) {
        let src = wg_src[sb + s];
        let x = buffer_1_global[src];
        let hh = fmix32(h1 ^ (key_hi ^ s));
        let w = wg_code[hh >> 22u];
        let prod = w * x - comp;
        let t = acc_z + prod;
        comp = (t - acc_z) - prod;
        acc_z = t;
        acc_x = acc_x + x;
    }

    let nz = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_code[nz >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * acc_z + loc * acc_x + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
