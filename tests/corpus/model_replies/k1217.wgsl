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
fn generated_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let base = wid.x * 256u + off;
    let r = base + lidx;

    // stage codebook (coalesced)
    for (var i = lidx; i < 1024u; i += 256u) {
        wg_code[i] = buffer_2_global[i];
    }
    // stage sources of this workgroup's rows (coalesced)
    let first = base * 10u;
    let total = n * 10u;
    for (var j = lidx; j < 2560u; j += 256u) {
        let g = first + j;
        if g < total {
            wg_src[j] = buffer_0_global[g];
        }
    }
    workgroupBarrier();

    if r >= n {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);

    var acc: f32 = 0.0;
    var comp: f32 = 0.0;
    var accx: f32 = 0.0;
    let sb = lidx * 10u;
    for (var s = 0u; s < 10u; s++) {
        let xs = buffer_1_global[wg_src[sb + s]];
        let c = wg_code[fmix32(h1 ^ (key_hi ^ s)) >> 22u];
        let term = c * xs;
        let y = term - comp;
        let tt = acc + y;
        comp = (tt - acc) - y;
        acc = tt;
        accx += xs;
    }

    let nz_h = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_code[nz_h >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * safe_tanh(pre);
}
