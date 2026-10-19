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
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let wgbase = wid.x * 256u + off;

    // stage codebook, coalesced
    for (var i = 0u; i < 1024u; i += 256u) {
        let j = i + lidx;
        wg_code[j] = buffer_2_global[j];
    }
    // stage this workgroup's source rows, coalesced
    let sbase = wgbase * 10u;
    for (var i = 0u; i < 2560u; i += 256u) {
        let j = i + lidx;
        if (wgbase + j / 10u < width) {
            wg_src[j] = buffer_0_global[sbase + j];
        }
    }
    workgroupBarrier();

    let r = wgbase + lidx;
    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix32(key_lo ^ r);

    var sum = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    let lb = lidx * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        let src = wg_src[lb + s];
        let xv = buffer_1_global[src];
        let w = wg_code[fmix32(h1 ^ (key_hi ^ s)) >> 22u];
        let term = w * xv;
        let y = term - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        accx = accx + xv;
    }

    let nh = fmix32(fmix32(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_code[nh >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * sum + loc * accx + buffer_3_global[r] + sigma * z;
    let xr = buffer_1_global[r];
    buffer_4_global[r] = (1.0 - leak) * xr + leak * safe_tanh(pre);
}
