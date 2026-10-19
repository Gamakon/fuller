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
var<workgroup> src: array<u32, 2560>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let lid = lid3.x;
    let id = gid.x + gid.y * nwg.x * 256u + gid.z * nwg.x * 256u * nwg.y;
    let off = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let r = id + off;
    let wbase = (id - lid) + off;

    // stage codebook (uniform control flow)
    for (var i = 0u; i < 4u; i += 1u) {
        let j = i * 256u + lid;
        cb[j] = buffer_2_global[j];
    }
    // stage sources contiguously (coalesced)
    let sbase = wbase * 10u;
    let slimit = width * 10u;
    for (var k = 0u; k < 10u; k += 1u) {
        let j = k * 256u + lid;
        let g = sbase + j;
        var v = 0u;
        if (g < slimit) {
            v = buffer_0_global[g];
        }
        src[j] = v;
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h0 = fmix(key_lo ^ r);

    var s_: array<f32, 10>;
    let b = lid * 10u;
    for (var s = 0u; s < 10u; s += 1u) {
        s_[s] = buffer_1_global[src[b + s]];
    }

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s += 1u) {
        let w = cb[fmix(h0 ^ (key_hi ^ s)) >> 22u];
        let term = w * s_[s] - comp;
        let t = acc + term;
        comp = (t - acc) - term;
        acc = t;
        accx = accx + s_[s];
    }

    let nz = cb[fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u])) >> 22u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = ((scale * acc + loc * accx) + buffer_3_global[r]) + sigma * nz;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
