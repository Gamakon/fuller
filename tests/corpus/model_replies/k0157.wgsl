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
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let l = lid.x;
    let offs = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let id = gid.x + gid.y * nwg.x * 256u;
    let wbase = (id - l) + offs;
    let r = id + offs;

    for (var k = 0u; k < 4u; k = k + 1u) {
        wg_cb[k * 256u + l] = buffer_2_global[k * 256u + l];
    }
    let total = width * 10u;
    let sbase = wbase * 10u;
    for (var j = 0u; j < 10u; j = j + 1u) {
        let li = j * 256u + l;
        let gi = sbase + li;
        if (gi < total) {
            wg_src[li] = buffer_0_global[gi];
        }
    }
    workgroupBarrier();

    if (r >= width) {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);

    var sum = 0.0;
    var comp = 0.0;
    var acc_x = 0.0;
    let sb = l * 10u;
    for (var k = 0u; k < 10u; k = k + 1u) {
        let x = buffer_1_global[wg_src[sb + k]];
        let hh = fmix(h1 ^ (key_hi ^ k));
        let y = wg_cb[hh >> 22u] * x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        acc_x = acc_x + x;
    }

    let n1 = fmix(buffer_6_global[1u] ^ r);
    let nh = fmix(n1 ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = wg_cb[nh >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let pre = (((scale * sum) + (loc * acc_x)) + buffer_3_global[r]) + (sigma * z);
    let th = select(tanh(pre), 1.0, pre > 43.0);
    buffer_4_global[r] = ((1.0 - leak) * buffer_1_global[r]) + (leak * th);
}
