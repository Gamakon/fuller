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
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let width = buffer_5_global[0u];
    let off = buffer_6_global[0u];
    let base = (wid.x + wid.y * nwg.x) * 256u + off;
    let l = lid.x;

    for (var k = 0u; k < 4u; k = k + 1u) {
        wg_cb[k * 256u + l] = buffer_2_global[k * 256u + l];
    }
    // coalesced staging of this workgroup's source rows
    let total = width * 10u;
    let gb = base * 10u;
    for (var j = 0u; j < 10u; j = j + 1u) {
        let li = j * 256u + l;
        let gi = gb + li;
        var v = 0u;
        if (gi < total) {
            v = buffer_0_global[gi];
        }
        wg_src[li] = v;
    }
    workgroupBarrier();

    let r = base + l;
    if (r >= width) {
        return;
    }
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);

    var sv: array<f32, 10>;
    let sb = l * 10u;
    for (var s = 0u; s < 10u; s = s + 1u) {
        sv[s] = buffer_1_global[wg_src[sb + s]];
    }

    var acc = 0.0;
    var comp = 0.0;
    var accx = 0.0;
    for (var s = 0u; s < 10u; s = s + 1u) {
        let x = sv[s];
        let hh = fmix(h1 ^ (key_hi ^ s));
        let y = wg_cb[hh >> 22u] * x - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx = accx + x;
    }

    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let step = buffer_6_global[3u];
    let hn = fmix(fmix(n_lo ^ r) ^ (n_hi ^ step));
    let z = wg_cb[hn >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = ((scale * acc + loc * accx) + buffer_3_global[r]) + sigma * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
