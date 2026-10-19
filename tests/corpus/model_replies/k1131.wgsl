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

@compute
@workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wg3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let lid = lid3.x;
    let wgi = wg3.y * nwg.x + wg3.x;
    let offset0 = buffer_6_global[0u];
    let width = buffer_5_global[0u];
    let base = wgi * 256u + offset0;

    for (var k = 0u; k < 4u; k++) {
        let i = k * 256u + lid;
        cb[i] = buffer_2_global[i];
    }
    for (var k = 0u; k < 10u; k++) {
        let i = k * 256u + lid;
        let row = base + i / 10u;
        var v = 0u;
        if row < width {
            v = buffer_0_global[base * 10u + i];
        }
        src[i] = v;
    }
    workgroupBarrier();

    let r = base + lid;
    if r >= width {
        return;
    }

    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let h1 = fmix(key_lo ^ r);

    var acc = 0f;
    var comp = 0f;
    var accx = 0f;
    let sb = lid * 10u;
    for (var s = 0u; s < 10u; s++) {
        let x = buffer_1_global[src[sb + s]];
        let h = fmix(h1 ^ (key_hi ^ s));
        let w = cb[h >> 22u];
        let y = w * x - comp;
        let t = acc + y;
        comp = (t - acc) - y;
        acc = t;
        accx += x;
    }

    let nlo = buffer_6_global[1u];
    let nhi = buffer_6_global[2u];
    let st = buffer_6_global[3u];
    let z = cb[fmix(fmix(nlo ^ r) ^ (nhi ^ st)) >> 22u];

    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];

    let pre = scale * acc + loc * accx + buffer_3_global[r] + sigma * z;
    var th = 1.0;
    if pre <= 43.0 {
        th = tanh(pre);
    }
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
