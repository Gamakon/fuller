@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn kstep(st: vec3<f32>, h: u32, x: f32) -> vec3<f32> {
    // st = (sum, comp, accx)
    let term = buffer_2_global[fmix(h) >> 22u] * x;
    let y = term - st.y;
    let ns = st.x + y;
    let nc = (ns - st.x) - y;
    return vec3<f32>(ns, nc, st.z + x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let b = r * 10u;
    let p0 = buffer_0_global[b];
    let p1 = buffer_0_global[b + 1u];
    let p2 = buffer_0_global[b + 2u];
    let p3 = buffer_0_global[b + 3u];
    let p4 = buffer_0_global[b + 4u];
    let p5 = buffer_0_global[b + 5u];
    let p6 = buffer_0_global[b + 6u];
    let p7 = buffer_0_global[b + 7u];
    let p8 = buffer_0_global[b + 8u];
    let p9 = buffer_0_global[b + 9u];
    let x0 = buffer_1_global[p0];
    let x1 = buffer_1_global[p1];
    let x2 = buffer_1_global[p2];
    let x3 = buffer_1_global[p3];
    let x4 = buffer_1_global[p4];
    let x5 = buffer_1_global[p5];
    let x6 = buffer_1_global[p6];
    let x7 = buffer_1_global[p7];
    let x8 = buffer_1_global[p8];
    let x9 = buffer_1_global[p9];
    let own = buffer_1_global[r];
    let dr = buffer_3_global[r];
    let k2 = buffer_5_global[2u];
    let hr = fmix(buffer_5_global[1u] ^ r);
    var st = vec3<f32>(0.0, 0.0, 0.0);
    st = kstep(st, hr ^ (k2 ^ 0u), x0);
    st = kstep(st, hr ^ (k2 ^ 1u), x1);
    st = kstep(st, hr ^ (k2 ^ 2u), x2);
    st = kstep(st, hr ^ (k2 ^ 3u), x3);
    st = kstep(st, hr ^ (k2 ^ 4u), x4);
    st = kstep(st, hr ^ (k2 ^ 5u), x5);
    st = kstep(st, hr ^ (k2 ^ 6u), x6);
    st = kstep(st, hr ^ (k2 ^ 7u), x7);
    st = kstep(st, hr ^ (k2 ^ 8u), x8);
    st = kstep(st, hr ^ (k2 ^ 9u), x9);
    let hn = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[hn >> 22u];
    let pre = buffer_7_global[1u] * st.x + buffer_7_global[0u] * st.z + dr + buffer_7_global[4u] * z;
    var th = 1.0;
    if (!(pre > 43.0)) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * own + leak * th;
}
