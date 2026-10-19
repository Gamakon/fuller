@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn kstep(s: f32, c: f32, term: f32) -> vec2<f32> {
    let y = term - c;
    let t = s + y;
    let nc = (t - s) - y;
    return vec2<f32>(t, nc);
}

@compute @workgroup_size(128, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 128u;
    let r = id + buffer_6_global[0u];
    if (r >= buffer_5_global[0u]) {
        return;
    }
    let base = r * 10u;
    let klo = buffer_5_global[1u];
    let khi = buffer_5_global[2u];
    let h1 = fmix(klo ^ r);

    let a0 = buffer_0_global[base];
    let a1 = buffer_0_global[base + 1u];
    let a2 = buffer_0_global[base + 2u];
    let a3 = buffer_0_global[base + 3u];
    let a4 = buffer_0_global[base + 4u];
    let a5 = buffer_0_global[base + 5u];
    let a6 = buffer_0_global[base + 6u];
    let a7 = buffer_0_global[base + 7u];
    let a8 = buffer_0_global[base + 8u];
    let a9 = buffer_0_global[base + 9u];

    let x0 = buffer_1_global[a0];
    let x1 = buffer_1_global[a1];
    let x2 = buffer_1_global[a2];
    let x3 = buffer_1_global[a3];
    let x4 = buffer_1_global[a4];
    let x5 = buffer_1_global[a5];
    let x6 = buffer_1_global[a6];
    let x7 = buffer_1_global[a7];
    let x8 = buffer_1_global[a8];
    let x9 = buffer_1_global[a9];

    let c0 = buffer_2_global[fmix(h1 ^ (khi ^ 0u)) >> 22u];
    let c1 = buffer_2_global[fmix(h1 ^ (khi ^ 1u)) >> 22u];
    let c2 = buffer_2_global[fmix(h1 ^ (khi ^ 2u)) >> 22u];
    let c3 = buffer_2_global[fmix(h1 ^ (khi ^ 3u)) >> 22u];
    let c4 = buffer_2_global[fmix(h1 ^ (khi ^ 4u)) >> 22u];
    let c5 = buffer_2_global[fmix(h1 ^ (khi ^ 5u)) >> 22u];
    let c6 = buffer_2_global[fmix(h1 ^ (khi ^ 6u)) >> 22u];
    let c7 = buffer_2_global[fmix(h1 ^ (khi ^ 7u)) >> 22u];
    let c8 = buffer_2_global[fmix(h1 ^ (khi ^ 8u)) >> 22u];
    let c9 = buffer_2_global[fmix(h1 ^ (khi ^ 9u)) >> 22u];

    var k = vec2<f32>(0.0, 0.0);
    k = kstep(k.x, k.y, c0 * x0);
    k = kstep(k.x, k.y, c1 * x1);
    k = kstep(k.x, k.y, c2 * x2);
    k = kstep(k.x, k.y, c3 * x3);
    k = kstep(k.x, k.y, c4 * x4);
    k = kstep(k.x, k.y, c5 * x5);
    k = kstep(k.x, k.y, c6 * x6);
    k = kstep(k.x, k.y, c7 * x7);
    k = kstep(k.x, k.y, c8 * x8);
    k = kstep(k.x, k.y, c9 * x9);
    let acc_z = k.x;

    var acc_x = 0.0;
    acc_x = acc_x + x0;
    acc_x = acc_x + x1;
    acc_x = acc_x + x2;
    acc_x = acc_x + x3;
    acc_x = acc_x + x4;
    acc_x = acc_x + x5;
    acc_x = acc_x + x6;
    acc_x = acc_x + x7;
    acc_x = acc_x + x8;
    acc_x = acc_x + x9;

    let nh = fmix(fmix(buffer_6_global[1u] ^ r) ^ (buffer_6_global[2u] ^ buffer_6_global[3u]));
    let z = buffer_2_global[nh >> 22u];

    let pre = buffer_7_global[1u] * acc_z + buffer_7_global[0u] * acc_x + buffer_3_global[r] + buffer_7_global[4u] * z;
    var th = 1.0;
    if (pre <= 43.0) {
        th = tanh(pre);
    }
    let leak = buffer_7_global[2u];
    buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
}
