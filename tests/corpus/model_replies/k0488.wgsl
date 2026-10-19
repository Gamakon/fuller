// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var f16r: f32 = 0.0; var f17: f32 = 0.0; var f18: f32 = 0.0; var f19: f32 = 0.0; var f20: f32 = 0.0; var f21: f32 = 0.0; var f22: f32 = 0.0; var f23: f32 = 0.0;
    var f24: f32 = 0.0; var f25: f32 = 0.0; var f26: f32 = 0.0; var f27: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u;
    var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u;
    var b0: bool = false; var b1: bool = false;
    var t: u32 = 0u; var k: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[(k) & 63u] = 0.0f;
    }
    i0 = (lid.x + (32u * (lid.y & 1u)));
    i1 = (lid.y >> 1u);
    i2 = ((wid.y * 64u) + i0);
    i3 = (cfg.r_off + i2);
    i4 = ((wid.x * 64u) + i0);
    b0 = ((i2 < cfg.r_rows) && (i3 < cfg.d));
    b1 = (i4 < cfg.d);
    for (t = 0u; t < cfg.m; t = t + 32u) {
    for (k = 0u; k < 8u; k = k + 1u) {
    f26 = 0.0f;
    f27 = 0.0f;
    i5 = (i1 + (4u * k));
    i6 = (t + i5);
    if ((i6 < cfg.m)) {
    i7 = (i6 * cfg.d);
    if (b0) {
    f26 = phi[(i7 + i3)];
    }
    if (b1) {
    f27 = phi[(i7 + i4)];
    }
    }
    wg_a[(((i5 * 64u) + i0)) & 2047u] = f26;
    wg_b[(((i5 * 64u) + i0)) & 2047u] = f27;
    }
    workgroupBarrier();
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    f8 = 0.0f;
    f9 = 0.0f;
    f10 = 0.0f;
    f11 = 0.0f;
    f12 = 0.0f;
    f13 = 0.0f;
    f14 = 0.0f;
    f15 = 0.0f;
    for (k = 0u; k < 32u; k = k + 2u) {
    i8 = (k * 64u);
    f16r = wg_a[((i8 + lid.y)) & 2047u];
    f17 = wg_a[(((i8 + lid.y) + 8u)) & 2047u];
    f18 = wg_a[(((i8 + lid.y) + 16u)) & 2047u];
    f19 = wg_a[(((i8 + lid.y) + 24u)) & 2047u];
    f20 = wg_a[(((i8 + lid.y) + 32u)) & 2047u];
    f21 = wg_a[(((i8 + lid.y) + 40u)) & 2047u];
    f22 = wg_a[(((i8 + lid.y) + 48u)) & 2047u];
    f23 = wg_a[(((i8 + lid.y) + 56u)) & 2047u];
    f24 = wg_b[((i8 + lid.x)) & 2047u];
    f25 = wg_b[(((i8 + lid.x) + 32u)) & 2047u];
    f0 = (f0 + (f16r * f24));
    f1 = (f1 + (f16r * f25));
    f2 = (f2 + (f17 * f24));
    f3 = (f3 + (f17 * f25));
    f4 = (f4 + (f18 * f24));
    f5 = (f5 + (f18 * f25));
    f6 = (f6 + (f19 * f24));
    f7 = (f7 + (f19 * f25));
    f8 = (f8 + (f20 * f24));
    f9 = (f9 + (f20 * f25));
    f10 = (f10 + (f21 * f24));
    f11 = (f11 + (f21 * f25));
    f12 = (f12 + (f22 * f24));
    f13 = (f13 + (f22 * f25));
    f14 = (f14 + (f23 * f24));
    f15 = (f15 + (f23 * f25));
    i8 = (i8 + 64u);
    f16r = wg_a[((i8 + lid.y)) & 2047u];
    f17 = wg_a[(((i8 + lid.y) + 8u)) & 2047u];
    f18 = wg_a[(((i8 + lid.y) + 16u)) & 2047u];
    f19 = wg_a[(((i8 + lid.y) + 24u)) & 2047u];
    f20 = wg_a[(((i8 + lid.y) + 32u)) & 2047u];
    f21 = wg_a[(((i8 + lid.y) + 40u)) & 2047u];
    f22 = wg_a[(((i8 + lid.y) + 48u)) & 2047u];
    f23 = wg_a[(((i8 + lid.y) + 56u)) & 2047u];
    f24 = wg_b[((i8 + lid.x)) & 2047u];
    f25 = wg_b[(((i8 + lid.x) + 32u)) & 2047u];
    f0 = (f0 + (f16r * f24));
    f1 = (f1 + (f16r * f25));
    f2 = (f2 + (f17 * f24));
    f3 = (f3 + (f17 * f25));
    f4 = (f4 + (f18 * f24));
    f5 = (f5 + (f18 * f25));
    f6 = (f6 + (f19 * f24));
    f7 = (f7 + (f19 * f25));
    f8 = (f8 + (f20 * f24));
    f9 = (f9 + (f20 * f25));
    f10 = (f10 + (f21 * f24));
    f11 = (f11 + (f21 * f25));
    f12 = (f12 + (f22 * f24));
    f13 = (f13 + (f22 * f25));
    f14 = (f14 + (f23 * f24));
    f15 = (f15 + (f23 * f25));
    }
    pa[(0u) & 63u] = (pa[(0u) & 63u] + f0);
    pa[(1u) & 63u] = (pa[(1u) & 63u] + f1);
    pa[(2u) & 63u] = (pa[(2u) & 63u] + f2);
    pa[(3u) & 63u] = (pa[(3u) & 63u] + f3);
    pa[(4u) & 63u] = (pa[(4u) & 63u] + f4);
    pa[(5u) & 63u] = (pa[(5u) & 63u] + f5);
    pa[(6u) & 63u] = (pa[(6u) & 63u] + f6);
    pa[(7u) & 63u] = (pa[(7u) & 63u] + f7);
    pa[(8u) & 63u] = (pa[(8u) & 63u] + f8);
    pa[(9u) & 63u] = (pa[(9u) & 63u] + f9);
    pa[(10u) & 63u] = (pa[(10u) & 63u] + f10);
    pa[(11u) & 63u] = (pa[(11u) & 63u] + f11);
    pa[(12u) & 63u] = (pa[(12u) & 63u] + f12);
    pa[(13u) & 63u] = (pa[(13u) & 63u] + f13);
    pa[(14u) & 63u] = (pa[(14u) & 63u] + f14);
    pa[(15u) & 63u] = (pa[(15u) & 63u] + f15);
    workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    i9 = (((wid.y * 64u) + lid.y) + (8u * k));
    if (((i9 < cfg.r_rows) && ((cfg.r_off + i9) < cfg.d))) {
    i10 = ((wid.x * 64u) + lid.x);
    if ((i10 < cfg.d)) {
    h[((i9 * cfg.d) + i10)] = (h[((i9 * cfg.d) + i10)] + pa[((k * 2u)) & 63u]);
    }
    i10 = (i10 + 32u);
    if ((i10 < cfg.d)) {
    h[((i9 * cfg.d) + i10)] = (h[((i9 * cfg.d) + i10)] + pa[(((k * 2u) + 1u)) & 63u]);
    }
    }
    }
}
