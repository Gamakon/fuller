// grid 256 1 1 1
@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;
var<workgroup> wg_a: array<f32, 1024>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < min((lit_u32(1024u)), 16777216u); t = t + 256u) {
    wg_a[(((t + lid.x)) & 1023u)] = buffer_2_global[((t + lid.x))];
    }
    workgroupBarrier();
    i0 = buffer_5_global[(lit_u32(2u))];
    i2 = 0u;
    i2 = u32((u32(buffer_5_global[(lit_u32(1u))]) ^ u32((gid.x + buffer_6_global[(lit_u32(0u))]))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i1 = i2;
    i3 = ((gid.x + buffer_6_global[(lit_u32(0u))]) * 10u);
    for (k = 0u; k < 10u; k = k + 1u) {
    i2 = 0u;
    i2 = u32((u32(i1) ^ (u32(i0) ^ u32(k))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i4 = i2;
    f2 = (f0 + ((wg_a[((((i4) >> ((22u) & 31u))) & 1023u)] * buffer_1_global[(buffer_0_global[((i3 + k))])]) - f1));
    f1 = ((f2 - f0) - ((wg_a[((((i4) >> ((22u) & 31u))) & 1023u)] * buffer_1_global[(buffer_0_global[((i3 + k))])]) - f1));
    f0 = f2;
    f3 = (f3 + buffer_1_global[(buffer_0_global[((i3 + k))])]);
    }
    i2 = 0u;
    i2 = u32((u32(buffer_6_global[(lit_u32(1u))]) ^ u32((gid.x + buffer_6_global[(lit_u32(0u))]))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i5 = i2;
    i2 = 0u;
    i2 = u32((u32(i5) ^ (u32(buffer_6_global[(lit_u32(2u))]) ^ u32(buffer_6_global[(lit_u32(3u))]))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i6 = i2;
    if ((!((gid.x + buffer_6_global[(lit_u32(0u))]) >= buffer_5_global[(lit_u32(0u))]))) {
    buffer_4_global[((gid.x + buffer_6_global[(lit_u32(0u))]))] = ((((lit_f32(1.0f) - buffer_7_global[(lit_u32(2u))]) * buffer_1_global[((gid.x + buffer_6_global[(lit_u32(0u))]))]) + (buffer_7_global[(lit_u32(2u))] * select(tanh(((((buffer_7_global[(lit_u32(1u))] * f0) + (buffer_7_global[(lit_u32(0u))] * f3)) + buffer_3_global[((gid.x + buffer_6_global[(lit_u32(0u))]))]) + (buffer_7_global[(lit_u32(4u))] * wg_a[((((i6) >> ((22u) & 31u))) & 1023u)]))), 1.0f, (((((buffer_7_global[(lit_u32(1u))] * f0) + (buffer_7_global[(lit_u32(0u))] * f3)) + buffer_3_global[((gid.x + buffer_6_global[(lit_u32(0u))]))]) + (buffer_7_global[(lit_u32(4u))] * wg_a[((((i6) >> ((22u) & 31u))) & 1023u)])) > 43.0f)))));
    }
}
