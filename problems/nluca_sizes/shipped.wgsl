// nluca generated_step (tabled wiring), the shipping f32 kernel: lines 40-203
// of noctiluca-dev/results/nluca/step_kernels_emitted.wgsl (CubeCL 0.6 for
// wgpu/Metal, degree 10 baked in), copied verbatim; the reference kernel of
// problems/nluca_step (its timing denominator).

@group(0)
@binding(0)
var<storage, read_write> buffer_0_global: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> buffer_1_global: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> buffer_2_global: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> buffer_3_global: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> buffer_4_global: array<f32>;

@group(0)
@binding(5)
var<storage, read_write> buffer_5_global: array<u32>;

@group(0)
@binding(6)
var<storage, read_write> buffer_6_global: array<u32>;

@group(0)
@binding(7)
var<storage, read_write> buffer_7_global: array<f32>;

@group(0)
@binding(8)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 256u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
var l_mut_14: u32;
var l_mut_5: f32;
var l_mut_6: f32;
var l_mut_7: f32;
let l_0 = buffer_6_global[0u];
let l_1 = id + l_0;
let l_2 = buffer_5_global[0u];
let l_3 = l_1 >= l_2;
if l_3 {
return;
}
let l_4 = l_1 * 10u;
l_mut_5 = 0f;
l_mut_6 = 0f;
l_mut_7 = 0f;

for (var l_mut_8: u32 = 0u; l_mut_8 < 10u; l_mut_8++) {
let l_9 = l_4 + l_mut_8;
let l_10 = buffer_0_global[l_9];
let l_11 = buffer_1_global[l_10];
let l_12 = buffer_5_global[1u];
let l_13 = l_12 ^ l_1;
l_mut_14 = l_13;
let l_15 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_15;
l_mut_14 = l_mut_14 * 2246822507u;
let l_16 = l_mut_14 >> u32(13u);
l_mut_14 = l_mut_14 ^ l_16;
l_mut_14 = l_mut_14 * 3266489909u;
let l_17 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_17;
let l_18 = buffer_5_global[2u];
let l_19 = l_18 ^ l_mut_8;
let l_20 = l_mut_14 ^ l_19;
l_mut_14 = l_20;
let l_21 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_21;
l_mut_14 = l_mut_14 * 2246822507u;
let l_22 = l_mut_14 >> u32(13u);
l_mut_14 = l_mut_14 ^ l_22;
l_mut_14 = l_mut_14 * 3266489909u;
let l_23 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_23;
let l_24 = l_mut_14 >> u32(22u);
let l_25 = buffer_2_global[l_24];
let l_26 = l_25 * l_11;
let l_27 = l_26 - l_mut_7;
let l_28 = l_mut_5 + l_27;
let l_29 = l_28 - l_mut_5;
let l_30 = l_29 - l_27;
l_mut_7 = l_30;
l_mut_5 = l_28;
l_mut_6 = l_mut_6 + l_11;
}
let l_31 = buffer_6_global[1u];
let l_32 = buffer_6_global[2u];
let l_33 = buffer_6_global[3u];
let l_34 = l_31 ^ l_1;
l_mut_14 = l_34;
let l_35 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_35;
l_mut_14 = l_mut_14 * 2246822507u;
let l_36 = l_mut_14 >> u32(13u);
l_mut_14 = l_mut_14 ^ l_36;
l_mut_14 = l_mut_14 * 3266489909u;
let l_37 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_37;
let l_38 = l_32 ^ l_33;
let l_39 = l_mut_14 ^ l_38;
l_mut_14 = l_39;
let l_40 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_40;
l_mut_14 = l_mut_14 * 2246822507u;
let l_41 = l_mut_14 >> u32(13u);
l_mut_14 = l_mut_14 ^ l_41;
l_mut_14 = l_mut_14 * 3266489909u;
let l_42 = l_mut_14 >> u32(16u);
l_mut_14 = l_mut_14 ^ l_42;
let l_43 = l_mut_14 >> u32(22u);
let l_44 = buffer_2_global[l_43];
let l_45 = buffer_7_global[1u];
let l_46 = l_45 * l_mut_5;
let l_47 = buffer_7_global[0u];
let l_48 = l_47 * l_mut_6;
let l_49 = l_46 + l_48;
let l_50 = buffer_3_global[l_1];
let l_51 = l_49 + l_50;
let l_52 = buffer_7_global[4u];
let l_53 = l_52 * l_44;
let l_54 = l_51 + l_53;
let l_55 = buffer_7_global[2u];
let l_56 = 1f - l_55;
let l_57 = buffer_1_global[l_1];
let l_58 = l_56 * l_57;
let l_59 = buffer_7_global[2u];
let l_60 = safe_tanh(l_54);
let l_61 = l_59 * l_60;
let l_62 = l_58 + l_61;
buffer_4_global[l_1] = l_62;
}
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
fn safe_tanh_scalar(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    } else {
        return tanh(x);
    }
}

fn safe_tanh(x: f32) -> f32 {
    return safe_tanh_scalar(x);
}





