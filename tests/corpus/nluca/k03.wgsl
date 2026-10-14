// nluca::gpu::generated_step_window::GeneratedStepWindow  (variant 1 of 1, from cubecl_window_kernel_reads_where_the_host_says.log)


@group(0)
@binding(0)
var<storage, read_write> buffer_0_global: array<f32>;

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
var<storage, read_write> buffer_4_global: array<u32>;

@group(0)
@binding(5)
var<storage, read_write> buffer_5_global: array<u32>;

@group(0)
@binding(6)
var<storage, read_write> buffer_6_global: array<f32>;

@group(0)
@binding(7)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 256u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(256, 1, 1)
fn generated_step_window(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
var l_mut_7: u32;
var l_mut_13: u32;
var l_mut_18: f32;
var l_mut_19: f32;
var l_mut_20: f32;
let l_0 = buffer_5_global[0u];
let l_1 = id + l_0;
let l_2 = buffer_4_global[0u];
let l_3 = l_1 >= l_2;
if l_3 {
return;
}
let l_4 = buffer_4_global[3u];
let l_5 = buffer_4_global[1u];
let l_6 = l_5 ^ 1542469173u;
l_mut_7 = l_6;
let l_8 = l_mut_7 >> u32(16u);
l_mut_7 = l_mut_7 ^ l_8;
l_mut_7 = l_mut_7 * 2246822507u;
let l_9 = l_mut_7 >> u32(13u);
l_mut_7 = l_mut_7 ^ l_9;
l_mut_7 = l_mut_7 * 3266489909u;
let l_10 = l_mut_7 >> u32(16u);
l_mut_7 = l_mut_7 ^ l_10;
let l_11 = buffer_4_global[2u];
let l_12 = l_11 ^ l_1;
l_mut_13 = l_12;
let l_14 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_14;
l_mut_13 = l_mut_13 * 2246822507u;
let l_15 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_15;
l_mut_13 = l_mut_13 * 3266489909u;
let l_16 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_16;
let l_17 = l_mut_7 ^ l_mut_13;
l_mut_18 = 0f;
l_mut_19 = 0f;
l_mut_20 = 0f;

for (var l_mut_21: u32 = 0u; l_mut_21 < 8u; l_mut_21++) {
let l_22 = l_17 ^ l_mut_21;
l_mut_13 = l_22;
let l_23 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_23;
l_mut_13 = l_mut_13 * 2246822507u;
let l_24 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_24;
l_mut_13 = l_mut_13 * 3266489909u;
let l_25 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_25;
let l_26 = l_mut_13 % l_4;
let l_27 = 1u + l_26;
let l_28 = l_1 + l_27;
l_mut_13 = l_28;
let l_29 = l_mut_13 >= l_2;
if l_29 {
l_mut_13 = l_mut_13 - l_2;
}
let l_30 = buffer_0_global[l_mut_13];
let l_31 = buffer_4_global[1u];
let l_32 = l_31 ^ l_1;
l_mut_13 = l_32;
let l_33 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_33;
l_mut_13 = l_mut_13 * 2246822507u;
let l_34 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_34;
l_mut_13 = l_mut_13 * 3266489909u;
let l_35 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_35;
let l_36 = buffer_4_global[2u];
let l_37 = l_36 ^ l_mut_21;
let l_38 = l_mut_13 ^ l_37;
l_mut_13 = l_38;
let l_39 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_39;
l_mut_13 = l_mut_13 * 2246822507u;
let l_40 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_40;
l_mut_13 = l_mut_13 * 3266489909u;
let l_41 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_41;
let l_42 = l_mut_13 >> u32(22u);
let l_43 = buffer_1_global[l_42];
let l_44 = l_43 * l_30;
let l_45 = l_44 - l_mut_20;
let l_46 = l_mut_18 + l_45;
let l_47 = l_46 - l_mut_18;
let l_48 = l_47 - l_45;
l_mut_20 = l_48;
l_mut_18 = l_46;
l_mut_19 = l_mut_19 + l_30;
}
let l_49 = buffer_5_global[1u];
let l_50 = buffer_5_global[2u];
let l_51 = buffer_5_global[3u];
let l_52 = l_49 ^ l_1;
l_mut_13 = l_52;
let l_53 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_53;
l_mut_13 = l_mut_13 * 2246822507u;
let l_54 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_54;
l_mut_13 = l_mut_13 * 3266489909u;
let l_55 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_55;
let l_56 = l_50 ^ l_51;
let l_57 = l_mut_13 ^ l_56;
l_mut_13 = l_57;
let l_58 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_58;
l_mut_13 = l_mut_13 * 2246822507u;
let l_59 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_59;
l_mut_13 = l_mut_13 * 3266489909u;
let l_60 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_60;
let l_61 = l_mut_13 >> u32(22u);
let l_62 = buffer_1_global[l_61];
let l_63 = buffer_6_global[1u];
let l_64 = l_63 * l_mut_18;
let l_65 = buffer_6_global[0u];
let l_66 = l_65 * l_mut_19;
let l_67 = l_64 + l_66;
let l_68 = buffer_2_global[l_1];
let l_69 = l_67 + l_68;
let l_70 = buffer_6_global[4u];
let l_71 = l_70 * l_62;
let l_72 = l_69 + l_71;
let l_73 = buffer_6_global[2u];
let l_74 = 1f - l_73;
let l_75 = buffer_0_global[l_1];
let l_76 = l_74 * l_75;
let l_77 = buffer_6_global[2u];
let l_78 = safe_tanh(l_72);
let l_79 = l_77 * l_78;
let l_80 = l_76 + l_79;
buffer_3_global[l_1] = l_80;
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





