// nluca::gpu::generated_step_wireless::GeneratedStepWireless  (variant 1 of 1, from cubecl_wireless.log)


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
fn generated_step_wireless(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
var l_mut_6: u32;
var l_mut_12: u32;
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
let l_4 = buffer_4_global[1u];
let l_5 = l_4 ^ 1542469173u;
l_mut_6 = l_5;
let l_7 = l_mut_6 >> u32(16u);
l_mut_6 = l_mut_6 ^ l_7;
l_mut_6 = l_mut_6 * 2246822507u;
let l_8 = l_mut_6 >> u32(13u);
l_mut_6 = l_mut_6 ^ l_8;
l_mut_6 = l_mut_6 * 3266489909u;
let l_9 = l_mut_6 >> u32(16u);
l_mut_6 = l_mut_6 ^ l_9;
let l_10 = buffer_4_global[2u];
let l_11 = l_10 ^ l_1;
l_mut_12 = l_11;
let l_13 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_13;
l_mut_12 = l_mut_12 * 2246822507u;
let l_14 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_14;
l_mut_12 = l_mut_12 * 3266489909u;
let l_15 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_15;
let l_16 = l_mut_6 ^ l_mut_12;
let l_17 = buffer_4_global[3u];
l_mut_18 = 0f;
l_mut_19 = 0f;
l_mut_20 = 0f;

for (var l_mut_21: u32 = 0u; l_mut_21 < 8u; l_mut_21++) {
let l_22 = l_16 ^ l_mut_21;
l_mut_12 = l_22;
let l_23 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_23;
l_mut_12 = l_mut_12 * 2246822507u;
let l_24 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_24;
l_mut_12 = l_mut_12 * 3266489909u;
let l_25 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_25;
l_mut_6 = l_mut_12;
loop {
let l_26 = l_mut_6 & l_17;
let l_27 = l_26 >= l_2;
let l_28 = !l_27;
if l_28 {
break;
}
let l_29 = l_mut_6 ^ 2654435769u;
l_mut_12 = l_29;
let l_30 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_30;
l_mut_12 = l_mut_12 * 2246822507u;
let l_31 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_31;
l_mut_12 = l_mut_12 * 3266489909u;
let l_32 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_32;
l_mut_6 = l_mut_12;
}
let l_33 = l_mut_6 & l_17;
let l_34 = buffer_0_global[l_33];
let l_35 = buffer_4_global[1u];
let l_36 = l_35 ^ l_1;
l_mut_12 = l_36;
let l_37 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_37;
l_mut_12 = l_mut_12 * 2246822507u;
let l_38 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_38;
l_mut_12 = l_mut_12 * 3266489909u;
let l_39 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_39;
let l_40 = buffer_4_global[2u];
let l_41 = l_40 ^ l_mut_21;
let l_42 = l_mut_12 ^ l_41;
l_mut_12 = l_42;
let l_43 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_43;
l_mut_12 = l_mut_12 * 2246822507u;
let l_44 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_44;
l_mut_12 = l_mut_12 * 3266489909u;
let l_45 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_45;
let l_46 = l_mut_12 >> u32(22u);
let l_47 = buffer_1_global[l_46];
let l_48 = l_47 * l_34;
let l_49 = l_48 - l_mut_20;
let l_50 = l_mut_18 + l_49;
let l_51 = l_50 - l_mut_18;
let l_52 = l_51 - l_49;
l_mut_20 = l_52;
l_mut_18 = l_50;
l_mut_19 = l_mut_19 + l_34;
}
let l_53 = buffer_5_global[1u];
let l_54 = buffer_5_global[2u];
let l_55 = buffer_5_global[3u];
let l_56 = l_53 ^ l_1;
l_mut_12 = l_56;
let l_57 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_57;
l_mut_12 = l_mut_12 * 2246822507u;
let l_58 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_58;
l_mut_12 = l_mut_12 * 3266489909u;
let l_59 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_59;
let l_60 = l_54 ^ l_55;
let l_61 = l_mut_12 ^ l_60;
l_mut_12 = l_61;
let l_62 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_62;
l_mut_12 = l_mut_12 * 2246822507u;
let l_63 = l_mut_12 >> u32(13u);
l_mut_12 = l_mut_12 ^ l_63;
l_mut_12 = l_mut_12 * 3266489909u;
let l_64 = l_mut_12 >> u32(16u);
l_mut_12 = l_mut_12 ^ l_64;
let l_65 = l_mut_12 >> u32(22u);
let l_66 = buffer_1_global[l_65];
let l_67 = buffer_6_global[1u];
let l_68 = l_67 * l_mut_18;
let l_69 = buffer_6_global[0u];
let l_70 = l_69 * l_mut_19;
let l_71 = l_68 + l_70;
let l_72 = buffer_2_global[l_1];
let l_73 = l_71 + l_72;
let l_74 = buffer_6_global[4u];
let l_75 = l_74 * l_66;
let l_76 = l_73 + l_75;
let l_77 = buffer_6_global[2u];
let l_78 = 1f - l_77;
let l_79 = buffer_0_global[l_1];
let l_80 = l_78 * l_79;
let l_81 = buffer_6_global[2u];
let l_82 = safe_tanh(l_76);
let l_83 = l_81 * l_82;
let l_84 = l_80 + l_83;
buffer_3_global[l_1] = l_84;
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





