// nluca::gpu::generated_pre::GeneratedPre  (variant 2 of 3, from cubecl_sphere.log)


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
fn generated_pre(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
var l_mut_18: f32;
var l_mut_19: f32;
var l_mut_20: f32;
var l_mut_7: u32;
var l_mut_13: u32;
var l_mut_23: u32;
let l_0 = buffer_6_global[0u];
let l_1 = id + l_0;
let l_2 = buffer_5_global[0u];
let l_3 = l_1 >= l_2;
if l_3 {
return;
}
let l_4 = buffer_5_global[3u];
let l_5 = buffer_5_global[1u];
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
let l_11 = buffer_5_global[2u];
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
let l_21 = l_1 * 10u;

for (var l_mut_22: u32 = 0u; l_mut_22 < 10u; l_mut_22++) {
l_mut_23 = 0u;
let l_24 = l_17 ^ l_mut_22;
l_mut_13 = l_24;
let l_25 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_25;
l_mut_13 = l_mut_13 * 2246822507u;
let l_26 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_26;
l_mut_13 = l_mut_13 * 3266489909u;
let l_27 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_27;
l_mut_7 = l_mut_13;
loop {
let l_28 = l_mut_7 & l_4;
let l_29 = l_28 >= l_2;
let l_30 = !l_29;
if l_30 {
break;
}
let l_31 = l_mut_7 ^ 2654435769u;
l_mut_13 = l_31;
let l_32 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_32;
l_mut_13 = l_mut_13 * 2246822507u;
let l_33 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_33;
l_mut_13 = l_mut_13 * 3266489909u;
let l_34 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_34;
l_mut_7 = l_mut_13;
}
let l_35 = l_mut_7 & l_4;
l_mut_23 = l_35;
let l_36 = buffer_1_global[l_mut_23];
let l_37 = buffer_5_global[1u];
let l_38 = l_37 ^ l_1;
l_mut_13 = l_38;
let l_39 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_39;
l_mut_13 = l_mut_13 * 2246822507u;
let l_40 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_40;
l_mut_13 = l_mut_13 * 3266489909u;
let l_41 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_41;
let l_42 = buffer_5_global[2u];
let l_43 = l_42 ^ l_mut_22;
let l_44 = l_mut_13 ^ l_43;
l_mut_13 = l_44;
let l_45 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_45;
l_mut_13 = l_mut_13 * 2246822507u;
let l_46 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_46;
l_mut_13 = l_mut_13 * 3266489909u;
let l_47 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_47;
let l_48 = l_mut_13 >> u32(22u);
let l_49 = buffer_2_global[l_48];
let l_50 = l_49 * l_36;
let l_51 = l_50 - l_mut_20;
let l_52 = l_mut_18 + l_51;
let l_53 = l_52 - l_mut_18;
let l_54 = l_53 - l_51;
l_mut_20 = l_54;
l_mut_18 = l_52;
l_mut_19 = l_mut_19 + l_36;
}
let l_55 = buffer_7_global[1u];
let l_56 = l_55 * l_mut_18;
let l_57 = buffer_7_global[0u];
let l_58 = l_57 * l_mut_19;
let l_59 = l_56 + l_58;
let l_60 = buffer_3_global[l_1];
let l_61 = l_59 + l_60;
let l_62 = buffer_6_global[1u];
let l_63 = buffer_6_global[2u];
let l_64 = buffer_6_global[3u];
let l_65 = l_62 ^ l_1;
l_mut_13 = l_65;
let l_66 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_66;
l_mut_13 = l_mut_13 * 2246822507u;
let l_67 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_67;
l_mut_13 = l_mut_13 * 3266489909u;
let l_68 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_68;
let l_69 = l_63 ^ l_64;
let l_70 = l_mut_13 ^ l_69;
l_mut_13 = l_70;
let l_71 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_71;
l_mut_13 = l_mut_13 * 2246822507u;
let l_72 = l_mut_13 >> u32(13u);
l_mut_13 = l_mut_13 ^ l_72;
l_mut_13 = l_mut_13 * 3266489909u;
let l_73 = l_mut_13 >> u32(16u);
l_mut_13 = l_mut_13 ^ l_73;
let l_74 = l_mut_13 >> u32(22u);
let l_75 = buffer_2_global[l_74];
let l_76 = buffer_7_global[2u];
let l_77 = 1f - l_76;
let l_78 = buffer_1_global[l_1];
let l_79 = l_77 * l_78;
let l_80 = buffer_7_global[2u];
let l_81 = buffer_7_global[4u];
let l_82 = l_81 * l_75;
let l_83 = l_61 + l_82;
let l_84 = l_80 * l_83;
let l_85 = l_79 + l_84;
buffer_4_global[l_1] = l_85;
}


