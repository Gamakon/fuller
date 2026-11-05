//! The cuneus visual shaders' bindings (`~/Dev/gamakon/cuneus/examples/shaders/*.wgsl`, copied unchanged under
//! `tests/corpus/cuneus_visual`): the uniform blocks and the output texture a run of each binds, the WGSL kingdom's image
//! variant (`crate::wgsl_rows`, `wgsl.image`) among them.

use crate::wgsl::image::{words_f32, BindingData};

/// The fixed uniforms of a run: the host's defaults (`cuneus/examples/<shader>.rs`) and a fixed time.
pub fn case_bindings(shader: &str, time: f32, width: u32, height: u32) -> Result<Vec<BindingData>, String> {
    let time_block = {
        let mut b = Vec::new();
        b.extend_from_slice(&time.to_le_bytes());
        b.extend_from_slice(&0.016f32.to_le_bytes());
        b.extend_from_slice(&300u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b
    };
    match shader {
        // VolumeParams { speed, intensity, color1, color2, color3, gamma, zoom, 3 pads }
        "volumepassage" => Ok(vec![
            BindingData::Uniform { group: 0, binding: 0, bytes: time_block },
            BindingData::Output { group: 1, binding: 0 },
            BindingData::Uniform { group: 1, binding: 1, bytes: words_f32(&[1.0, 0.001, 0.1, 0.3, 0.7, 0.8, 0.4, 0.2, 1.0, 1.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0]) },
        ]),
        // NebulaParams: iterations 17, formuparam 0.52, volsteps 6, stepsize 0.31, zoom 5, tile 0.35, speed 0.02, brightness 0.00062,
        // dust_intensity 1, distfading 0.95, color_variation 0.51, n_boxes 10, rotation 1, depth 5, color_mode 1, pad, rotation_x/y 0,
        // click_state 0, scale 1, exposure 1.6, gamma 0.4, 7 pads, time_scale 1, visual_mode 0, 2 pads (+ the host's 3 trailing pads)
        "nebula" => {
            let mut params = Vec::new();
            let i = |v: i32| v.to_le_bytes();
            let f = |v: f32| v.to_le_bytes();
            for w in [
                i(17), f(0.52), i(6), f(0.31), f(5.0), f(0.35), f(0.020), f(0.00062), f(1.0), f(0.95), f(0.51), f(10.0), i(1), f(5.0), i(1), f(0.0),
                f(0.0), f(0.0), i(0), f(1.0), f(1.6), f(0.4), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(1.0), i(0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0),
            ] {
                params.extend_from_slice(&w);
            }
            Ok(vec![
                BindingData::Uniform { group: 0, binding: 0, bytes: time_block },
                BindingData::Output { group: 1, binding: 0 },
                BindingData::Uniform { group: 1, binding: 1, bytes: params },
                BindingData::ZeroStorage { group: 2, binding: 0, bytes: 3 * u64::from(width) * u64::from(height) * 4 },
            ])
        }
        // Params (cuneus `examples/cuneus.rs` defaults): background_color 0.4, 3 pads, hue_color (1, 2, 3) + pad,
        // light_intensity 1.8, rim_power 3, ao_strength 0.1, env_light_strength 0.5, iridescence_power 0.2,
        // falloff_distance 1, global_light 1, alpha_threshold 1, mix_factor_scale 0.3, then the host's 11 pads.
        "cuneus" => {
            let mut w = vec![0.4f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.8, 3.0, 0.1, 0.5, 0.2, 1.0, 1.0, 1.0, 0.3];
            w.resize(28, 0.0);
            Ok(vec![
                BindingData::Uniform { group: 0, binding: 0, bytes: time_block },
                BindingData::Output { group: 1, binding: 0 },
                BindingData::Uniform { group: 1, binding: 1, bytes: words_f32(&w) },
            ])
        }
        other => Err(format!("no case for {other}")),
    }
}
