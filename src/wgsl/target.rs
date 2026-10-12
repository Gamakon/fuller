//! The device a kernel is admitted for: the naga capabilities its features
//! grant and its compute limits. Every validation in the kingdom (reader,
//! renderer, scaffold, oracle, gram) goes through [`Target::validate`], so a
//! kernel using a feature the device lacks (f64, subgroups on an adapter
//! without them) or a workgroup larger than its limits is refused here, not on
//! the device (Codex review 2026-10-12: every naga capability was enabled).
//!
//! The capabilities follow wgpu-core 0.21's `create_validator` exactly: they
//! come from the DEVICE's features (what was requested), not the adapter's.
//! The device in use installs its target ([`install`]); until one does, the
//! target is [`Target::recorded`], the M3 Max's oracle device as measured by
//! `examples/wgsl_adapter.rs` on 2026-10-12, so reading needs no adapter.

use std::sync::RwLock;

use naga::valid::{Capabilities, ModuleInfo, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};

/// A device's validation capabilities and compute limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub caps: Capabilities,
    pub max_invocations: u32,
    pub max_size: [u32; 3],
    pub max_workgroup_storage: u32,
}

static CURRENT: RwLock<Option<Target>> = RwLock::new(None);

/// Make `target` the one every later validation uses (the device just opened).
pub fn install(target: Target) {
    *CURRENT.write().expect("target lock") = Some(target);
}

impl Target {
    /// The M3 Max oracle device (wgpu 0.20.1, Metal): subgroups and the
    /// subgroup barrier requested; no f64, no int64 requested; Metal's
    /// downlevel flags (multisampled shading, cube arrays) present.
    pub const fn recorded() -> Target {
        Target {
            caps: Capabilities::SUBGROUP
                .union(Capabilities::SUBGROUP_BARRIER)
                .union(Capabilities::MULTISAMPLED_SHADING)
                .union(Capabilities::CUBE_ARRAY_TEXTURES),
            max_invocations: 1024,
            max_size: [1024, 1024, 1024],
            max_workgroup_storage: 32768,
        }
    }

    /// The installed device's target, else [`Target::recorded`].
    pub fn current() -> Target {
        CURRENT.read().expect("target lock").unwrap_or(Target::recorded())
    }

    /// The target of a device opened on `adapter` with `features` requested
    /// and `limits` required (wgpu-core's mapping from features to naga
    /// capabilities).
    #[cfg(feature = "gpu")]
    pub fn of_device(adapter: &wgpu::Adapter, features: wgpu::Features, limits: &wgpu::Limits) -> Target {
        use wgpu::{DownlevelFlags as D, Features as F};
        let down = adapter.get_downlevel_capabilities().flags;
        let mut caps = Capabilities::empty();
        caps.set(Capabilities::PUSH_CONSTANT, features.contains(F::PUSH_CONSTANTS));
        caps.set(Capabilities::FLOAT64, features.contains(F::SHADER_F64));
        caps.set(Capabilities::PRIMITIVE_INDEX, features.contains(F::SHADER_PRIMITIVE_INDEX));
        caps.set(Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING, features.contains(F::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
        caps.set(Capabilities::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING, features.contains(F::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING));
        caps.set(Capabilities::SAMPLER_NON_UNIFORM_INDEXING, features.contains(F::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
        caps.set(Capabilities::STORAGE_TEXTURE_16BIT_NORM_FORMATS, features.contains(F::TEXTURE_FORMAT_16BIT_NORM));
        caps.set(Capabilities::MULTIVIEW, features.contains(F::MULTIVIEW));
        caps.set(Capabilities::EARLY_DEPTH_TEST, features.contains(F::SHADER_EARLY_DEPTH_TEST));
        caps.set(Capabilities::SHADER_INT64, features.contains(F::SHADER_INT64));
        caps.set(Capabilities::MULTISAMPLED_SHADING, down.contains(D::MULTISAMPLED_SHADING));
        caps.set(Capabilities::DUAL_SOURCE_BLENDING, features.contains(F::DUAL_SOURCE_BLENDING));
        caps.set(Capabilities::CUBE_ARRAY_TEXTURES, down.contains(D::CUBE_ARRAY_TEXTURES));
        caps.set(Capabilities::SUBGROUP, features.intersects(F::SUBGROUP | F::SUBGROUP_VERTEX));
        caps.set(Capabilities::SUBGROUP_BARRIER, features.intersects(F::SUBGROUP_BARRIER));
        Target {
            caps,
            max_invocations: limits.max_compute_invocations_per_workgroup,
            max_size: [limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z],
            max_workgroup_storage: limits.max_compute_workgroup_storage_size,
        }
    }

    /// A validator with this target's capabilities and wgpu's subgroup
    /// stages and operations (compute and fragment, every operation set,
    /// when the device has subgroups).
    pub fn validator(&self, flags: ValidationFlags) -> Validator {
        let mut v = Validator::new(flags, self.caps);
        if self.caps.contains(Capabilities::SUBGROUP) {
            v.subgroup_stages(ShaderStages::COMPUTE | ShaderStages::FRAGMENT);
            use SubgroupOperationSet as S;
            v.subgroup_operations(S::BASIC | S::VOTE | S::ARITHMETIC | S::BALLOT | S::SHUFFLE | S::SHUFFLE_RELATIVE);
        }
        v
    }

    /// Validate `module` for this device: naga with its capabilities, then
    /// every compute entry point's workgroup size against the limits and the
    /// workgroup memory the module declares (each variable rounded up to 16
    /// bytes, as Metal allocates it) against the storage limit.
    /// `source` is the text the module was parsed from (the error names the
    /// line), or "" for a module built in memory.
    pub fn validate(&self, module: &naga::Module, source: &str) -> Result<ModuleInfo, String> {
        let info = self.validator(ValidationFlags::all()).validate(module).map_err(|e| e.emit_to_string(source))?;
        self.check_limits(module)?;
        Ok(info)
    }

    /// The limits half of [`Target::validate`].
    pub fn check_limits(&self, module: &naga::Module) -> Result<(), String> {
        for ep in module.entry_points.iter().filter(|ep| ep.stage == naga::ShaderStage::Compute) {
            let [x, y, z] = ep.workgroup_size;
            if x > self.max_size[0] || y > self.max_size[1] || z > self.max_size[2] {
                return Err(format!("{}: @workgroup_size({x}, {y}, {z}) exceeds the device's {:?} per axis", ep.name, self.max_size));
            }
            if u64::from(x) * u64::from(y) * u64::from(z) > u64::from(self.max_invocations) {
                return Err(format!("{}: @workgroup_size({x}, {y}, {z}) exceeds the device's {} invocations", ep.name, self.max_invocations));
            }
        }
        let bytes: u64 = module
            .global_variables
            .iter()
            .filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup)
            .map(|(_, g)| u64::from(module.types[g.ty].inner.size(module.to_ctx())).div_ceil(16) * 16)
            .sum();
        if bytes > u64::from(self.max_workgroup_storage) {
            return Err(format!("workgroup memory {bytes} bytes exceeds the device's {}", self.max_workgroup_storage));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(text: &str) -> naga::Module {
        naga::front::wgsl::parse_str(text).unwrap_or_else(|e| panic!("{}", e.emit_to_string(text)))
    }

    const PLAIN: &str = "@group(0) @binding(0) var<storage, read_write> h: array<f32>;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) { h[gid.x] = h[gid.x] + 1.0; }\n";

    /// A kernel using a feature the device lacks is not admitted: f64 on
    /// the M3 Max (no SHADER_F64), subgroups on a device without them.
    #[test]
    fn a_kernel_using_a_feature_the_device_lacks_is_not_admitted() {
        let t = Target::recorded();
        t.validate(&module(PLAIN), PLAIN).unwrap();
        let f64_text = PLAIN.replace("h[gid.x] + 1.0", "f32(f64(h[gid.x]) + 1.0lf)");
        assert!(t.validate(&module(&f64_text), &f64_text).is_err(), "f64 admitted without SHADER_F64");
        let sg = PLAIN.replace("h[gid.x] + 1.0", "subgroupAdd(h[gid.x])");
        t.validate(&module(&sg), &sg).unwrap();
        let without = Target { caps: t.caps - Capabilities::SUBGROUP - Capabilities::SUBGROUP_BARRIER, ..t };
        assert!(without.validate(&module(&sg), &sg).is_err(), "subgroups admitted without the feature");
        let every = Target { caps: Capabilities::all(), ..t };
        every.validate(&module(&f64_text), &f64_text).unwrap();
    }

    /// The workgroup size and workgroup memory are checked against the
    /// device's limits, and the reader refuses what the device would.
    #[test]
    fn a_kernel_beyond_the_devices_limits_is_not_admitted() {
        let t = Target::recorded();
        let wide = PLAIN.replace("(64, 1, 1)", "(2048, 1, 1)");
        assert!(t.validate(&module(&wide), &wide).unwrap_err().contains("per axis"));
        let square = PLAIN.replace("(64, 1, 1)", "(64, 32, 1)");
        assert!(t.validate(&module(&square), &square).unwrap_err().contains("invocations"));
        let mem = |n: u32| format!("var<workgroup> a: array<f32, {n}>;\n{}", PLAIN.replace("+ 1.0", "+ a[gid.x & 63u]"));
        t.validate(&module(&mem(8192)), &mem(8192)).unwrap();
        assert!(t.validate(&module(&mem(8193)), &mem(8193)).unwrap_err().contains("workgroup memory"));
        let Err(e) = crate::wgsl::reader::read(&square) else { panic!("the reader admitted 64 x 32 invocations") };
        assert!(e.contains("invocations"), "{e}");
    }
}
