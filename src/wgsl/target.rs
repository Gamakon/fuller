//! The device a kernel is admitted for: the naga capabilities its features
//! grant and its compute limits. Every validation in the kingdom (reader,
//! renderer, scaffold, oracle, gram) goes through [`Target::validate`], so a
//! kernel using a feature the device lacks (f64, subgroups on an adapter
//! without them) or a workgroup larger than its limits is refused here, not on
//! the device (Codex review 2026-10-12: every naga capability was enabled).
//!
//! The capabilities are wgpu 30's own (`wgpu_naga_bridge::
//! features_to_naga_capabilities`, which wgpu-core calls, and its
//! `create_validator`: `Validator::new` with them, the subgroup stages and
//! operations naga's own; 2026-10-19, the naga/wgpu 30 upgrade): they come
//! from the DEVICE's features (what was requested), not the adapter's.
//! The device in use installs its target ([`install`]); until one does, the
//! target is [`Target::recorded`], the M3 Max's oracle device as measured by
//! `examples/wgsl_adapter.rs` on 2026-10-12, so reading needs no adapter.

use std::sync::RwLock;

use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};

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
    /// The M3 Max oracle device (measured on wgpu 0.20.1, Metal; the same
    /// features requested under wgpu 30): subgroups and the
    /// subgroup barrier requested; no f64, no int64 requested; Metal's
    /// downlevel flags present, every one (wgpu-hal 30's Metal adapter starts
    /// from `DownlevelFlags::all()`), so wgpu 30's mapping also grants
    /// f16 values held in f32 (`pack2x16float`, `unpack2x16float`).
    pub const fn recorded() -> Target {
        Target {
            caps: Capabilities::SUBGROUP
                .union(Capabilities::SUBGROUP_BARRIER)
                .union(Capabilities::MULTISAMPLED_SHADING)
                .union(Capabilities::CUBE_ARRAY_TEXTURES)
                .union(Capabilities::SHADER_FLOAT16_IN_FLOAT32)
                // f16 values and the `enable f16;` directive: the device is opened with wgpu's SHADER_F16, which Metal on the M3 Max has.
                .union(Capabilities::SHADER_FLOAT16)
                // 16-bit normalised storage texture formats (r16snorm and the like): wgpu's TEXTURE_FORMAT_16BIT_NORM, which Metal on the M3 Max has.
                .union(Capabilities::STORAGE_TEXTURE_16BIT_NORM_FORMATS),
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
        let caps = wgpu_naga_bridge::features_to_naga_capabilities(features, adapter.get_downlevel_capabilities().flags);
        Target {
            caps,
            max_invocations: limits.max_compute_invocations_per_workgroup,
            max_size: [limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z],
            max_workgroup_storage: limits.max_compute_workgroup_storage_size,
        }
    }

    /// A validator with this target's capabilities, as wgpu 30 makes the
    /// device's (`Validator::new` fills the subgroup stages and operations
    /// from the capabilities).
    pub fn validator(&self, flags: ValidationFlags) -> Validator {
        Validator::new(flags, self.caps)
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
        // Workgroup memory per compute entry point, as wgpu validates a
        // pipeline and Metal allocates it: the workgroup variables that entry
        // point (and every function it calls) uses (2026-10-17: a module of
        // v24's `ingest` and `search` may each use most of the device's).
        for ep in module.entry_points.iter().filter(|ep| ep.stage == naga::ShaderStage::Compute) {
            let mut used = std::collections::BTreeSet::new();
            globals_used(module, &ep.function, &mut used, &mut std::collections::BTreeSet::new());
            let bytes: u64 = module
                .global_variables
                .iter()
                .filter(|(h, g)| g.space == naga::AddressSpace::WorkGroup && used.contains(h))
                .map(|(_, g)| u64::from(module.types[g.ty].inner.size(module.to_ctx())).div_ceil(16) * 16)
                .sum();
            if bytes > u64::from(self.max_workgroup_storage) {
                return Err(format!("{}: workgroup memory {bytes} bytes exceeds the device's {}", ep.name, self.max_workgroup_storage));
            }
        }
        Ok(())
    }
}

/// The global variables a function's expressions name, with those of every
/// function it calls.
fn globals_used(module: &naga::Module, f: &naga::Function, out: &mut std::collections::BTreeSet<naga::Handle<naga::GlobalVariable>>, seen: &mut std::collections::BTreeSet<naga::Handle<naga::Function>>) {
    for (_, e) in f.expressions.iter() {
        if let naga::Expression::GlobalVariable(g) = e {
            out.insert(*g);
        }
    }
    fn calls(b: &naga::Block, out: &mut Vec<naga::Handle<naga::Function>>) {
        for s in b.iter() {
            match s {
                naga::Statement::Call { function, .. } => out.push(*function),
                naga::Statement::Block(x) => calls(x, out),
                naga::Statement::If { accept, reject, .. } => {
                    calls(accept, out);
                    calls(reject, out);
                }
                naga::Statement::Switch { cases, .. } => cases.iter().for_each(|c| calls(&c.body, out)),
                naga::Statement::Loop { body, continuing, .. } => {
                    calls(body, out);
                    calls(continuing, out);
                }
                _ => {}
            }
        }
    }
    let mut callees = Vec::new();
    calls(&f.body, &mut callees);
    for c in callees {
        if seen.insert(c) {
            globals_used(module, &module.functions[c], out, seen);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(text: &str) -> naga::Module {
        crate::wgsl::naga_shape::parse(text).unwrap_or_else(|e| panic!("{e}"))
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
        // Per entry point (2026-10-17): two entry points of 20 KB each pass, as
        // wgpu validates each pipeline; one entry point using 36 KB fails,
        // named; a variable no entry point uses counts for none.
        let two = |second: &str| format!("var<workgroup> a: array<f32, 5120>;\nvar<workgroup> b: array<f32, 5120>;\nvar<workgroup> c: array<f32, 4096>;\n{}@compute @workgroup_size(64, 1, 1)\nfn other(@builtin(global_invocation_id) gid: vec3<u32>) {{ h[gid.x] = {second}; }}\n", PLAIN.replace("+ 1.0", "+ a[gid.x & 63u]"));
        t.validate(&module(&two("b[gid.x & 63u]")), &two("b[gid.x & 63u]")).unwrap();
        let both = two("b[gid.x & 63u] + c[gid.x & 63u]");
        let e = t.validate(&module(&both), &both).unwrap_err();
        assert!(e.contains("other: workgroup memory 36864 bytes exceeds the device's 32768"), "{e}");
        let unused = format!("var<workgroup> z: array<f32, 8192>;\n{}", mem(8192));
        t.validate(&module(&unused), &unused).unwrap();
        let Err(e) = crate::wgsl::reader::read(&square) else { panic!("the reader admitted 64 x 32 invocations") };
        assert!(e.contains("invocations"), "{e}");
    }
}
