//! Every kernel this crate compiles for a device keeps wgpu's runtime bounds
//! checks, and what those checks DO under wgpu 30 on Metal (re-read 2026-10-21 in
//! the sources the lock file resolves; the old 0.20.1 reading no longer holds).
//!
//! WHAT wgpu 30.0.1 DOES, with the lines read:
//!
//! * `Device::create_shader_module` passes `wgt::ShaderRuntimeChecks::checked()`
//!   (wgpu-30.0.1/src/api/device.rs:139-143); `checked()` is `all(true)`
//!   (wgpu-types-30.0.1/src/shader.rs:64-66), `bounds_checks: true`
//!   (shader.rs:24). Only `create_shader_module_trusted` (device.rs:181-190, with
//!   `ShaderRuntimeChecks::unchecked()`, shader.rs:75), the deprecated
//!   `create_shader_module_unchecked` (device.rs:158-163) and the passthrough
//!   modules (wgpu-core-30.0.1 resource.rs:2499) turn them off.
//! * The Metal backend maps `bounds_checks: true` to naga's
//!   `BoundsCheckPolicy::Restrict`, and to `Unchecked` otherwise, for the index,
//!   buffer and image-load policies (wgpu-hal-30.0.1/src/metal/device.rs:185-189,
//!   216-222).
//! * `Restrict` is NOT read-zero/skip-write: naga's definition is "replace
//!   out-of-bounds indexes with some arbitrary in-bounds index"
//!   (naga-30.0.1/src/proc/index.rs:43-50), and the MSL writer emits
//!   `metal::min(unsigned(index), upper_bound)` (naga-30.0.1/src/back/msl/
//!   writer.rs:3522-3530). So on Metal an out-of-range load reads the LAST
//!   element and an out-of-range store WRITES the last element. Under wgpu
//!   0.20.1 the policy was `ReadZeroSkipWrite` (a load past the buffer read zero
//!   and a store was skipped); `ReadZeroSkipWrite` is still defined
//!   (index.rs:51-60) but wgpu-hal 30's Metal backend no longer selects it.
//!
//! CONSEQUENCE, recorded here and not hidden: `src/wgsl/cpu.rs`, `interp.rs`,
//! `harness.rs` and `render.rs` model out-of-range accesses as
//! `ReadZeroSkipWrite`. For a kernel whose indices are all in range (what the
//! safety layers prove) the two agree; for one that is not, the device clamps
//! where the CPU reference reads zero.
//!
//! The evolved kernels compile at `src/wgsl/gram.rs` (`Gram::compile_with`),
//! `src/wgsl/oracle.rs` (the oracle's device) and `src/wgsl/harness.rs` (the
//! generic problem harness); phylu compiles them through `fuller::wgsl::gram`.
//!
//! This test pins the facts it can: no source of the crate names a way to turn
//! the checks off, the wgpu the lock file resolves is the one read above (an
//! upgrade fails here until its policy is read again), and, when the registry
//! sources are on disk, that they still say what was read.

use std::path::Path;

/// The calls and types that make a module without runtime checks.
const UNCHECKED: [&str; 8] = [
    "create_shader_module_unchecked",
    "create_shader_module_trusted",
    "create_shader_module_passthrough",
    "create_shader_module_spirv",
    "ShaderRuntimeChecks::unchecked",
    "ShaderRuntimeChecks::all(",
    "ShaderRuntimeChecks {",
    "ShaderBoundChecks",
];

fn sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.unwrap_or_else(|e| panic!("{}: {e}", dir.display())).path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_kernel_compiles_with_runtime_bounds_checks() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in ["src", "examples"] {
        sources(&root.join(dir), &mut files);
    }
    let mut compiles = 0usize;
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        for bad in UNCHECKED {
            assert!(!text.contains(bad), "{} names {bad}: a kernel compiled without wgpu's runtime bounds checks", f.display());
        }
        compiles += text.matches("create_shader_module(").count();
    }
    // The evolved kernels' two compile sites are among them.
    for (file, site) in [("src/wgsl/gram.rs", "dev.device.create_shader_module(wgpu::ShaderModuleDescriptor"), ("src/wgsl/oracle.rs", "self.device.create_shader_module(wgpu::ShaderModuleDescriptor"), ("src/wgsl/harness.rs", "dev.device.create_shader_module(wgpu::ShaderModuleDescriptor")] {
        let text = std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(text.contains(site), "{file}: the kernels' compile site moved; read it again");
    }
    assert!(compiles >= 2, "{compiles} compile sites");
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock");
    for (name, version) in [("wgpu", "30.0.1"), ("wgpu-hal", "30.0.1"), ("wgpu-core", "30.0.1"), ("wgpu-types", "30.0.1"), ("naga", "30.0.1")] {
        assert!(lock.contains(&format!("name = \"{name}\"\nversion = \"{version}\"")), "{name} is no longer {version}: read create_shader_module's bounds-check policy again (see this file's header)");
    }
    // The sources the header cites, when the registry is on disk.
    let cargo_home = std::env::var_os("CARGO_HOME").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cargo")));
    let registry = cargo_home.map(|h| h.join("registry/src")).and_then(|d| std::fs::read_dir(d).ok()).and_then(|mut it| it.find_map(|e| e.ok().map(|e| e.path()).filter(|p| p.join("wgpu-hal-30.0.1").is_dir())));
    let Some(registry) = registry else {
        eprintln!("SKIPPED the source reads: no cargo registry holding wgpu-hal-30.0.1; the lock-file and call-site checks above ran");
        return;
    };
    let read = |rel: &str| std::fs::read_to_string(registry.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
    let metal = read("wgpu-hal-30.0.1/src/metal/device.rs");
    assert!(
        metal.contains("let bounds_check_policy = if stage.module.runtime_checks.bounds_checks {\n                    naga::proc::BoundsCheckPolicy::Restrict\n                } else {\n                    naga::proc::BoundsCheckPolicy::Unchecked"),
        "wgpu-hal 30's Metal backend no longer maps bounds_checks to Restrict/Unchecked: read it again"
    );
    assert!(!metal.contains("ReadZeroSkipWrite"), "the Metal backend names ReadZeroSkipWrite now: the policy changed, read it again");
    let device = read("wgpu-30.0.1/src/api/device.rs");
    assert!(device.contains(".create_shader_module(desc, wgt::ShaderRuntimeChecks::checked());"), "Device::create_shader_module no longer passes checked()");
    let restrict = read("naga-30.0.1/src/proc/index.rs");
    assert!(restrict.contains("Replace out-of-bounds indexes with some arbitrary in-bounds index."), "naga's Restrict definition changed");
    let msl = read("naga-30.0.1/src/back/msl/writer.rs");
    assert!(msl.contains("let restriction_needed = if policy == index::BoundsCheckPolicy::Restrict {") && msl.contains("write!(self.out, \"{NAMESPACE}::min(unsigned(\")?;"), "naga's MSL Restrict no longer clamps with min");
}
