//! Every kernel this crate compiles for a device keeps wgpu's runtime bounds
//! checks (2026-10-12, Codex's review of the overflow decision): a read
//! outside a storage buffer reads zero and a write outside it is skipped.
//!
//! wgpu 0.20.1's `Device::create_shader_module` passes
//! `wgt::ShaderBoundChecks::new()`, checks on (wgpu-0.20.1/src/lib.rs:2537);
//! wgpu-hal 0.21.1's Metal backend then sets naga's
//! `BoundsCheckPolicy::ReadZeroSkipWrite` (metal/device.rs:81), and
//! `Unchecked` only for a module made by `create_shader_module_unchecked`.
//! The evolved kernels compile at `src/wgsl/gram.rs` (`Gram::compile_with`)
//! and `src/wgsl/oracle.rs` (the oracle's device); phylu compiles them
//! through `fuller::wgsl::gram`.
//!
//! This test pins both facts: no source of the crate names a way to turn the
//! checks off, and the wgpu the lock file resolves is the one read above (an
//! upgrade fails here until its policy is read again).

use std::path::Path;

/// The calls and types that make a module without runtime checks.
const UNCHECKED: [&str; 4] = ["create_shader_module_unchecked", "ShaderBoundChecks::unchecked", "create_shader_module_spirv", "create_shader_module_trusted"];

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
    for (file, site) in [("src/wgsl/gram.rs", "dev.device.create_shader_module(wgpu::ShaderModuleDescriptor"), ("src/wgsl/oracle.rs", "self.device.create_shader_module(wgpu::ShaderModuleDescriptor")] {
        let text = std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(text.contains(site), "{file}: the kernels' compile site moved; read it again");
    }
    assert!(compiles >= 2, "{compiles} compile sites");
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock");
    assert!(lock.contains("name = \"wgpu\"\nversion = \"0.20.1\""), "wgpu is no longer 0.20.1: read create_shader_module's bounds-check policy again");
}
