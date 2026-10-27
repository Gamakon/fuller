//! Write `webgpu/webgpu.h` from the crate's Rust definitions, with cbindgen,
//! into `OUT_DIR/include/webgpu/webgpu.h` — on every build, so the header the
//! C compiles against can never drift from the functions it links to.
//!
//! The committed `include/webgpu/webgpu.h` is a copy for review and for the
//! link step; the test `the_committed_header_is_the_generated_one` checks the
//! two byte for byte. `FUTHARK_WGPU_SHIM_BLESS=1 cargo build` refreshes the
//! copy after a deliberate change to the surface.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("include").join("webgpu");
    std::fs::create_dir_all(&out).expect("create OUT_DIR/include/webgpu");
    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("read cbindgen.toml");
    let header = out.join("webgpu.h");
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen could not generate webgpu.h")
        .write_to_file(&header);
    if std::env::var("FUTHARK_WGPU_SHIM_BLESS").is_ok_and(|v| v == "1") {
        std::fs::copy(&header, crate_dir.join("include").join("webgpu").join("webgpu.h")).expect("bless the committed header");
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-env-changed=FUTHARK_WGPU_SHIM_BLESS");
}
