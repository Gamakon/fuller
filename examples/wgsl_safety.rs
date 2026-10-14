//! The general safety layer (`src/wgsl/safety.rs`) over kernel texts: for
//! each file, the verdict and what it was proved over. With `--safe` the text
//! is first read into the safeWGSL grammar and rendered (what check.rs's
//! layer 7 proves); `--print` writes the text that was proved; `--seed <n>`
//! instead draws random sampler gene n (Gram role) and proves its rendering.
//! The contract is the Gram problem's (`--contract <file>` for another).
//!
//!   cargo run --release --features wgsl --example wgsl_safety -- [--safe] [--print] [--contract <file>] <file.wgsl>... | --seed <n>

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::{gram_role, sample, Renderer};
use fuller::wgsl::safe_reader::read_safe;
use fuller::wgsl::safety::{check_text, gram_contract_for, Contract};
use fuller::wgsl::table::FunctionTable;

fn main() -> Result<(), String> {
    let mut safe = false;
    let mut print = false;
    let mut contract: Option<Contract> = None;
    let mut files = Vec::new();
    let mut seeds = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--safe" => safe = true,
            "--print" => print = true,
            "--contract" => contract = Some(Contract::parse(&std::fs::read_to_string(args.next().ok_or("--contract needs a file")?).map_err(|e| e.to_string())?)?),
            "--seed" => seeds.push(args.next().ok_or("--seed needs a number")?.parse::<u64>().map_err(|e| e.to_string())?),
            _ => files.push(a),
        }
    }
    let kingdom = WgslKingdom::load_safe();
    let table = FunctionTable::safe();
    let role = gram_role();
    let renderer = Renderer::new(&kingdom, &table, &role);
    let mut texts: Vec<(String, String)> = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        if safe {
            let r = read_safe(&src, &kingdom).map_err(|e| format!("{f}: does not read: {e}"))?;
            let text = Renderer::new(&kingdom, &table, &r.role).kernel(&r.tree)?.text;
            texts.push((f.clone(), text));
        } else {
            texts.push((f.clone(), src));
        }
    }
    for s in seeds {
        let root = fuller::wgsl::render::dual_ty(&kingdom, "kernel.safe")?;
        let tree = sample(&kingdom, &role, root, 7, s)?;
        texts.push((format!("seed {s}"), renderer.kernel(&tree)?.text));
    }
    for (name, text) in texts {
        if print {
            println!("{text}");
        }
        let c = contract.clone().unwrap_or_else(|| gram_contract_for(&text));
        match check_text(&text, &c) {
            Ok(r) => println!("{name}: proved ({} write sites, {} output reads, {} workgroup accesses, {} pairs)", r.write_sites, r.read_sites, r.workgroup_sites, r.pairs),
            Err(e) => println!("{name}: {e}"),
        }
    }
    Ok(())
}
