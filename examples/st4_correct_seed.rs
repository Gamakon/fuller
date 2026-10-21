//! ST4 follow-up: the shipped Gram kernel as a gene of `kingdoms/wgsl_true`.
//!
//!   cargo run --release --features gpu,wgsl --example st4_correct_seed -- --seeds SEEDS.tsv --out OUT.tsv
//!
//! Builds the shipped kernel's tree from rows (`true_gram`), writes it into a gene with the section 5 write-back (the
//! unread rows keep a typed-initialiser draw), expresses the gene and checks it is the same tree, renders, validates with
//! naga, compiles on the Metal pipeline and runs it under the work guard on the three Gram inputs, reporting the three
//! tolerance ratios and the device time against the shipped kernel's. With `--seeds` the gene is appended to those genes
//! and written to `--out` (one row of words a line).

use std::io::Write;

use fuller::typed_gep::{express, write_back, GeneLayout};
use fuller::wgsl::gram::{check, effective_buffer_limit, phi_of, reference, Gram, Kind, SHIPPED};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_gram::shipped_gram_tree;
use fuller::wgsl::true_kingdom::{default_dir, random_gene, Rng, Share, TrueKingdom};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = arg(&args, "--n").map_or(Ok(8), |v| v.parse().map_err(|e| format!("--n: {e}")))?;
    let kingdom = TrueKingdom::load(&default_dir())?;
    let contract = kingdom.relations().check().map_err(|d| format!("the kingdom check failed: {:?}", &d[..d.len().min(3)]))?;
    let layout = GeneLayout::from_program_size(&contract, kingdom.root, n).map_err(|e| e.to_string())?;
    let const_len = kingdom.const_lens();
    let mut shape = "16 16";
    let tree = match shipped_gram_tree(&kingdom, shape) {
        Ok(t) => t,
        Err(e) => {
            println!("# {e}");
            shape = "8 8";
            println!("# the nearest workgroup in the table is {shape:?}: one cell a thread as the shipped kernel, 64 threads a workgroup instead of 256");
            shipped_gram_tree(&kingdom, shape)?
        }
    };
    println!("# tree: {} nodes", tree.level_order().len());
    let mut rng = Rng(0x57_4C);
    let previous = random_gene(&contract, &layout, &const_len, &mut rng, Share::Half, true);
    let gene = write_back(&contract, &layout, &previous, &tree).map_err(|e| format!("write-back: {e}"))?;
    let expressed = express(&contract, &layout, &gene, &const_len).map_err(|e| format!("express: {e}"))?;
    if expressed.tree() != tree {
        return Err("our software has a bug: the expressed tree is not the tree written".into());
    }
    println!("# write-back and express agree: the gene expresses the same {}-node tree", expressed.nodes.len());
    let src = kingdom.render(&tree)?;
    std::fs::write("/private/tmp/claude-501/st4_correct_seed.wgsl", &src).map_err(|e| e.to_string())?;
    let module = fuller::wgsl::naga_shape::parse_unscanned(&src).map_err(|e| format!("naga: {e}"))?;
    fuller::wgsl::target::Target::current().validate(&module, &src).map_err(|e| format!("naga validate: {e}"))?;
    let dev = Device::new()?;
    let limit = effective_buffer_limit(&dev.device);
    let fuel = kingdom.fuel(&contract, n)?;
    let gram = Gram::compile(&dev, &src, fuel).map_err(|e| format!("pipeline: {e}"))?;
    let shipped = Gram::compile(&dev, SHIPPED, Fuel::Proven("the shipped Gram kernel"))?;
    let (d, m) = (16usize, 4usize);
    let mut ratios = Vec::new();
    for (k, s) in [(Kind::Gaussian, 1u64), (Kind::Weighted, 2), (Kind::HeavyTailed, 3)] {
        let p = phi_of(k, d, m, s);
        let r = reference(&p, d, m);
        let b = gram.build(&p, d, m, limit).map_err(|e| format!("run: {e}"))?;
        let v = check(&b.h, &r, d, m);
        let bs = shipped.build(&p, d, m, limit)?;
        let vs = check(&bs.h, &r, d, m);
        println!("# {k:?}: this gene mean tolerance ratio {:.4} (non-finite {}), the shipped text {:.4}; device ms {:.4} against {:.4}", v.mean_ratio, v.non_finite, vs.mean_ratio, b.gpu_ms, bs.gpu_ms);
        ratios.push(v.mean_ratio);
    }
    if let (Some(seeds), Some(out)) = (arg(&args, "--seeds"), arg(&args, "--out")) {
        let mut text = std::fs::read_to_string(&seeds).map_err(|e| format!("{seeds}: {e}"))?;
        text.push_str(&gene.iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
        text.push('\n');
        std::fs::File::create(&out).and_then(|mut f| f.write_all(text.as_bytes())).map_err(|e| e.to_string())?;
        println!("# wrote {} genes to {out} (the correct gene last)", text.lines().count());
    }
    println!("# ratios {ratios:?}");
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
