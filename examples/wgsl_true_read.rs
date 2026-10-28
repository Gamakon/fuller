//! A kernel of a harness problem read exactly into the true WGSL kingdom (`fuller::wgsl::true_reader`), put through
//! its genes (the per-type write-back of `fuller::wgsl::true_module`), and the expressed text run on the device
//! against the kernel's own text on every constraint case of the problem (`fuller::wgsl::case_set`): bit for bit, case
//! by case.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_true_read -- --problem problems/kmeans_update \
//!       --kingdom kingdoms/wgsl_true/mod/kmeans_update [--fields x,y,z,w] [--out DIR] <kernel.wgsl>...
//!
//! `--read-only`: read and write back only, no device. `--out DIR`: each read kernel's expressed text and its genes
//! (one line of words a gene) saved as `DIR/<kernel>.expressed.wgsl` and `DIR/<kernel>.genes`, the start of a P run.
//! Every dispatch of a read kernel, its own text included (the reading is exact: the same loops), is bounded before it
//! is submitted by the work guard on the read tree (`TrueKingdom::tree_work_bound_workgroup_in` with the dispatch's own
//! uniform words, `DeviceExecutor::with_dispatch_bound`). One GPU job at a time, under the GPU lock.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use fuller::typed_gep::Tree;
use fuller::wgsl::case_set::{CaseSet, Split};
use fuller::wgsl::harness::{DeviceExecutor, Problem};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::{Rng, Share, TrueKingdom};
use fuller::wgsl::true_module::TrueModule;
use fuller::wgsl::true_reader::{read_module, Contract};

fn stamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

fn functions(t: &Tree, k: &TrueKingdom) -> usize {
    usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Option<String> { args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned() };
    let problem_dir = PathBuf::from(value("--problem").ok_or("--problem DIR")?);
    let kingdom_dir = PathBuf::from(value("--kingdom").ok_or("--kingdom DIR")?);
    let fields: Vec<String> = value("--fields").unwrap_or_else(|| "x,y,z,w".into()).split(',').map(str::to_string).collect();
    let out_dir = value("--out").map(PathBuf::from);
    let read_only = args.iter().any(|a| a == "--read-only");
    let mut kernels = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--problem" | "--kingdom" | "--fields" | "--out" => i += 2,
            "--read-only" => i += 1,
            a => {
                kernels.push(PathBuf::from(a));
                i += 1;
            }
        }
    }
    let k = Arc::new(TrueKingdom::load(&kingdom_dir)?);
    let field_refs: Vec<&str> = fields.iter().map(String::as_str).collect();
    let contract = Contract::parse(&std::fs::read_to_string(problem_dir.join("safety.contract")).map_err(|e| format!("safety.contract: {e}"))?, &field_refs)?;
    let dev = if read_only { None } else { Some(Device::new()?) };
    println!("# {} {} kernels against {}", stamp(), kernels.len(), problem_dir.display());
    for path in &kernels {
        let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let trees = match read_module(&k, &text, &contract) {
            Ok(t) => t,
            Err(e) => {
                println!("{name}\tnot read\t{e}");
                continue;
            }
        };
        let sizes: Vec<usize> = trees.iter().map(|t| functions(t, &k)).collect();
        let m = TrueModule::new(&k, &sizes)?;
        let prev = m.random(1, &mut Rng(5), Share::Half, true)?;
        let genes = m.encode(&trees, &prev)?;
        if m.entry_trees(&genes)? != trees {
            println!("{name}\tbug\tour software has a bug: the genes express another tree");
            continue;
        }
        let rendered = m.render(&genes)?;
        println!("{name}\tread\t{} entries, {} functions, {} nodes; expressed text {} bytes", trees.len(), sizes.iter().sum::<usize>(), trees.iter().map(|t| t.level_order().len()).sum::<usize>(), rendered.len());
        if let Some(d) = &out_dir {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
            std::fs::write(d.join(format!("{name}.expressed.wgsl")), &rendered).map_err(|e| e.to_string())?;
            let words: Vec<String> = genes.genes.iter().map(|g| g.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")).collect();
            std::fs::write(d.join(format!("{name}.genes")), words.join("\n") + "\n").map_err(|e| e.to_string())?;
        }
        let Some(dev) = &dev else { continue };
        let tb: Rc<RefCell<Vec<Tree>>> = Rc::new(RefCell::new(trees.clone()));
        let (kb, tc, fb) = (k.clone(), tb.clone(), fields.clone());
        let bound = move |entry: usize, words: &[u8], invocations: u64| -> Result<u128, String> {
            let w: Vec<u64> = words.chunks_exact(4).map(|c| u64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
            let f: BTreeMap<String, u64> = fb.iter().zip(&w).map(|(f, v)| (f.clone(), *v)).collect();
            let t = tc.borrow();
            let tree = t.get(entry).ok_or_else(|| format!("no tree for entry {entry}"))?;
            kb.tree_work_bound_workgroup_in(tree, &f, invocations)
        };
        let problem = Problem::load(&problem_dir)?;
        let exec = DeviceExecutor::for_problem(dev, &problem, Fuel::Proven("unused: every dispatch is bounded by with_dispatch_bound")).with_dispatch_bound(Box::new(bound));
        let set = CaseSet::new(&problem_dir, exec)?;
        let mut outs = Vec::new();
        for src in [&text, &rendered] {
            let (kern, info) = set.compile(src)?;
            outs.push(set.judge(&kern, &info)?);
        }
        let mut all = true;
        for (a, b) in outs[0].iter().zip(&outs[1]) {
            let same = a.out.iter().zip(&b.out).all(|(x, y)| x.to_bits() == y.to_bits());
            all &= same;
            let split = if a.split == Split::Held { "held" } else { "train" };
            println!("{name}\t{}\t{split}\ttext norm {:.3e} differ {}\texpressed norm {:.3e} differ {}\t{}", a.name, a.norm, a.differ, b.norm, b.differ, if same { "bit for bit" } else { "VALUES DIFFER" });
        }
        println!("{name}\t{}", if all { "the expressed text equals the kernel's own on every case, bit for bit" } else { "THE EXPRESSED TEXT DIFFERS" });
    }
    println!("# {} done", stamp());
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
