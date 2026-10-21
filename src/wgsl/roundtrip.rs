//! THE READER'S ROUND TRIP BY VALUE (2026-10-14, after nluca_s2): a kernel
//! the safe reader admits, rendered again from its tree, must compute what
//! the source computes. nluca_s2's Sonnet seeds staged their codebook with
//! `for (var i = lidx; i < 1024u; i += 128u)`; the reader read the loop from
//! 0, and 12 of 24 seeds became kernels in which every thread staged the same
//! eight entries. A text comparison cannot see such a drop (the texts always
//! differ: registers, rows, literals); a run can. [`same_values`] runs the
//! source and its rendering on the CPU executor (`cpu.rs`, the whole
//! dispatch, workgroup memory shared, wgpu's bounds policy) over the same
//! small, deterministic inputs, bound by binding number, and compares every
//! storage buffer bit for bit.

use std::collections::HashMap;

use naga::{AddressSpace, Module};

use crate::wgsl::cpu::{self, Val};
use crate::wgsl::loader::WgslKingdom;

/// The inputs of one small dispatch.
#[derive(Debug, Clone, Copy)]
pub struct Trial {
    pub workgroups: [u32; 3],
    /// The length of every runtime-sized storage array.
    pub elements: usize,
    pub seed: u64,
    /// The executor's cap on statements executed.
    pub max_steps: u64,
}

impl Default for Trial {
    fn default() -> Trial {
        Trial { workgroups: [2, 2, 1], elements: 64, seed: 1, max_steps: 200_000_000 }
    }
}

/// What the round trip found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundTrip {
    /// The rendering computes the source's values in every storage buffer.
    Same,
    /// The reader refuses the kernel (its message): nothing to compare.
    Refused(String),
    /// The CPU executor cannot run the source (its message): nothing to compare.
    NotRunnable(String),
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

/// Every scalar of a zero value replaced by a deterministic small one: f32
/// in [-1, 1) in steps of 1/64, u32 in [lo, 16), i32 in [-8, 8).
fn fill(z: &Val, state: &mut u64, lo: u32) -> Val {
    match z {
        Val::F(_) => Val::F((next(state) % 128) as f32 / 64.0 - 1.0),
        Val::U(_) => Val::U(lo + (next(state) % u64::from(16 - lo)) as u32),
        Val::I(_) => Val::I((next(state) % 16) as i32 - 8),
        Val::B(_) => Val::B(next(state) % 2 == 1),
        Val::V(v) => Val::V(v.iter().map(|x| fill(x, state, lo)).collect()),
        Val::M(c) => Val::M(c.iter().map(|col| col.iter().map(|_| (next(state) % 128) as f32 / 64.0 - 1.0).collect()).collect()),
        Val::P(p) => Val::P(p.clone()),
    }
}

/// A module's bindings by (group, binding): name, and whether it is storage.
fn bindings(module: &Module) -> Vec<((u32, u32), String, bool)> {
    let mut out: Vec<((u32, u32), String, bool)> = module.global_variables.iter().filter_map(|(_, g)| g.binding.as_ref().map(|b| ((b.group, b.binding), g.name.clone().unwrap_or_default(), matches!(g.space, AddressSpace::Storage { .. })))).collect();
    out.sort();
    out
}

/// The inputs of every binding of `module`, generated from the trial's seed
/// and the binding's number (so two modules that name a binding differently
/// get the same data at it): storage arrays of `elements` (or their fixed
/// length), the uniform's scalars at least 1 (a bound or a stride of 0 would
/// make the trial run nothing).
pub fn inputs(module: &Module, t: &Trial) -> Result<HashMap<String, Val>, String> {
    let mut out = HashMap::new();
    for (_, g) in module.global_variables.iter() {
        let Some(b) = &g.binding else { continue };
        let mut state = t.seed ^ (u64::from(b.group) << 40) ^ (u64::from(b.binding) << 20) ^ 0x9e37_79b9;
        let lo = if matches!(g.space, AddressSpace::Uniform) { 1 } else { 0 };
        let v = match &module.types[g.ty].inner {
            naga::TypeInner::Array { base, size: naga::ArraySize::Dynamic, .. } => {
                let z = cpu::zero(module, *base)?;
                Val::V((0..t.elements).map(|_| fill(&z, &mut state, lo)).collect())
            }
            _ => fill(&cpu::zero(module, g.ty)?, &mut state, lo),
        };
        out.insert(g.name.clone().unwrap_or_default(), v);
    }
    Ok(out)
}

/// A dispatch's storage buffers by (group, binding), as words.
fn storage_words(module: &Module, d: &cpu::Dispatch) -> Vec<((u32, u32), Vec<u32>)> {
    bindings(module)
        .into_iter()
        .filter(|(_, _, storage)| *storage)
        .map(|(k, name, _)| {
            let mut w = Vec::new();
            if let Some(v) = d.buffers.get(&name) {
                v.words(&mut w);
            }
            (k, w)
        })
        .collect()
}

/// THE ROUND TRIP BY VALUE of `source`: read by the safe reader, rendered
/// under the interface it was read under, both run on the CPU executor over
/// the trial's inputs; `Same` when every storage buffer holds the same bits
/// after both, an error naming the first buffer and element that differ.
pub fn same_values(source: &str, kingdom: &WgslKingdom, t: &Trial) -> Result<RoundTrip, String> {
    let read = match crate::wgsl::safe_reader::read_safe(source, kingdom) {
        Ok(r) => r,
        Err(e) => return Ok(RoundTrip::Refused(e)),
    };
    let rendered = crate::wgsl::safe_karva::render_tree_with(&read.tree, kingdom, &read.role)?.text;
    let a = crate::wgsl::naga_shape::parse(source)?;
    let b = crate::wgsl::naga_shape::parse(&rendered).map_err(|e| format!("the rendering does not parse: {e}\n{rendered}"))?;
    let (ka, kb) = (bindings(&a), bindings(&b));
    let (na, nb): (Vec<_>, Vec<_>) = (ka.iter().map(|x| (x.0, x.2)).collect(), kb.iter().map(|x| (x.0, x.2)).collect());
    if na != nb {
        return Err(format!("the rendering binds {nb:?}, the source {na:?}"));
    }
    let ia = inputs(&a, t)?;
    let ib: HashMap<String, Val> = {
        // The same data at the same binding numbers, by the rendering's names.
        let by_key: HashMap<(u32, u32), &String> = ka.iter().map(|(k, n, _)| (*k, n)).collect();
        kb.iter().map(|(k, n, _)| (n.clone(), ia[by_key[k]].clone())).collect()
    };
    let ra = match cpu::run(&a, ia, t.workgroups, t.max_steps) {
        Ok(d) => d,
        Err(e) => return Ok(RoundTrip::NotRunnable(e)),
    };
    let rb = cpu::run(&b, ib, t.workgroups, t.max_steps).map_err(|e| format!("the rendering does not run where the source does: {e}"))?;
    for ((k, wa), (_, wb)) in storage_words(&a, &ra).iter().zip(storage_words(&b, &rb).iter()) {
        if wa != wb {
            let at = wa.iter().zip(wb.iter()).position(|(x, y)| x != y).unwrap_or(wa.len().min(wb.len()));
            return Err(format!("binding {k:?} differs at word {at} of {}: source {:?}, rendering {:?} (f32 {} against {})", wa.len(), wa.get(at), wb.get(at), wa.get(at).map_or(f32::NAN, |w| f32::from_bits(*w)), wb.get(at).map_or(f32::NAN, |w| f32::from_bits(*w))));
        }
    }
    Ok(RoundTrip::Same)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn wgsl_in(dir: &Path, ext: &str) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(ext)).collect();
        v.sort();
        v
    }

    /// A model reply's kernel: its fenced block, else the whole text.
    fn reply_kernel(text: &str) -> String {
        match text.find("```") {
            Some(s) => {
                let after = &text[s + 3..];
                let body = &after[after.find('\n').map_or(0, |i| i + 1)..];
                body[..body.find("```").unwrap_or(body.len())].to_string()
            }
            None => text.to_string(),
        }
    }

    /// The thread-strided loop of nluca_s2's seed01 is never read from 0:
    /// before 24a209e it was refused by name; since, it reads as the guarded
    /// loop `for (k = 0; k < n; k += s) { if (e + k < n) { body(e + k) } }`
    /// (the grid-stride reading), and its rendering computes the source's
    /// values, as the same staging written from 0 with the thread's offset does.
    #[test]
    fn a_loop_entered_at_a_threads_index_reads_with_its_own_values() {
        let k = WgslKingdom::load_safe();
        let kernel = |stage: &str| {
            format!(
                "@group(0) @binding(0) var<storage, read> src: array<f32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<f32>;\nvar<workgroup> cb: array<f32, 256>;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(local_invocation_index) li: u32, @builtin(global_invocation_id) gid: vec3<u32>) {{\n{stage}\n    workgroupBarrier();\n    dst[gid.x] = cb[(gid.x * 7u) & 255u];\n}}\n"
            )
        };
        let strided = kernel("    for (var i: u32 = li; i < 256u; i += 64u) {\n        cb[i] = src[i];\n    }");
        assert_eq!(same_values(&strided, &k, &Trial { elements: 512, ..Trial::default() }).expect("the round trip"), RoundTrip::Same);
        let from_zero = kernel("    for (var i: u32 = 0u; i < 256u; i += 64u) {\n        let j = i + li;\n        if (j < 256u) {\n            cb[j] = src[j];\n        }\n    }");
        assert_eq!(same_values(&from_zero, &k, &Trial { elements: 512, ..Trial::default() }).expect("the round trip"), RoundTrip::Same);
    }

    /// EVERY KERNEL THE READER ADMITS COMPUTES ITS SOURCE'S VALUES WHEN
    /// RENDERED AGAIN, on the CPU executor over a small dispatch: the safeWGSL
    /// corpus, the interfaces corpus, nluca's six CubeCL kernels, the saved
    /// populations and the library, and nluca_s3's Sonnet seeds and replies.
    /// Counts printed: compared, refused, not runnable on the CPU.
    #[test]
    fn every_admitted_kernel_renders_to_the_same_values() {
        let k = WgslKingdom::load_safe();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut sources: Vec<(String, String)> = Vec::new();
        // Tracked directories, each with the number of kernels it holds (a directory that shrank fails here).
        for (d, expected) in [("tests/corpus/safewgsl", 112usize), ("tests/corpus/interfaces", 2), ("tests/corpus/nluca", 6), ("logs/HGP/001/gen0", 31), ("tests/fixtures/wgsl_genomes/popA2_g006j/texts", 20), ("tests/fixtures/wgsl_genomes/library_ep/texts", 441)] {
            let files = wgsl_in(&root.join(d), ".wgsl");
            assert_eq!(files.len(), expected, "{d}: {} kernels, expected {expected}", files.len());
            for f in files {
                sources.push((f.display().to_string(), std::fs::read_to_string(&f).expect("read")));
            }
        }
        // The saved populations the logs held, tracked as fixtures (T63).
        for set in ["popA_g001", "popA2_g001b", "library_texts", "popB_g080", "reserve2", "popC_rand20", "popR_rules", "nluca_s3_seeds", "opus_cross"] {
            sources.extend(crate::wgsl::fixture_texts::set(set).into_iter().map(|(name, text)| (format!("{set}/{name}"), text)));
        }
        sources.extend(crate::wgsl::fixture_texts::set("nluca_s3_replies").into_iter().map(|(name, text)| (format!("nluca_s3_replies/{name}"), reply_kernel(&text))));
        assert!(sources.len() > 100, "the corpora are missing: {} kernels", sources.len());
        let (mut same, mut refused, mut not_runnable) = (0usize, 0usize, 0usize);
        let mut failed: Vec<String> = Vec::new();
        // The kernels in parallel (one CPU dispatch each, two runs), the
        // results in the corpora's order.
        use rayon::prelude::*;
        let results: Vec<Result<RoundTrip, String>> = sources.par_iter().map(|(_, src)| same_values(src, &k, &Trial::default())).collect();
        for ((name, _), result) in sources.iter().zip(results) {
            match result {
                Ok(RoundTrip::Same) => same += 1,
                Ok(RoundTrip::Refused(_)) => refused += 1,
                Ok(RoundTrip::NotRunnable(e)) => {
                    not_runnable += 1;
                    println!("not runnable on the CPU: {name}: {}", e.chars().take(120).collect::<String>());
                }
                Err(e) => failed.push(format!("{name}: {}", e.chars().take(400).collect::<String>())),
            }
        }
        println!("{} kernels: {same} render to the same values, {refused} refused by the reader, {not_runnable} not runnable on the CPU, {} differ", sources.len(), failed.len());
        assert!(failed.is_empty(), "{} kernels render to other values:\n{}", failed.len(), failed.join("\n"));
    }

    /// THE READER GAPS CLOSED (T43, 2026-10-21), one cause at a time: each kernel below was refused by the safe reader
    /// (`examples/wgsl_reader_census`) and is now read, rebuilt by the renderer, and run on the CPU executor beside its
    /// source with the same values in every storage buffer. Each cause's kernels are listed with the cause.
    const CLOSED: [(&str, &[&str]); 3] = [
        // F: the thread's row and column from the linear index (`lidx & 15u`, `lidx >> 4u`)
        ("F lidx axes", &["tests/corpus/model_replies/k0017.wgsl", "tests/corpus/model_replies/k0038.wgsl"]),
        // F: a private array holding two axes (`pi[k]` the rows, `pi[k + 8u]` the columns)
        ("F two axes in one array", &["tests/corpus/model_replies/k0443.wgsl"]),
        // I: a store to one lane of a vec4<u32> local
        ("I vec4<u32> lane store", &["tests/corpus/model_replies/k1765.wgsl", "tests/corpus/model_replies/k1767.wgsl", "tests/corpus/model_replies/k1771.wgsl"]),
    ];

    #[test]
    fn the_kernels_of_the_closed_reader_gaps_read_and_render_to_their_own_values() {
        let k = WgslKingdom::load_safe();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut failed = Vec::new();
        for (cause, files) in CLOSED {
            for f in files {
                let text = std::fs::read_to_string(root.join(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
                match same_values(&text, &k, &Trial::default()) {
                    Ok(RoundTrip::Same) => {}
                    other => failed.push(format!("{cause}: {f}: {:?}", other.map_err(|e| e.chars().take(300).collect::<String>()).map(|r| format!("{r:?}").chars().take(300).collect::<String>()))),
                }
            }
        }
        assert!(failed.is_empty(), "{} kernels:\n{}", failed.len(), failed.join("\n"));
    }
}
