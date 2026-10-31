//! The rings' checks: a pool line builds back to its tree at its type; a program's ring sequences are exactly the
//! coding cells `write_back` writes, type by type in frame order; a ring the programs never reach is filled from its
//! class, or from its slot family with the class named as a gap; a canvas gene is a valid gene of the layout, and a
//! program written onto it expresses that program.

use super::*;
use crate::futhark_sig::{build, load, problem, random_gene, sexp};
use crate::typed_gep::{express, validate, write_back, Domain};

fn functions(k: &FutharkKingdom, t: &Tree) -> usize {
    usize::from(!k.rows[t.symbol as usize].inputs.is_empty()) + t.kids.iter().map(|c| functions(k, c)).sum::<usize>()
}

fn kingdom(name: &str) -> (FutharkKingdom, KingdomContract, GeneLayout, Tree) {
    let p = problem(name).unwrap();
    let k = load(&p.instance).unwrap();
    let contract = k.relations().check().unwrap();
    let reference = build(&k, p.reference).unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], functions(&k, &reference)).unwrap();
    (k, contract, layout, reference)
}

#[test]
fn a_pool_line_builds_back_to_its_tree_at_its_type() {
    for name in ["histogram", "pathfinder"] {
        let (k, contract, layout, reference) = kingdom(name);
        let const_len = k.const_len();
        let mut rng = GeneRng::new(7);
        let mut trees = vec![reference.clone()];
        for _ in 0..200 {
            let g = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
            let t = express(&contract, &layout, &g, &const_len).unwrap().tree();
            // Every subtree, at its own type: the pool holds programs of any type.
            for (s, _) in t.level_order() {
                trees.push(s.clone());
            }
        }
        for t in &trees {
            let dual = &k.dual(contract.output(t.symbol)).unwrap().dual;
            assert_eq!(&build_at(&k, &sexp(&k, t), dual).unwrap(), t, "{name}: {}", sexp(&k, t));
        }
    }
}

#[test]
fn the_ring_sequences_are_the_coding_cells_write_back_writes() {
    for name in ["histogram", "pathfinder"] {
        let (k, contract, layout, reference) = kingdom(name);
        let const_len = k.const_len();
        let mut rng = GeneRng::new(11);
        let mut trees = vec![reference];
        for _ in 0..100 {
            let g = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
            trees.push(express(&contract, &layout, &g, &const_len).unwrap().tree());
        }
        for t in &trees {
            let mut rings = Rings::default();
            rings.add(&contract, t);
            // The cells write_back writes onto an empty gene: those two different fills agree on.
            let a = write_back(&contract, &layout, &vec![0u32; layout.width], t).unwrap();
            let b = write_back(&contract, &layout, &vec![u32::MAX; layout.width], t).unwrap();
            for f in &layout.frames {
                let rows: Vec<u32> = (0..f.head_len + f.tail_len).map(|i| f.row_pos(i).unwrap()).filter(|&p| a[p] == b[p]).map(|p| a[p]).collect();
                assert_eq!(rows.as_slice(), rings.get(f.ty, Ring::Head), "{name}: the head ring of {:?}", k.dual(f.ty).unwrap().dual);
                let leaves: Vec<u32> = rows.iter().copied().filter(|s| contract.is_terminal(*s)).collect();
                assert_eq!(leaves.as_slice(), rings.get(f.ty, Ring::Terminal));
                let dc: Vec<u32> = f.dc_off.map(|o| (o..o + f.dc_len).filter(|&p| a[p] == b[p]).map(|p| a[p]).collect()).unwrap_or_default();
                assert_eq!(dc.as_slice(), rings.get(f.ty, Ring::Constant));
            }
        }
    }
}

/// Rings from random programs (a unit test of the mechanics; the real pool is read from programs).
fn random_rings(k: &FutharkKingdom, contract: &KingdomContract, layout: &GeneLayout, n: usize, seed: u64) -> Rings {
    let const_len = k.const_len();
    let mut rng = GeneRng::new(seed);
    let mut rings = Rings::default();
    for _ in 0..n {
        let g = random_gene(contract, layout, &const_len, &mut rng).unwrap();
        rings.add(contract, &express(contract, layout, &g, &const_len).unwrap().tree());
    }
    rings
}

#[test]
fn empty_rings_are_filled_from_their_class_and_a_missing_class_is_named() {
    for (name, gap_class) in [("histogram", "ix.k"), ("pathfinder", "a.rows")] {
        let (k, contract, layout, _) = kingdom(name);
        let class_of = |ty: &Ty| k.dual(*ty).unwrap().dual.split(':').next().unwrap().to_string();
        let mut rings = random_rings(&k, &contract, &layout, 300, 5);
        // Half the scopes of every class emptied: those are filled from the class.
        // (A type outside every scope, pathfinder's `a.rows.cols` input, has no other scope and is kept.)
        rings.rings.retain(|(ty, _), _| class_of(ty) == gap_class || !k.dual(*ty).unwrap().dual.contains(':') || matches!(ty, Ty::FutharkShape(i) if i % 2 == 0));
        // The whole of one class emptied: filled from its slot family, and named a class gap.
        rings.rings.retain(|(ty, _), _| class_of(ty) != gap_class);
        let (full, fills) = rings.complete(&k, &contract, &layout).unwrap();
        assert!(fills.iter().any(|f| f.from == "class"), "{name}: no ring filled from its class");
        let gaps: Vec<&Fill> = fills.iter().filter(|f| f.from != "class").collect();
        assert!(!gaps.is_empty() && gaps.iter().all(|f| f.dual.starts_with(&format!("{gap_class}:"))), "{name}: {:?}", gaps.iter().map(|f| &f.dual).collect::<Vec<_>>());
        // Every filled symbol is a legal row of its own type, and a terminal in a terminal ring.
        for ((ty, ring), syms) in &full.rings {
            for s in syms {
                match ring {
                    Ring::Constant => assert!(*s < k.const_len()[ty]),
                    Ring::Terminal => assert!(contract.output(*s) == *ty && contract.is_terminal(*s)),
                    Ring::Head => assert_eq!(contract.output(*s), *ty),
                }
            }
        }
        // A slot family with nothing recorded at all is an error that names the rings, never a random fill.
        let slot = k.dual(full.rings.keys().find(|(t, _)| class_of(t) == gap_class).unwrap().0).unwrap().slot.clone();
        rings.rings.retain(|(ty, _), _| k.dual(*ty).unwrap().slot != slot);
        let e = rings.complete(&k, &contract, &layout).expect_err("an empty slot family");
        assert!(e.contains("a kingdom gap") && e.contains(&format!("{gap_class}:")), "{e}");
    }
}

#[test]
fn a_canvas_is_a_valid_gene_and_a_program_written_onto_it_expresses_that_program() {
    for name in ["histogram", "pathfinder"] {
        let (k, contract, layout, reference) = kingdom(name);
        let const_len = k.const_len();
        let mut rings = random_rings(&k, &contract, &layout, 300, 9);
        rings.add(&contract, &reference);
        let (full, _) = rings.complete(&k, &contract, &layout).unwrap();
        let mut rng = GeneRng::new(3);
        for _ in 0..200 {
            let g = full.canvas(&contract, &layout, &mut rng).unwrap();
            validate(&contract, &layout, &g, &const_len).unwrap();
            let root = layout.frame(layout.root).unwrap().row_pos(0).unwrap();
            assert!(!contract.is_terminal(g[root]), "{name}: the root cell is a function");
            express(&contract, &layout, &g, &const_len).unwrap();
            let w = write_back(&contract, &layout, &g, &reference).unwrap();
            assert_eq!(express(&contract, &layout, &w, &const_len).unwrap().tree(), reference);
        }
        // Every cell of a canvas comes from its type's ring of its domain.
        let g = full.canvas(&contract, &layout, &mut rng).unwrap();
        for s in layout.spans() {
            let ring = match s.domain {
                Domain::Head => Ring::Head,
                Domain::Tail => Ring::Terminal,
                Domain::Dc => Ring::Constant,
            };
            assert!(g[s.off..s.off + s.len].iter().all(|c| full.get(s.ty, ring).contains(c)));
        }
    }
}

#[test]
fn a_ring_walk_shifts_one_window_along_its_ring_and_keeps_the_gene_valid() {
    for name in ["histogram", "pathfinder"] {
        let (k, contract, layout, reference) = kingdom(name);
        let const_len = k.const_len();
        let mut rings = random_rings(&k, &contract, &layout, 300, 13);
        rings.add(&contract, &reference);
        let (full, _) = rings.complete(&k, &contract, &layout).unwrap();
        let mut rng = GeneRng::new(21);
        for _ in 0..300 {
            let mut g = write_back(&contract, &layout, &full.canvas(&contract, &layout, &mut rng).unwrap(), &reference).unwrap();
            let before = g.clone();
            let coding = coding_cells(&contract, &layout, &g, &const_len).unwrap();
            let w = full.walk(&layout, &mut g, &coding, 4, &mut rng).unwrap().expect("the reference has coding cells");
            validate(&contract, &layout, &g, &const_len).unwrap();
            let s = layout.spans().into_iter().find(|s| s.ty == w.ty && s.domain == w.domain).unwrap();
            assert!(coding.iter().any(|&p| p >= s.off && p < s.off + s.len), "the walk moved a domain with no coding cell");
            // Inside the window: shifted by k; outside it: unchanged.
            assert_eq!(&g[s.off..s.off + s.len - w.k], &before[s.off + w.k..s.off + s.len]);
            for p in (0..layout.width).filter(|&p| p < s.off || p >= s.off + s.len) {
                assert_eq!(g[p], before[p]);
            }
        }
    }
}

#[test]
fn a_growth_mutation_gives_one_head_cell_more_arguments_whatever_the_padding() {
    let (k, contract, layout, reference) = kingdom("histogram");
    let const_len = k.const_len();
    let mut rings = random_rings(&k, &contract, &layout, 300, 17);
    rings.add(&contract, &reference);
    let (full, _) = rings.complete(&k, &contract, &layout).unwrap();
    let mut pad = GeneRng::new(1);
    for seed in 0..100 {
        let mut a = write_back(&contract, &layout, &full.canvas(&contract, &layout, &mut pad).unwrap(), &reference).unwrap();
        let mut b = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut pad).unwrap(), &reference).unwrap();
        let ca = coding_cells(&contract, &layout, &a, &const_len).unwrap();
        let cb = coding_cells(&contract, &layout, &b, &const_len).unwrap();
        assert_eq!(ca, cb, "the coding cells of one program do not depend on its padding");
        let ga = grow(&contract, &layout, &mut a, &ca, &const_len, &mut GeneRng::new(seed)).unwrap();
        let gb = grow(&contract, &layout, &mut b, &cb, &const_len, &mut GeneRng::new(seed)).unwrap();
        assert_eq!(ga, gb, "the same draws make the same edit on both paddings");
        assert!(contract.signature(ga.2).len() > contract.signature(ga.1).len());
        validate(&contract, &layout, &a, &const_len).unwrap();
        express(&contract, &layout, &a, &const_len).unwrap();
    }
}

#[test]
fn the_committed_pools_fill_every_ring_of_the_default_layout() {
    for name in ["histogram", "pathfinder"] {
        let (k, contract, layout, _) = kingdom(name);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("kingdoms/futhark_shapes/sig/{name}/pool"));
        let excluded = published_source(name).unwrap();
        let mut rings = Rings::default();
        for f in ["bench.tsv", "llm.tsv"] {
            for e in load_pool(&dir.join(f)).unwrap() {
                if e.origin != excluded {
                    rings.add(&contract, &build_at(&k, &e.sexp, &e.dual).unwrap());
                }
            }
        }
        let (_, fills) = rings.complete(&k, &contract, &layout).unwrap_or_else(|e| panic!("{name}: {}", &e[..e.len().min(600)]));
        eprintln!("{name}: {} programs; {} rings filled from their class, {} from their slot family", rings.programs, fills.iter().filter(|f| f.from == "class").count(), fills.iter().filter(|f| f.from != "class").count());
    }
}

/// The seed of `--gen0 fable-mutants` from the committed pool, on a canvas of the committed rings.
fn fable_setup(name: &str) -> (FutharkKingdom, KingdomContract, GeneLayout, Rings, Tree) {
    let p = problem(name).unwrap();
    let k = load(&p.instance).unwrap();
    let contract = k.relations().check().unwrap();
    let n = if name == "histogram" { 8 } else { 13 };
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], n).unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("kingdoms/futhark_shapes/sig/{name}/pool"));
    let mut rings = Rings::default();
    for f in ["bench.tsv", "llm.tsv"] {
        for e in load_pool(&dir.join(f)).unwrap() {
            if e.origin != published_source(name).unwrap() {
                rings.add(&contract, &build_at(&k, &e.sexp, &e.dual).unwrap());
            }
        }
    }
    let (full, _) = rings.complete(&k, &contract, &layout).unwrap();
    let seed = load_pool(&dir.join("seed.tsv")).unwrap().remove(0);
    let t = build_at(&k, &seed.sexp, &seed.dual).unwrap();
    (k, contract, layout, full, t)
}

fn grow_population(name: &str, walk: bool, seed: u64) -> Mutants {
    let (k, contract, layout, rings, t) = fable_setup(name);
    let const_len = k.const_len();
    let mut rng = GeneRng::new(seed);
    let g0 = write_back(&contract, &layout, &rings.canvas(&contract, &layout, &mut rng).unwrap(), &t).unwrap();
    let spec = MutantSpec { contract: &contract, layout: &layout, const_len: &const_len, pop: 24, max_tries: 24 * 200 };
    let mut mutate = |g: &mut Vec<u32>, r: &mut GeneRng| -> Result<(), String> {
        let coding = coding_cells(&contract, &layout, g, &const_len)?;
        if walk {
            rings.walk(&layout, g, &coding, 1 + r.below(4), r)?;
        } else {
            // One point mutation of a coding cell, as `futhark_evolve`'s `Genetics::mutate`.
            let pos = coding[r.below(coding.len())];
            let a = layout.alphabet(&contract, pos, &const_len);
            g[pos] = a[r.below(a.len())];
        }
        Ok(())
    };
    let render = |t: &Tree| -> Result<String, String> { Ok(sexp(&k, t)) };
    let m = mutants(&spec, g0, &mut mutate, &render, &mut rng).unwrap();
    for g in &m.genes {
        validate(&contract, &layout, g, &const_len).unwrap();
    }
    assert_eq!(m.trees[0], t, "{name}: member 1 is the seed program");
    m
}

#[test]
fn generation_0_from_one_seed_is_24_distinct_legal_members_and_deterministic() {
    for name in ["histogram", "pathfinder"] {
        for walk in [true, false] {
            let m = grow_population(name, walk, 77);
            assert_eq!(m.genes.len(), 24);
            for i in 0..24 {
                for j in 0..i {
                    assert_ne!(m.texts[i], m.texts[j], "{name}: members {i} and {j} have one text");
                    assert_ne!(m.trees[i], m.trees[j]);
                }
            }
            assert_eq!(m, grow_population(name, walk, 77), "{name}: the same seed gives the same population");
            assert_ne!(m.genes, grow_population(name, walk, 78).genes, "{name}: another seed gives another population");
            eprintln!("{name} ({}): 24 members from {} mutants tried", if walk { "ring walk" } else { "point" }, m.tried);
        }
    }
}

/// Generation 0 from several starts: the starts enter first and in order, every mutant descends from the first start
/// (never from a later one), no two members share a text or a tree, and a start that repeats another is an error.
/// With one start it is `mutants` draw for draw. The mutation is `point_mutate`, which keeps every gene legal.
#[test]
fn generation_0_from_several_starts_grows_from_the_first_only() {
    let (k, contract, layout, rings, t) = fable_setup("histogram");
    let const_len = k.const_len();
    let mut rng = GeneRng::new(5);
    let first = write_back(&contract, &layout, &rings.canvas(&contract, &layout, &mut rng).unwrap(), &t).unwrap();
    let mut second = first.clone();
    point_mutate(&contract, &layout, &const_len, &mut second, 3, &mut rng).unwrap();
    validate(&contract, &layout, &second, &const_len).unwrap();
    let spec = MutantSpec { contract: &contract, layout: &layout, const_len: &const_len, pop: 24, max_tries: 24 * 200 };
    let render = |t: &Tree| -> Result<String, String> { Ok(sexp(&k, t)) };
    let mut mutate = |g: &mut Vec<u32>, r: &mut GeneRng| -> Result<(), String> { point_mutate(&contract, &layout, &const_len, g, 1, r) };
    let m = mutants_of_first(&spec, vec![first.clone(), second.clone()], &mut mutate, &render, &mut GeneRng::new(9)).unwrap();
    assert_eq!(m.genes.len(), 24);
    assert_eq!((&m.genes[0], &m.genes[1]), (&first, &second), "the starts first, in order");
    assert_eq!((m.parents[0], m.parents[1]), (None, None));
    for (i, p) in m.parents.iter().enumerate().skip(2) {
        let p = p.expect("a mutant has a parent");
        assert!(p == 0 || p >= 2, "member {i} was made from start {p}, not the first start or a mutant");
        assert!(p < i);
        validate(&contract, &layout, &m.genes[i], &const_len).unwrap();
    }
    for i in 0..24 {
        for j in 0..i {
            assert!(m.texts[i] != m.texts[j] && m.trees[i] != m.trees[j], "members {i} and {j} repeat");
        }
    }
    assert!(mutants_of_first(&spec, vec![first.clone(), first.clone()], &mut mutate, &render, &mut GeneRng::new(9)).is_err(), "a start repeating another is an error");
    let one = mutants_of_first(&spec, vec![first.clone()], &mut mutate, &render, &mut GeneRng::new(9)).unwrap();
    assert_eq!(one, mutants(&spec, first, &mut mutate, &render, &mut GeneRng::new(9)).unwrap(), "one start is `mutants`");
}
