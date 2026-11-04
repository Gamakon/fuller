//! THE K-MEANS VIEW'S ELEMENT (`docs/kmeans_seed.md`): the faithful two-element view (i32 assignments, `lets a.i32.n`)
//! is measured: 131,825 types without the kernel boundary and 237,325 with it, over the 65,536 type codes, so it does
//! not load as a kingdom; the one-element view holds the integers exactly in f32. (The proof of the translation, the
//! reference program against UltraDim's three kernels ported to Rust, moved with the k-means measurement to phylu's
//! `futhark_eval::measure`, plan 5b.)

use super::*;

/// THE FAITHFUL TWO-ELEMENT VIEW, the one the plan's item 1 asked for (the assignments an `i32` array beside the f32
/// points, `lets a.i32.n`), measured 2026-11-03: at master version 4 (the pairs, no triple, no kernel boundary) it is
/// 53,773 types and 416,671 rows, 14 times the view of 4a, and its gene at N = 96 is 62,693,641 words, laid out in
/// 488 s; at version 5 the triples take it to 131,825 types and 1,014,061 rows WITHOUT the kernel boundary, and to
/// 237,325 types with it (`lets a.i32.n`, loaded once in 4.5 minutes): both over the 65,536 type codes a Futhark kingdom
/// has (`Ty::MAX_FUTHARK_DUALS`), so neither loads as a kingdom and the i32 translation cannot be built as a gene. That
/// is why the problem's view holds the assignments and the count as integers in f32 (`docs/kmeans_seed.md`, "The two
/// views"). This test pins the version-5 count without the let (96 s to load) and the limit.
#[test]
fn the_two_element_view_is_over_the_type_codes_with_or_without_the_kernel_boundary() {
    let two = KMEANS.spec.replace("element\tf32\t\t\n", "element\tf32,i32\t\t\n").replace("lets\ta.n\t\t\n", "").replace("literals\t0,1,2,-1,0.5,10\t\t\n", "literals\t0,1,2,-1,0.5,10\tf32\t\nliterals\t0,1,2,-1,10\ti32\t\n");
    assert_ne!(two, KMEANS.spec);
    let t0 = std::time::Instant::now();
    let view = crate::futhark_view::View::load(&crate::futhark_view::Master::committed().unwrap(), &two).unwrap();
    let types = view.types_tsv.lines().count() - 1;
    let rows = view.functions_tsv.lines().count() - 1;
    eprintln!("kmeans with i32 assignments, no kernel boundary: {types} types, {rows} rows, loaded in {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!((types, rows), (131_825, 1_014_061), "the two-element view's size changed");
    assert!(types > Ty::MAX_FUTHARK_DUALS, "the two-element view is over the type codes");
    assert!(view.kingdom("two elements").unwrap_err().contains("65536"));
    // The problem's one-element view, for the ratio.
    let one = load(&KMEANS).unwrap();
    assert_eq!(one.duals.len(), 14_359);
    eprintln!("the two-element view is {:.1}x the one-element view with the kernel boundary, {:.1}x the one-element view of 4a (3,804 types)", types as f64 / one.duals.len() as f64, types as f64 / 3804.0);
    // With the kernel boundary the view has 237,325 types (measured 2026-11-03, 4.5 minutes to load); the kingdom
    // loader names the limit: `Ty::check_dual_count`.
    let e = Ty::check_dual_count("Futhark Sig kmeans two elements", 237_325, Ty::MAX_FUTHARK_DUALS).unwrap_err();
    assert!(e.contains("237325") && e.contains("65536"), "{e}");
}
