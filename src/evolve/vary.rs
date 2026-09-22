//! Step 2 — selection and variation: the CPU reference.
//!
//! One generation's variation phase, with geppy's rules (tools/mutation.py,
//! tools/crossover.py, as the engine schedules them): per island, a tournament
//! replaces the deme, the best `elites` rows pass unchanged, every other row is
//! the winner of a second tournament among the first's winners, cloned, put
//! through the nine mutation operators in order, then the three crossovers over
//! consecutive pairs.
//!
//! Written as index arithmetic over the flat buffers, the way `evolve.wgsl`
//! does it, so the two can be read side by side; `device.rs` asserts they agree
//! bit for bit. Every random decision is a `draw` keyed by (generation, row,
//! slot, stream): nothing depends on the order rows are processed in.

use super::{below, coin, draw, Layout, Population, SymbolCodes};

pub const STREAM_SELECT_1: u32 = 6;
pub const STREAM_SELECT_2: u32 = 7;
pub const STREAM_MUT_HIT: u32 = 8;
pub const STREAM_MUT_KIND: u32 = 9;
pub const STREAM_MUT_SYMBOL: u32 = 10;
pub const STREAM_OPERATOR: u32 = 11;
pub const STREAM_INVERT: u32 = 12;
pub const STREAM_IS: u32 = 13;
pub const STREAM_RIS: u32 = 14;
pub const STREAM_GENE_T: u32 = 15;
pub const STREAM_DC_HIT: u32 = 16;
pub const STREAM_DC_VALUE: u32 = 17;
pub const STREAM_INVERT_DC: u32 = 18;
pub const STREAM_TRANSPOSE_DC: u32 = 19;
pub const STREAM_RNC_HIT: u32 = 20;
pub const STREAM_RNC_VALUE: u32 = 21;
pub const STREAM_CX_1P: u32 = 22;
pub const STREAM_CX_2P: u32 = 23;
pub const STREAM_CX_GENE: u32 = 24;
pub const STREAM_CLEANSE: u32 = 25;

/// Slots of `STREAM_OPERATOR`: does this operator act on this row (pair)?
pub const OP_INVERT: u32 = 0;
pub const OP_IS: u32 = 1;
pub const OP_RIS: u32 = 2;
pub const OP_GENE_T: u32 = 3;
pub const OP_INVERT_DC: u32 = 4;
pub const OP_TRANSPOSE_DC: u32 = 5;
pub const OP_CX_1P: u32 = 6;
pub const OP_CX_2P: u32 = 7;
pub const OP_CX_GENE: u32 = 8;
pub const OP_CLEANSE: u32 = 9;

/// The longest segment a transposition moves; bounds the kernel's scratch array.
pub const MAX_SEGMENT: u32 = 512;

/// The cleansing mutation works on genes whose head + tail fit its scratch
/// arrays; a longer gene is left alone by it.
pub const MAX_CLEANSE: u32 = 128;

/// Rows `lo .. hi` evolve together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Island {
    pub lo: u32,
    pub hi: u32,
    pub elites: u32,
    pub tournsize: u32,
}

/// Probabilities as thresholds on the high word of a draw: an event happens
/// when `high_word < threshold`, and `u32::MAX` means always.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rates {
    pub mut_point: u32,
    pub invert: u32,
    pub is_transpose: u32,
    pub ris_transpose: u32,
    pub gene_transpose: u32,
    pub dc_point: u32,
    pub invert_dc: u32,
    pub transpose_dc: u32,
    pub rnc_point: u32,
    pub cx_one_point: u32,
    pub cx_two_point: u32,
    pub cx_gene: u32,
    /// The cleansing mutation, a row: a function node of one gene's expression
    /// is replaced by one of its own children (`f(u) -> u`, `op(a, b) -> a`), or
    /// — with probability `cleanse_collapse` — its whole subtree by a constant.
    pub cleanse: u32,
    pub cleanse_collapse: u32,
}

pub fn threshold(p: f64) -> u32 {
    if p >= 1.0 {
        u32::MAX
    } else if p <= 0.0 {
        0
    } else {
        (p * 4_294_967_296.0) as u32
    }
}

impl Rates {
    /// The engine's schedule: point mutation 0.05 a slot, the transpositions and
    /// inversions 0.1 a row, 0.5 expected constant mutations per individual,
    /// crossovers 0.3 / 0.2 / 0.1 a pair.
    pub fn engine_defaults(layout: Layout) -> Rates {
        Rates {
            mut_point: threshold(0.05),
            invert: threshold(0.1),
            is_transpose: threshold(0.1),
            ris_transpose: threshold(0.1),
            gene_transpose: threshold(0.1),
            dc_point: threshold(0.05),
            invert_dc: threshold(0.1),
            transpose_dc: threshold(0.1),
            rnc_point: threshold(0.5 / f64::from(layout.n_genes * layout.n_rnc)),
            cx_one_point: threshold(0.3),
            cx_two_point: threshold(0.2),
            cx_gene: threshold(0.1),
            cleanse: 0,
            cleanse_collapse: threshold(0.25),
        }
    }

    /// The engine's schedule plus the cleansing mutation at `p` a row.
    pub fn with_cleanse(layout: Layout, p: f64) -> Rates {
        Rates { cleanse: threshold(p), ..Rates::engine_defaults(layout) }
    }
}

/// The cleansing mutation on ONE gene (`tokens` = its head, tail and Dc).
///
/// GEP's own operators keep a gene's length and let its EXPRESSED tree drift
/// larger; this is the pressure the other way, and it costs fitness nothing:
/// selection keeps a cleansed gene that fits as well and drops one that lost
/// something real. It is not a token edit — removing a node moves everything
/// under it in level order — so the expression is decoded, edited as a tree and
/// written back in level order, the Dc domain rebuilt so each surviving "?"
/// still reads ITS constant. Left unchanged when the expression does not close,
/// has no function, the result would put a function outside the head, or would
/// need more Dc slots than there are. `pick`, `kind`, `which`, `value` are the
/// draws (a function node; collapse or promote; which child; the new "?"'s Dc).
pub fn cleanse_gene(tokens: &mut [u32], layout: Layout, codes: &SymbolCodes, pick: u64, collapse: bool, which: u64, value: u64) -> bool {
    cleanse_gene_within(tokens, layout, layout.head, codes, (pick, collapse, which, value))
}

/// [`cleanse_gene`] with the head its functions must stay inside given: the
/// virtual head. `draws` = (pick, collapse, which, value).
pub fn cleanse_gene_within(tokens: &mut [u32], layout: Layout, vhead: u32, codes: &SymbolCodes, draws: (u64, bool, u64, u64)) -> bool {
    let (pick, collapse, which, value) = draws;
    let Some(tree) = GeneTree::of(tokens, layout, codes) else { return false };
    let arity = |id: u32| codes.arity[id as usize] as usize;
    let n_fn = (0..tree.n).filter(|&i| arity(tokens[i]) > 0).count() as u32;
    if n_fn == 0 {
        return false;
    }
    let nth = below(pick, n_fn) as usize;
    let p = (0..tree.n).filter(|&i| arity(tokens[i]) > 0).nth(nth).unwrap_or(0);
    let collapse = collapse && codes.rnc_id.is_some();
    // Collapsed, the subtree becomes ONE new "?" reading a drawn constant.
    let leaf = [Graft { token: codes.rnc_id.unwrap_or(0), kids: [0, 0], dc: below(value, layout.n_rnc) }];
    let q = if collapse { GRAFT } else { tree.child[p] + below(which, arity(tokens[p]) as u32) as usize };
    relevel(tokens, layout, vhead, codes, &tree, &[(p, q)], &leaf).is_ok()
}

/// A gene's expressed tree by gene position (Karva): position `i`'s children are
/// `child[i] ..`, and a "?" is the `ordinal[i]`-th of the gene (`usize::MAX`:
/// not a "?"). Fixed scratch, as the kernel has it.
pub struct GeneTree {
    /// Expressed positions.
    pub n: usize,
    pub child: [usize; MAX_CLEANSE as usize],
    pub ordinal: [usize; MAX_CLEANSE as usize],
}

impl GeneTree {
    /// `None`: the gene is longer than the scratch, or its expression does not
    /// close within head + tail.
    pub fn of(tokens: &[u32], layout: Layout, codes: &SymbolCodes) -> Option<GeneTree> {
        let ht = (layout.head + layout.tail) as usize;
        if ht > MAX_CLEANSE as usize {
            return None;
        }
        let arity = |id: u32| codes.arity[id as usize] as usize;
        let (mut need, mut n) = (1i64, 0usize);
        while need > 0 && n < ht {
            need += arity(tokens[n]) as i64 - 1;
            n += 1;
        }
        if need > 0 {
            return None;
        }
        let mut tree = GeneTree { n, child: [0; MAX_CLEANSE as usize], ordinal: [usize::MAX; MAX_CLEANSE as usize] };
        let (mut at, mut n_rnc) = (1usize, 0usize);
        for (i, &token) in tokens.iter().enumerate().take(n) {
            tree.child[i] = at;
            at += arity(token);
            if Some(token) == codes.rnc_id {
                tree.ordinal[i] = n_rnc;
                n_rnc += 1;
            }
        }
        Some(tree)
    }
}

/// A node [`relevel`] writes that the gene did not hold: its token, its children
/// (tree entries, as many as the token's arity) and, for a "?", the Dc index it
/// reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Graft {
    pub token: u32,
    pub kids: [usize; 2],
    pub dc: u32,
}

/// A tree entry at or above this is `grafts[entry - GRAFT]`; below it, a gene
/// position.
pub const GRAFT: usize = 1 << 16;

/// Why [`relevel`] left a gene as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unfit {
    /// A function would sit at or past the (virtual) head.
    HeadOversize,
    /// The expression would not close within head + tail.
    NotClosed,
    /// More "?" than the Dc domain has slots.
    DcOversize,
}

/// THE RE-SERIALISER the cleanse and snap's write-back share: the gene's tree,
/// with each position `from` of `swaps` replaced by the entry `to` (a gene
/// position, or a [`Graft`]), written back in level order from the root, the Dc
/// domain rebuilt so each surviving "?" still reads ITS constant. On `Err` the
/// gene is exactly as it was.
pub fn relevel(
    tokens: &mut [u32],
    layout: Layout,
    vhead: u32,
    codes: &SymbolCodes,
    tree: &GeneTree,
    swaps: &[(usize, usize)],
    grafts: &[Graft],
) -> Result<(), Unfit> {
    let (h, ht) = (vhead as usize, (layout.head + layout.tail) as usize);
    let arity = |id: u32| codes.arity[id as usize] as usize;
    let swapped = |c: usize| swaps.iter().find(|(from, _)| *from == c).map_or(c, |(_, to)| *to);
    let mut queue = [0usize; MAX_CLEANSE as usize];
    let mut new_tok = [0u32; MAX_CLEANSE as usize];
    let mut new_dc = [0u32; MAX_CLEANSE as usize];
    queue[0] = swapped(0);
    let (mut head, mut tail, mut m, mut k) = (0usize, 1usize, 0usize, 0usize);
    while head < tail {
        let i = queue[head];
        head += 1;
        let (token, dc, kids) = if i >= GRAFT {
            let g = grafts[i - GRAFT];
            (g.token, g.dc, g.kids)
        } else {
            // A "?" the old Dc domain had no slot for read nothing; keep that.
            let o = tree.ordinal[i];
            let dc = if o < layout.tail as usize { tokens[ht + o] } else { 0 };
            (tokens[i], dc, [swapped(tree.child[i]), swapped(tree.child[i] + 1)])
        };
        if m >= ht || tail + arity(token) > MAX_CLEANSE as usize {
            return Err(Unfit::NotClosed);
        }
        new_tok[m] = token;
        m += 1;
        if Some(token) == codes.rnc_id {
            new_dc[k] = dc;
            k += 1;
        }
        for kid in kids.iter().take(arity(token)) {
            queue[tail] = *kid;
            tail += 1;
        }
    }
    if (0..m).any(|x| arity(new_tok[x]) > 0 && x >= h) {
        return Err(Unfit::HeadOversize);
    }
    if k > layout.tail as usize {
        return Err(Unfit::DcOversize);
    }
    tokens[..m].copy_from_slice(&new_tok[..m]);
    tokens[ht..ht + k].copy_from_slice(&new_dc[..k]);
    Ok(())
}

pub fn chance(h: u64, thr: u32) -> bool {
    thr == u32::MAX || ((h >> 32) as u32) < thr
}

// ---------------------------------------------------------------------------
// THE BEAM'S NEIGHBOURHOOD. One individual, many mutations of it — the operators
// above, aimed at ONE row instead of a whole generation. Nothing here scores
// anything: it proposes, and the caller (`engine::Engine::beam`) judges with HFF
// on the data, exactly as `physics::generate` proposes and its caller judges.
// ---------------------------------------------------------------------------

/// What the neighbourhood is drawn against: the row it mutates, the head its
/// functions must stay inside, and the constants a drawn one may take.
#[derive(Clone, Copy, Debug)]
pub struct BeamParams {
    pub seed: u32,
    /// The generation the beat is on: with `seed` and the mutant's index it keys
    /// every draw, so a beat's whole neighbourhood is reproducible.
    pub generation: u32,
    pub rnc_lo: i32,
    pub rnc_hi: i32,
    pub vhead: u32,
    /// Which of the chromosome's genes the model USES (bit g): a mutation of a
    /// gene the model does not use changes nothing it is scored on, so the
    /// neighbourhood never spends a mutant there. `Scored::genes`.
    pub genes: u32,
}

/// A mutant of one individual: its genome row, its constants, and how it was
/// made — so a beat can report which operator found the winner.
#[derive(Clone, Debug, PartialEq)]
pub struct Mutant {
    pub genome: Vec<u32>,
    pub rnc: Vec<f32>,
    pub kind: BeamKind,
}

/// The operator that made a mutant. The cleanse is first because it is the one
/// that removes a spurious factor — "one subtree away" — and the near misses in
/// the solution ledger are one subtree from their law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamKind {
    /// `cleanse_gene_within` promoting a child over its parent: `f(u) -> u`,
    /// `op(a, b) -> a`. The spurious factor removed.
    Promote,
    /// `cleanse_gene_within` collapsing a whole subtree to one drawn constant.
    Collapse,
    /// One symbol of one gene replaced, as `mutate`'s step 1 does it.
    Point,
    /// One Dc slot pointed at another of the gene's constants.
    Dc,
    /// One of the gene's constants redrawn.
    Rnc,
}

/// THE NEIGHBOURHOOD of one individual: up to `width` mutants of it, all
/// distinct from the original and from each other, every one a valid gene.
///
/// The mix, and why. The CLEANSE space is enumerated WHOLE and first — every
/// function node of every used gene, promoted to each of its children and
/// collapsed to a constant. It is small (a few hundred at head 34) and it is the
/// operator the evidence asks for: feynman_test_4's report is the law with one
/// spurious `sqrt(x_3 - 0.909)` factor, and promoting that factor's parent over
/// its other child deletes exactly that. Enumerated rather than drawn because a
/// few hundred is cheaper to enumerate than to hit by sampling, and because the
/// beat then cannot miss the one node that matters.
///
/// What is left of `width` is filled with DRAWN mutations, cycling point, Dc and
/// constant in that order: point mutation because a law is often one symbol from
/// an imitation (`x_2` where the gene has `x_3`), and the two Dc operators
/// because a near miss whose structure is right may need one constant moved. The
/// transpositions, the inversions and the crossovers are NOT here: they rearrange
/// large blocks, which is a different search from "one node away", and crossover
/// needs a second parent the beam does not have.
///
/// Draws are keyed by `(seed, generation, mutant index)` through the engine's own
/// counter-based `draw`, so the same beat gives the same neighbourhood. A
/// mutation that changes nothing (a cleanse that does not apply, a point mutation
/// that writes the symbol already there) is dropped, not counted.
pub fn neighbourhood(genome: &[u32], rnc: &[f32], layout: Layout, codes: &SymbolCodes, p: &BeamParams, width: u32) -> Result<Vec<Mutant>, String> {
    let vhead = super::virtual_head(p.vhead, layout)?;
    let (width_g, ht, g_n, nr) = (layout.gene_width() as usize, (layout.head + layout.tail) as usize, layout.n_genes as usize, layout.n_rnc as usize);
    if genome.len() != g_n * width_g || rnc.len() != g_n * nr {
        return Err(format!("the row is {} tokens and {} constants, the layout wants {} and {}", genome.len(), rnc.len(), g_n * width_g, g_n * nr));
    }
    let used: Vec<usize> = (0..g_n).filter(|g| p.genes >> g & 1 == 1).collect();
    if used.is_empty() {
        return Err("the neighbourhood needs at least one used gene".into());
    }
    let arity = |id: u32| codes.arity[id as usize] as usize;
    let mut out: Vec<Mutant> = Vec::new();
    let mut seen: Vec<(Vec<u32>, Vec<u32>)> = vec![(genome.to_vec(), rnc.iter().map(|v| v.to_bits()).collect())];
    // Keep a mutant unless it is the original or one already made.
    let keep = |genome: Vec<u32>, rnc: Vec<f32>, kind: BeamKind, out: &mut Vec<Mutant>, seen: &mut Vec<(Vec<u32>, Vec<u32>)>| {
        let key = (genome.clone(), rnc.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
        if seen.contains(&key) {
            return;
        }
        seen.push(key);
        out.push(Mutant { genome, rnc, kind });
    };

    // 1. THE CLEANSE NEIGHBOURHOOD, enumerated whole: every function node of every
    //    used gene, promoted over each child and collapsed to a constant. The
    //    draws `cleanse_gene_within` takes are built so it picks exactly the node
    //    and the child this entry names — `below(h, n)` maps the top bits, so
    //    `(nth << 1 | 1) * (u64::MAX / n)` lands inside bucket `nth`.
    let bucket = |nth: usize, n: usize| -> u64 {
        if n <= 1 {
            0
        } else {
            (u64::MAX / n as u64).saturating_mul(nth as u64) + u64::MAX / (2 * n as u64)
        }
    };
    for &g in &used {
        let tokens = &genome[g * width_g..(g + 1) * width_g];
        let Some(tree) = GeneTree::of(tokens, layout, codes) else { continue };
        let functions: Vec<usize> = (0..tree.n).filter(|&i| arity(tokens[i]) > 0).collect();
        for (nth, &node) in functions.iter().enumerate() {
            let pick = bucket(nth, functions.len());
            for child in 0..arity(tokens[node]) {
                let mut mutant = genome.to_vec();
                let gene = &mut mutant[g * width_g..(g + 1) * width_g];
                if cleanse_gene_within(gene, layout, vhead, codes, (pick, false, bucket(child, arity(tokens[node])), 0)) {
                    keep(mutant, rnc.to_vec(), BeamKind::Promote, &mut out, &mut seen);
                }
            }
            // The collapse puts ONE new "?" on each of the gene's constants in
            // turn: the same subtree removed, standing for a different value.
            if codes.rnc_id.is_some() {
                for slot in 0..nr {
                    let mut mutant = genome.to_vec();
                    let gene = &mut mutant[g * width_g..(g + 1) * width_g];
                    if cleanse_gene_within(gene, layout, vhead, codes, (pick, true, 0, bucket(slot, nr))) {
                        keep(mutant, rnc.to_vec(), BeamKind::Collapse, &mut out, &mut seen);
                    }
                }
            }
        }
    }
    out.truncate(width as usize);

    // 2. The drawn mutations, filling what is left: point, Dc, constant, in turn.
    //    Every one is a single edit of a single slot of a single used gene.
    let (nf, nt) = (codes.sample_functions.len() as u32, codes.sample_terminals.len() as u32);
    let span = (p.rnc_hi - p.rnc_lo + 1) as u32;
    // A bounded number of tries: a neighbourhood whose every drawn edit is a
    // no-op must end, not spin. Four tries per wanted mutant is generous and is
    // the whole rule — nothing here retries silently for ever.
    let mut index = 0u32;
    let tries = u64::from(width).saturating_mul(4);
    while (out.len() as u32) < width && u64::from(index) < tries {
        let d = |slot, stream| draw(p.seed, p.generation, index, slot, stream);
        let g = used[below(d(0, STREAM_BEAM), used.len() as u32) as usize];
        match index % 3 {
            0 => {
                // A point mutation of one head-or-tail slot, by `mutate`'s rule: a
                // function only below the virtual head, a terminal anywhere.
                let pos = below(d(1, STREAM_MUT_HIT), ht as u32);
                let token = if pos < vhead && coin(d(2, STREAM_MUT_KIND)) {
                    codes.sample_functions[below(d(3, STREAM_MUT_SYMBOL), nf) as usize]
                } else {
                    codes.sample_terminals[below(d(3, STREAM_MUT_SYMBOL), nt) as usize]
                };
                let at = g * width_g + pos as usize;
                if genome[at] != token {
                    let mut mutant = genome.to_vec();
                    mutant[at] = token;
                    keep(mutant, rnc.to_vec(), BeamKind::Point, &mut out, &mut seen);
                }
            }
            1 => {
                // One Dc slot pointed at another of the gene's constants.
                let pos = ht + below(d(1, STREAM_DC_HIT), layout.tail) as usize;
                let value = below(d(2, STREAM_DC_VALUE), layout.n_rnc);
                let at = g * width_g + pos;
                if genome[at] != value {
                    let mut mutant = genome.to_vec();
                    mutant[at] = value;
                    keep(mutant, rnc.to_vec(), BeamKind::Dc, &mut out, &mut seen);
                }
            }
            _ => {
                // One of the gene's constants redrawn, from the same range the
                // population was born in.
                let k = g * nr + below(d(1, STREAM_RNC_HIT), layout.n_rnc) as usize;
                let value = (p.rnc_lo + below(d(2, STREAM_RNC_VALUE), span) as i32) as f32;
                if rnc[k] != value {
                    let mut constants = rnc.to_vec();
                    constants[k] = value;
                    keep(genome.to_vec(), constants, BeamKind::Rnc, &mut out, &mut seen);
                }
            }
        }
        index += 1;
    }
    out.truncate(width as usize);
    Ok(out)
}

/// The beam's own stream: which used gene a drawn mutation edits.
pub const STREAM_BEAM: u32 = 26;

#[derive(Clone, Copy, Debug)]
pub struct GenParams {
    pub seed: u32,
    pub generation: u32,
    pub rnc_lo: i32,
    pub rnc_hi: i32,
    /// The virtual head (0 = the whole head): see [`super::virtual_head`]. Point
    /// mutation writes a function only below it; inversion, IS and RIS work inside
    /// it; the cleanse keeps functions inside it. Crossover needs no rule: both
    /// parents already keep theirs.
    pub vhead: u32,
}

/// A population with its fitness (NaN = not evaluated; lower is fitter).
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub pop: Population,
    pub fitness: Vec<f32>,
}

/// Fitness as a sort key: lower is fitter, an unevaluated row (NaN) is last.
/// Capped at f32::MAX, as the kernel does, so an infinite fitness and an
/// unevaluated one tie the same way on both sides.
fn key(f: f32) -> f32 {
    if f.is_nan() {
        f32::MAX
    } else {
        f.min(f32::MAX)
    }
}

pub fn validate(layout: Layout, islands: &[Island]) -> Result<(), String> {
    if layout.head < 2 || layout.head > MAX_SEGMENT || layout.tail > MAX_SEGMENT {
        return Err(format!("head must be 2..={MAX_SEGMENT} and tail <= {MAX_SEGMENT}"));
    }
    let mut next = 0;
    for (i, isl) in islands.iter().enumerate() {
        if isl.lo != next || isl.hi <= isl.lo || isl.elites >= isl.hi - isl.lo || isl.tournsize == 0 {
            return Err(format!("island {i} must follow the last, be non-empty, keep elites < size and tournsize > 0"));
        }
        if (isl.hi - isl.lo - isl.elites) % 2 != 0 {
            return Err(format!("island {i}: offspring are mated in pairs, so size - elites must be even"));
        }
        next = isl.hi;
    }
    if next != layout.pop {
        return Err(format!("islands cover {next} rows of a population of {}", layout.pop));
    }
    Ok(())
}

/// For every row of the NEXT generation, the row of this one it is cloned from.
pub fn select(fitness: &[f32], islands: &[Island], p: &GenParams) -> Vec<u32> {
    let d = |row, slot, stream| draw(p.seed, p.generation, row, slot, stream);
    let mut parent = vec![0u32; fitness.len()];
    for isl in islands {
        let n = isl.hi - isl.lo;
        // Elites: the fittest rows, ties to the lower row.
        let mut taken: Vec<u32> = Vec::new();
        for j in 0..isl.elites {
            let mut best = u32::MAX;
            for r in isl.lo..isl.hi {
                if taken.contains(&r) {
                    continue;
                }
                if best == u32::MAX || key(fitness[r as usize]) < key(fitness[best as usize]) {
                    best = r;
                }
            }
            taken.push(best);
            parent[(isl.lo + j) as usize] = best;
        }
        // The first tournament fills slot s of the replaced deme; the second
        // draws among those slots.
        let first = |s: u32| -> u32 {
            let mut w = u32::MAX;
            for i in 0..isl.tournsize {
                let c = isl.lo + below(d(isl.lo + s, i, STREAM_SELECT_1), n);
                if w == u32::MAX || key(fitness[c as usize]) < key(fitness[w as usize]) {
                    w = c;
                }
            }
            w
        };
        for r in isl.lo + isl.elites..isl.hi {
            let mut w = u32::MAX;
            for j in 0..isl.tournsize {
                let c = first(below(d(r, j, STREAM_SELECT_2), n));
                if w == u32::MAX || key(fitness[c as usize]) < key(fitness[w as usize]) {
                    w = c;
                }
            }
            parent[r as usize] = w;
        }
    }
    parent
}

/// Clone each row from its parent and apply the nine mutation operators, in
/// geppy's order. Elites are copied with their fitness; offspring are unevaluated.
pub fn mutate(
    now: &Generation,
    parent: &[u32],
    islands: &[Island],
    codes: &SymbolCodes,
    rates: &Rates,
    p: &GenParams,
) -> Generation {
    let l = now.pop.layout;
    let (h, t) = (l.head, l.tail);
    let (ht, width, g_n, nr) = (h + t, l.gene_width(), l.n_genes, l.n_rnc);
    // Every head operator works inside the VIRTUAL head; a wrong one is a bug in
    // the caller (`device::vary` and `init` refuse it), not a case to run with.
    let vh = super::virtual_head(p.vhead, l).expect("the virtual head");
    let row_w = (g_n * width) as usize;
    let rnc_w = (g_n * nr) as usize;
    let (nf, nt) = (codes.sample_functions.len() as u32, codes.sample_terminals.len() as u32);
    let span = (p.rnc_hi - p.rnc_lo + 1) as u32;
    let mut next = now.clone();
    for isl in islands {
        for r in isl.lo..isl.hi {
            let src = parent[r as usize] as usize;
            let ru = r as usize;
            next.pop.genome[ru * row_w..(ru + 1) * row_w].copy_from_slice(&now.pop.genome[src * row_w..(src + 1) * row_w]);
            next.pop.rnc[ru * rnc_w..(ru + 1) * rnc_w].copy_from_slice(&now.pop.rnc[src * rnc_w..(src + 1) * rnc_w]);
            next.pop.wrapper_id[ru] = now.pop.wrapper_id[src];
            if r < isl.lo + isl.elites {
                next.fitness[ru] = now.fitness[src];
                continue;
            }
            next.fitness[ru] = f32::NAN;
            let d = |slot, stream| draw(p.seed, p.generation, r, slot, stream);
            let acts = |op, thr| chance(d(op, STREAM_OPERATOR), thr);
            let row = &mut next.pop.genome[ru * row_w..(ru + 1) * row_w];
            let at = |g: u32, pos: u32| (g * width + pos) as usize;

            // 1. uniform point mutation
            for g in 0..g_n {
                for pos in 0..ht {
                    let slot = g * width + pos;
                    if !chance(d(slot, STREAM_MUT_HIT), rates.mut_point) {
                        continue;
                    }
                    row[at(g, pos)] = if pos < vh && coin(d(slot, STREAM_MUT_KIND)) {
                        codes.sample_functions[below(d(slot, STREAM_MUT_SYMBOL), nf) as usize]
                    } else {
                        codes.sample_terminals[below(d(slot, STREAM_MUT_SYMBOL), nt) as usize]
                    };
                }
            }
            // 2. inversion, inside one head
            if acts(OP_INVERT, rates.invert) {
                let g = below(d(0, STREAM_INVERT), g_n);
                let len = 2 + below(d(1, STREAM_INVERT), vh - 1);
                let start = below(d(2, STREAM_INVERT), vh - len + 1);
                row[at(g, start)..at(g, start + len)].reverse();
            }
            // 3. IS transposition: a segment from anywhere in head+tail, into a
            //    head, never at the root
            if acts(OP_IS, rates.is_transpose) {
                let donor = below(d(0, STREAM_IS), g_n);
                let donee = below(d(1, STREAM_IS), g_n);
                let len = 1 + below(d(2, STREAM_IS), vh - 1);
                let start = below(d(3, STREAM_IS), ht - len + 1);
                let ins = 1 + below(d(4, STREAM_IS), vh - len);
                let seg: Vec<u32> = row[at(donor, start)..at(donor, start + len)].to_vec();
                let head: Vec<u32> = row[at(donee, 0)..at(donee, vh)].to_vec();
                for pos in ins + len..vh {
                    row[at(donee, pos)] = head[(pos - len) as usize];
                }
                row[at(donee, ins)..at(donee, ins + len)].copy_from_slice(&seg);
            }
            // 4. RIS transposition: a segment that starts at a function, to the root
            if acts(OP_RIS, rates.ris_transpose) {
                for trial in 0..2 * g_n + 1 {
                    let donor = below(d(trial * 4, STREAM_RIS), g_n);
                    let donee = below(d(trial * 4 + 1, STREAM_RIS), g_n);
                    let n_fn = (0..vh).filter(|&pos| codes.arity[row[at(donor, pos)] as usize] > 0).count() as u32;
                    if n_fn == 0 {
                        continue;
                    }
                    let pick = below(d(trial * 4 + 2, STREAM_RIS), n_fn);
                    let start = (0..vh).filter(|&pos| codes.arity[row[at(donor, pos)] as usize] > 0).nth(pick as usize).unwrap_or(0);
                    let len = 2 + below(d(trial * 4 + 3, STREAM_RIS), vh.min(ht - start) - 1);
                    let seg: Vec<u32> = row[at(donor, start)..at(donor, start + len)].to_vec();
                    let head: Vec<u32> = row[at(donee, 0)..at(donee, vh)].to_vec();
                    for pos in len..vh {
                        row[at(donee, pos)] = head[(pos - len) as usize];
                    }
                    row[at(donee, 0)..at(donee, len)].copy_from_slice(&seg);
                    break;
                }
            }
            // 5. gene transposition: gene 0 changes places with another — its
            //    constants go with it
            if g_n > 1 && acts(OP_GENE_T, rates.gene_transpose) {
                let source = 1 + below(d(0, STREAM_GENE_T), g_n - 1);
                for pos in 0..width {
                    row.swap(at(0, pos), at(source, pos));
                }
                let consts = &mut next.pop.rnc[ru * rnc_w..(ru + 1) * rnc_w];
                for k in 0..nr {
                    consts.swap(k as usize, (source * nr + k) as usize);
                }
            }
            let row = &mut next.pop.genome[ru * row_w..(ru + 1) * row_w];
            // 6. Dc point mutation
            for g in 0..g_n {
                for pos in ht..width {
                    let slot = g * width + pos;
                    if chance(d(slot, STREAM_DC_HIT), rates.dc_point) {
                        row[at(g, pos)] = below(d(slot, STREAM_DC_VALUE), nr);
                    }
                }
            }
            // 7. Dc inversion. geppy chooses `len` elements and reverses len-1.
            if t >= 2 && acts(OP_INVERT_DC, rates.invert_dc) {
                let g = below(d(0, STREAM_INVERT_DC), g_n);
                let len = 2 + below(d(1, STREAM_INVERT_DC), t - 1);
                let start = ht + below(d(2, STREAM_INVERT_DC), t - len + 1);
                row[at(g, start)..at(g, start + len - 1)].reverse();
            }
            // 8. Dc transposition
            if acts(OP_TRANSPOSE_DC, rates.transpose_dc) {
                let donor = below(d(0, STREAM_TRANSPOSE_DC), g_n);
                let donee = below(d(1, STREAM_TRANSPOSE_DC), g_n);
                let len = 1 + below(d(2, STREAM_TRANSPOSE_DC), t);
                let start = ht + below(d(3, STREAM_TRANSPOSE_DC), t - len + 1);
                let ins = ht + below(d(4, STREAM_TRANSPOSE_DC), t - len + 1);
                let seg: Vec<u32> = row[at(donor, start)..at(donor, start + len)].to_vec();
                let dc: Vec<u32> = row[at(donee, ht)..at(donee, width)].to_vec();
                for pos in ins + len..width {
                    row[at(donee, pos)] = dc[(pos - len - ht) as usize];
                }
                row[at(donee, ins)..at(donee, ins + len)].copy_from_slice(&seg);
            }
            // 9. the constants themselves
            let consts = &mut next.pop.rnc[ru * rnc_w..(ru + 1) * rnc_w];
            for k in 0..g_n * nr {
                if chance(d(k, STREAM_RNC_HIT), rates.rnc_point) {
                    consts[k as usize] = (p.rnc_lo + below(d(k, STREAM_RNC_VALUE), span) as i32) as f32;
                }
            }
            // 10. cleanse: shrink one gene's expression
            if acts(OP_CLEANSE, rates.cleanse) {
                let g = below(d(0, STREAM_CLEANSE), g_n);
                let row = &mut next.pop.genome[ru * row_w..(ru + 1) * row_w];
                cleanse_gene_within(
                    &mut row[(g * width) as usize..((g + 1) * width) as usize],
                    l,
                    vh,
                    codes,
                    (d(1, STREAM_CLEANSE), chance(d(2, STREAM_CLEANSE), rates.cleanse_collapse), d(3, STREAM_CLEANSE), d(4, STREAM_CLEANSE)),
                );
            }
        }
    }
    next
}

/// The three crossovers, over consecutive offspring pairs, in place. A pair's
/// draws are keyed by its second row.
pub fn crossover(next: &mut Generation, islands: &[Island], rates: &Rates, p: &GenParams) {
    let l = next.pop.layout;
    let (width, g_n, nr) = (l.gene_width(), l.n_genes, l.n_rnc);
    let row_w = (g_n * width) as usize;
    let rnc_w = (g_n * nr) as usize;
    for isl in islands {
        let first = isl.lo + isl.elites;
        let mut b = first + 1;
        while b < isl.hi {
            let a = b - 1;
            let d = |slot, stream| draw(p.seed, p.generation, b, slot, stream);
            let (au, bu) = (a as usize, b as usize);
            let swap_tokens = |genome: &mut [u32], g: u32, from: u32, to: u32| {
                for pos in from..to {
                    genome.swap(au * row_w + (g * width + pos) as usize, bu * row_w + (g * width + pos) as usize);
                }
            };
            let swap_consts = |rnc: &mut [f32], ga: u32, gb: u32| {
                for k in 0..nr {
                    rnc.swap(au * rnc_w + (ga * nr + k) as usize, bu * rnc_w + (gb * nr + k) as usize);
                }
            };
            if chance(d(OP_CX_1P, STREAM_OPERATOR), rates.cx_one_point) {
                let g = below(d(0, STREAM_CX_1P), g_n);
                let point = below(d(1, STREAM_CX_1P), width);
                for whole in 0..g {
                    swap_tokens(&mut next.pop.genome, whole, 0, width);
                    swap_consts(&mut next.pop.rnc, whole, whole);
                }
                swap_tokens(&mut next.pop.genome, g, 0, point + 1);
            }
            if chance(d(OP_CX_2P, STREAM_OPERATOR), rates.cx_two_point) {
                let (x, y) = (below(d(0, STREAM_CX_2P), g_n), below(d(1, STREAM_CX_2P), g_n));
                let (g1, g2) = (x.min(y), x.max(y));
                let (p1, p2) = (below(d(2, STREAM_CX_2P), width), below(d(3, STREAM_CX_2P), width));
                if g1 == g2 {
                    swap_tokens(&mut next.pop.genome, g1, p1.min(p2), p1.max(p2) + 1);
                } else {
                    for whole in g1 + 1..g2 {
                        swap_tokens(&mut next.pop.genome, whole, 0, width);
                        swap_consts(&mut next.pop.rnc, whole, whole);
                    }
                    swap_tokens(&mut next.pop.genome, g1, p1, width);
                    swap_tokens(&mut next.pop.genome, g2, 0, p2 + 1);
                }
            }
            if chance(d(OP_CX_GENE, STREAM_OPERATOR), rates.cx_gene) {
                let (ga, gb) = (below(d(0, STREAM_CX_GENE), g_n), below(d(1, STREAM_CX_GENE), g_n));
                for pos in 0..width {
                    next.pop.genome.swap(au * row_w + (ga * width + pos) as usize, bu * row_w + (gb * width + pos) as usize);
                }
                swap_consts(&mut next.pop.rnc, ga, gb);
            }
            b += 2;
        }
    }
}

/// One generation's variation phase: select, clone + mutate, recombine.
pub fn vary(
    now: &Generation,
    islands: &[Island],
    codes: &SymbolCodes,
    rates: &Rates,
    p: &GenParams,
) -> Result<Generation, String> {
    validate(now.pop.layout, islands)?;
    let parent = select(&now.fitness, islands, p);
    let mut next = mutate(now, &parent, islands, codes, rates, p);
    crossover(&mut next, islands, rates, p);
    Ok(next)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tests::{codes, params};
    use super::super::init;
    use super::*;
    use crate::evolve::SymbolCodes;

    pub(crate) fn islands() -> Vec<Island> {
        vec![
            Island { lo: 0, hi: 600, elites: 2, tournsize: 42 },
            Island { lo: 600, hi: 800, elites: 2, tournsize: 14 },
        ]
    }

    pub(crate) fn start(seed: u32) -> Generation {
        let layout = Layout::for_arity(800, 3, 48, 2, 10);
        let pop = init(layout, &codes(), &params(seed)).unwrap();
        let fitness = (0..800).map(|r| below(draw(seed, 0, r, 0, 99), 1_000_000) as f32).collect();
        Generation { pop, fitness }
    }

    fn gen_params(seed: u32, generation: u32) -> GenParams {
        GenParams { seed, generation, rnc_lo: -100, rnc_hi: 100, vhead: 0 }
    }

    #[test]
    fn the_rules_hold_through_200_generations() {
        let (codes, isl) = (codes(), islands());
        let rates = Rates::engine_defaults(Layout::for_arity(800, 3, 48, 2, 10));
        let mut now = start(3);
        for generation in 1..=200 {
            now = vary(&now, &isl, &codes, &rates, &gen_params(3, generation)).unwrap();
            now.pop.check(&codes).unwrap();
            for (r, f) in now.fitness.iter_mut().enumerate() {
                *f = below(draw(3, generation, r as u32, 1, 99), 1_000_000) as f32;
            }
        }
    }

    /// The deepest position of any gene that holds a function, plus one: the head
    /// the population is actually using.
    fn head_in_use(g: &Generation, codes: &SymbolCodes) -> usize {
        let l = g.pop.layout;
        let (h, width) = (l.head as usize, l.gene_width() as usize);
        g.pop.genome.chunks(width).map(|gene| (0..h).rev().find(|&p| codes.arity[gene[p] as usize] > 0).map_or(0, |p| p + 1)).max().unwrap_or(0)
    }

    #[test]
    fn no_function_ever_sits_past_the_virtual_head_and_raising_it_changes_no_gene() {
        let (codes, isl) = (codes(), islands());
        let layout = Layout::for_arity(800, 3, 48, 2, 10);
        // every operator on, the cleanse included
        let rates = Rates::with_cleanse(layout, 0.5);
        let born = crate::evolve::InitParams { seed: 5, generation: 0, rnc_lo: -100, rnc_hi: 100, n_wrappers: 3, vhead: 8 };
        let pop = init(layout, &codes, &born).unwrap();
        let mut now = Generation { pop, fitness: (0..800).map(|r| below(draw(5, 0, r, 0, 99), 1_000_000) as f32).collect() };
        assert!(head_in_use(&now, &codes) <= 8, "born with a function past the virtual head");
        for generation in 1..=200u32 {
            // the head grows by one position every 50 generations: 8, 9, 10, 11, 12
            let vhead = 8 + generation / 50;
            let before = now.clone();
            now = vary(&now, &isl, &codes, &rates, &GenParams { seed: 5, generation, rnc_lo: -100, rnc_hi: 100, vhead }).unwrap();
            now.pop.check(&codes).unwrap();
            assert!(head_in_use(&now, &codes) <= vhead as usize, "generation {generation}: a function past virtual head {vhead}");
            // the elites are the old genes, untouched by the step to a longer head
            let row_w = (layout.n_genes * layout.gene_width()) as usize;
            let best = (0..600).min_by(|&a, &b| before.fitness[a].total_cmp(&before.fitness[b]).then(a.cmp(&b))).unwrap();
            assert_eq!(now.pop.genome[..row_w], before.pop.genome[best * row_w..(best + 1) * row_w], "generation {generation}");
            for (r, f) in now.fitness.iter_mut().enumerate() {
                *f = below(draw(5, generation, r as u32, 1, 99), 1_000_000) as f32;
            }
        }
        // the room is used once it is there
        assert!(head_in_use(&now, &codes) > 8, "the population never grew into the longer head");
        // and a virtual head outside 2..=head is refused, not run with
        assert!(crate::evolve::virtual_head(1, layout).is_err() && crate::evolve::virtual_head(49, layout).is_err());
        assert_eq!(crate::evolve::virtual_head(0, layout), Ok(48));
    }

    #[test]
    fn elites_pass_unchanged_with_their_fitness_and_offspring_are_unevaluated() {
        let (codes, isl) = (codes(), islands());
        let now = start(4);
        let rates = Rates::engine_defaults(now.pop.layout);
        let next = vary(&now, &isl, &codes, &rates, &gen_params(4, 1)).unwrap();
        let w = (now.pop.layout.n_genes * now.pop.layout.gene_width()) as usize;
        for island in &isl {
            let mut order: Vec<u32> = (island.lo..island.hi).collect();
            order.sort_by(|&a, &b| now.fitness[a as usize].total_cmp(&now.fitness[b as usize]).then(a.cmp(&b)));
            for (j, &fittest) in order.iter().take(island.elites as usize).enumerate() {
                let (to, from) = (island.lo as usize + j, fittest as usize);
                assert_eq!(next.pop.genome[to * w..(to + 1) * w], now.pop.genome[from * w..(from + 1) * w]);
                assert_eq!(next.fitness[to], now.fitness[from]);
            }
            assert!((island.lo + island.elites..island.hi).all(|r| next.fitness[r as usize].is_nan()));
        }
    }

    #[test]
    fn selection_prefers_the_fit_and_stays_inside_its_island() {
        let isl = islands();
        let now = start(5);
        let parent = select(&now.fitness, &isl, &gen_params(5, 1));
        for island in &isl {
            let rows = island.lo as usize..island.hi as usize;
            assert!(parent[rows.clone()].iter().all(|&p| p >= island.lo && p < island.hi));
            let mean = |v: &mut dyn Iterator<Item = f32>| { let v: Vec<f32> = v.collect(); v.iter().sum::<f32>() / v.len() as f32 };
            let all = mean(&mut now.fitness[rows.clone()].iter().copied());
            let chosen = mean(&mut parent[rows].iter().map(|&p| now.fitness[p as usize]));
            assert!(chosen < all * 0.2, "selected mean {chosen} against island mean {all}");
        }
    }

    #[test]
    fn the_same_seed_is_the_same_generation() {
        let (codes, isl) = (codes(), islands());
        let now = start(6);
        let rates = Rates::engine_defaults(now.pop.layout);
        let a = vary(&now, &isl, &codes, &rates, &gen_params(6, 9)).unwrap();
        assert_eq!(a.pop, vary(&now, &isl, &codes, &rates, &gen_params(6, 9)).unwrap().pop);
        assert_ne!(a.pop.genome, vary(&now, &isl, &codes, &rates, &gen_params(6, 10)).unwrap().pop.genome);
    }

    /// A gene's expression as a nested string with every "?" resolved to its Dc
    /// entry — and, when `swap` is given, with the subtree at old node `swap.0`
    /// replaced by `swap.1` (an old node, or None for a new constant whose Dc
    /// entry is `swap.2`). An independent recursive reading of level order.
    fn show(tokens: &[u32], layout: Layout, codes: &SymbolCodes, swap: Option<(usize, Option<usize>, u32)>) -> Option<String> {
        let ht = (layout.head + layout.tail) as usize;
        let arity = |id: u32| codes.arity[id as usize] as usize;
        let (mut need, mut n) = (1i64, 0usize);
        while need > 0 && n < ht {
            need += arity(tokens[n]) as i64 - 1;
            n += 1;
        }
        if need > 0 {
            return None;
        }
        let mut child = vec![0usize; n];
        let mut ordinal = vec![usize::MAX; n];
        let (mut ptr, mut k) = (1usize, 0usize);
        for i in 0..n {
            child[i] = ptr;
            ptr += arity(tokens[i]);
            if Some(tokens[i]) == codes.rnc_id {
                ordinal[i] = k;
                k += 1;
            }
        }
        fn go(i: usize, t: &[u32], child: &[usize], ordinal: &[usize], ht: usize, arity: &dyn Fn(u32) -> usize,
              swap: Option<(usize, Option<usize>, u32)>) -> String {
            if let Some((p, q, dc)) = swap {
                if i == p {
                    return match q {
                        Some(q) => go(q, t, child, ordinal, ht, arity, None),
                        None => format!("?{dc}"),
                    };
                }
            }
            if ordinal[i] != usize::MAX {
                return format!("?{}", t[ht + ordinal[i]]);
            }
            let kids: Vec<String> = (0..arity(t[i])).map(|j| go(child[i] + j, t, child, ordinal, ht, arity, swap)).collect();
            if kids.is_empty() { format!("{}", t[i]) } else { format!("{}({})", t[i], kids.join(",")) }
        }
        Some(go(0, tokens, &child, &ordinal, ht, &arity, swap))
    }

    #[test]
    fn cleanse_unwraps_a_function_layer() {
        let (codes, layout) = (codes(), Layout::for_arity(1, 1, 4, 2, 10));
        let mut gene = vec![3, 3, 4, 5, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0]; // f(f(x4))
        assert!(cleanse_gene(&mut gene, layout, &codes, 0, false, 0, 0));
        assert_eq!(show(&gene, layout, &codes, None).unwrap(), "3(4)");
    }

    #[test]
    fn cleanse_keeps_each_surviving_constant_on_its_own_value() {
        let (codes, layout) = (codes(), Layout::for_arity(1, 1, 4, 2, 10));
        // +(?7, *(?8, ?9)) — head 4, tail 5, then the Dc domain 7, 8, 9.
        let gene = vec![0, 6, 1, 6, 6, 4, 4, 4, 4, 7, 8, 9, 0, 0];
        let mut promoted = gene.clone();
        assert!(cleanse_gene(&mut promoted, layout, &codes, 0, false, u64::MAX, 0)); // root -> its second child
        assert_eq!(show(&promoted, layout, &codes, None).unwrap(), "1(?8,?9)");
        let mut collapsed = gene.clone();
        assert!(cleanse_gene(&mut collapsed, layout, &codes, u64::MAX, true, 0, 0)); // the `*` subtree -> a constant
        assert_eq!(show(&collapsed, layout, &codes, None).unwrap(), "0(?7,?0)");
        let mut dropped = gene;
        assert!(cleanse_gene(&mut dropped, layout, &codes, 0, true, 0, u64::MAX)); // the root -> the whole gene a constant
        assert_eq!(show(&dropped, layout, &codes, None).unwrap(), "?9");
    }

    /// On thousands of random genes: a cleanse that applies gives exactly the
    /// old tree with the chosen subtree replaced (read back by an independent
    /// decoder), strictly smaller, still a valid gene; one that does not apply
    /// changes nothing.
    #[test]
    fn cleanse_is_the_tree_edit_it_claims_to_be_on_random_genes() {
        let codes = codes();
        let layout = Layout::for_arity(3000, 1, 12, 2, 10);
        let pop = init(layout, &codes, &params(21)).unwrap();
        let width = layout.gene_width() as usize;
        let ht = (layout.head + layout.tail) as usize;
        let arity = |id: u32| codes.arity[id as usize] as usize;
        let (mut applied, mut refused) = (0, 0);
        for (r, original) in pop.genome.chunks(width).enumerate() {
            let d = |slot| draw(21, 1, r as u32, slot, 77);
            let (pick, collapse, which, value) = (d(0), coin(d(1)), d(2), d(3));
            let mut gene = original.to_vec();
            let before = show(original, layout, &codes, None);
            if !cleanse_gene(&mut gene, layout, &codes, pick, collapse, which, value) {
                assert_eq!(gene, original, "row {r}: a refused cleanse changed the gene");
                refused += 1;
                continue;
            }
            applied += 1;
            // the node the draws name, found the way the operator documents it
            let n = { let (mut need, mut n) = (1i64, 0usize); while need > 0 { need += arity(original[n]) as i64 - 1; n += 1; } n };
            let functions: Vec<usize> = (0..n).filter(|&i| arity(original[i]) > 0).collect();
            let p = functions[below(pick, functions.len() as u32) as usize];
            let first_child = 1 + (0..p).map(|i| arity(original[i])).sum::<usize>();
            let q = if collapse { None } else { Some(first_child + below(which, arity(original[p]) as u32) as usize) };
            let expected = show(original, layout, &codes, Some((p, q, below(value, layout.n_rnc))));
            assert_eq!(show(&gene, layout, &codes, None), expected, "row {r}: {before:?}");
            assert!(gene[layout.head as usize..ht].iter().all(|&id| arity(id) == 0), "row {r}: a function in the tail");
            assert!(gene[ht..].iter().all(|&k| k < layout.n_rnc), "row {r}: a Dc index out of range");
        }
        // About half of all random genes are a bare terminal (a head slot is a
        // terminal with probability 1/2): nothing to cleanse, so they are refused.
        assert!(applied > 1000 && refused > 0, "applied {applied}, refused {refused}");
    }

    // -----------------------------------------------------------------------
    // THE BEAM'S NEIGHBOURHOOD.
    // -----------------------------------------------------------------------

    fn beam_params(seed: u32, genes: u32) -> BeamParams {
        BeamParams { seed, generation: 17, rnc_lo: -5, rnc_hi: 5, vhead: 0, genes }
    }

    /// One row of a drawn population: its genome and its constants.
    fn one_row(layout: Layout, seed: u32) -> (Vec<u32>, Vec<f32>) {
        let pop = init(layout, &codes(), &params(seed)).unwrap();
        let (row_w, rnc_w) = ((layout.n_genes * layout.gene_width()) as usize, (layout.n_genes * layout.n_rnc) as usize);
        (pop.genome[..row_w].to_vec(), pop.rnc[..rnc_w].to_vec())
    }

    /// A neighbourhood is `width` mutants, every one DIFFERENT from the original
    /// and from every other, every one a valid gene, and the same seed gives the
    /// same set.
    #[test]
    fn the_neighbourhood_is_distinct_valid_and_reproducible() {
        let (codes, layout) = (codes(), Layout::for_arity(64, 3, 12, 2, 10));
        let (genome, rnc) = one_row(layout, 31);
        let p = beam_params(31, 0b111);
        let out = neighbourhood(&genome, &rnc, layout, &codes, &p, 400).unwrap();
        assert!(out.len() > 50, "only {} mutants", out.len());
        // distinct from the original, and from each other
        for m in &out {
            assert!(m.genome != genome || m.rnc != rnc, "a mutant is the original");
        }
        let mut keys: Vec<(Vec<u32>, Vec<u32>)> = out.iter().map(|m| (m.genome.clone(), m.rnc.iter().map(|v| v.to_bits()).collect())).collect();
        let before = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), before, "the neighbourhood repeated a mutant");
        // every one passes the structural rules, as a population row must
        for m in &out {
            let pop = Population { layout: Layout { pop: 1, ..layout }, genome: m.genome.clone(), rnc: m.rnc.clone(), wrapper_id: vec![0] };
            pop.check(&codes).expect("a mutant broke the gene rules");
        }
        // reproducible from the seed, and another beat is another neighbourhood
        assert_eq!(out, neighbourhood(&genome, &rnc, layout, &codes, &p, 400).unwrap());
        let other = neighbourhood(&genome, &rnc, layout, &codes, &BeamParams { generation: 18, ..p }, 400).unwrap();
        assert_ne!(out, other, "two beats gave the same neighbourhood");
    }

    /// The CLEANSE NEIGHBOURHOOD is enumerated WHOLE: for a gene with a known
    /// tree, every function node promoted over each child is there, by name. The
    /// gene is `*( *(x4, x5), f(x6) )` — promoting the root's first child deletes
    /// the `f(x6)` factor, which is the "remove the spurious factor" move.
    #[test]
    fn every_promotion_of_every_function_node_is_in_the_neighbourhood() {
        let (codes, layout) = (codes(), Layout::for_arity(1, 1, 8, 2, 10));
        // ids: 0,1,2 arity 2; 3 arity 1; 4,5 plain terminals (6 is "?"). The gene
        // is `*( *(x4, x5), f(x4) )` in Karva: 0 1 3 4 5 4, the trailing `3(4)` the
        // SPURIOUS factor the promotion of the root's first child deletes.
        let mut genome = vec![4u32; (layout.head + layout.tail) as usize];
        genome[..6].copy_from_slice(&[0, 1, 3, 4, 5, 4]);
        genome.extend(std::iter::repeat_n(0u32, layout.tail as usize));
        let rnc = vec![1.0f32; layout.n_rnc as usize];
        let p = beam_params(5, 0b1);
        let out = neighbourhood(&genome, &rnc, layout, &codes, &p, 5000).unwrap();
        let shown: Vec<String> = out.iter().filter(|m| m.kind == BeamKind::Promote).filter_map(|m| show(&m.genome, layout, &codes, None)).collect();
        assert_eq!(show(&genome, layout, &codes, None).unwrap(), "0(1(4,5),3(4))", "the gene under test");
        // the root over each child: the spurious 3(4) factor deleted, or the other
        for want in ["1(4,5)", "3(4)"] {
            assert!(shown.iter().any(|s| s == want), "promotion to {want} is missing from {shown:?}");
        }
        // the inner `1` over each of its children, and the unary `3` over its own
        for want in ["0(4,3(4))", "0(5,3(4))", "0(1(4,5),4)"] {
            assert!(shown.iter().any(|s| s == want), "promotion giving {want} is missing from {shown:?}");
        }
        // and the collapses are there too: a subtree replaced by one "?"
        assert!(out.iter().any(|m| m.kind == BeamKind::Collapse), "no collapse in the neighbourhood");
    }

    /// A mutation is only ever offered on a gene the model USES: with one gene in
    /// `genes`, no other gene's tokens or constants ever move.
    #[test]
    fn the_neighbourhood_leaves_the_unused_genes_alone() {
        let (codes, layout) = (codes(), Layout::for_arity(64, 3, 12, 2, 10));
        let (genome, rnc) = one_row(layout, 41);
        let (width, nr) = (layout.gene_width() as usize, layout.n_rnc as usize);
        let out = neighbourhood(&genome, &rnc, layout, &codes, &beam_params(41, 0b010), 300).unwrap();
        assert!(!out.is_empty());
        for m in &out {
            for g in [0usize, 2] {
                assert_eq!(m.genome[g * width..(g + 1) * width], genome[g * width..(g + 1) * width], "gene {g} moved");
                assert_eq!(m.rnc[g * nr..(g + 1) * nr], rnc[g * nr..(g + 1) * nr], "gene {g}'s constants moved");
            }
        }
    }

    /// A row that does not fit the layout, and a chromosome with no used gene,
    /// are errors — not a guess and not an empty neighbourhood that looks like
    /// "nothing to try".
    #[test]
    fn a_neighbourhood_of_nothing_is_an_error() {
        let (codes, layout) = (codes(), Layout::for_arity(64, 3, 12, 2, 10));
        let (genome, rnc) = one_row(layout, 51);
        assert!(neighbourhood(&genome[..10], &rnc, layout, &codes, &beam_params(51, 0b111), 10).is_err());
        assert!(neighbourhood(&genome, &rnc, layout, &codes, &beam_params(51, 0), 10).is_err());
        // a virtual head outside 2..=head is refused by the same rule as everywhere
        assert!(neighbourhood(&genome, &rnc, layout, &codes, &BeamParams { vhead: 99, ..beam_params(51, 0b111) }, 10).is_err());
    }

    /// The width is a CAP, honoured exactly: asking for few gives few.
    #[test]
    fn the_neighbourhood_honours_its_width() {
        let (codes, layout) = (codes(), Layout::for_arity(64, 3, 12, 2, 10));
        let (genome, rnc) = one_row(layout, 61);
        for width in [1u32, 7, 33] {
            let out = neighbourhood(&genome, &rnc, layout, &codes, &beam_params(61, 0b111), width).unwrap();
            assert_eq!(out.len(), width as usize, "width {width}");
        }
    }

    #[test]
    fn islands_that_do_not_tile_the_population_are_refused() {
        let layout = Layout::for_arity(800, 3, 48, 2, 10);
        assert!(validate(layout, &[Island { lo: 0, hi: 700, elites: 2, tournsize: 7 }]).is_err());
        assert!(validate(layout, &[Island { lo: 0, hi: 800, elites: 1, tournsize: 7 }]).is_err()); // odd offspring
        assert!(validate(layout, &islands()).is_ok());
    }
}
