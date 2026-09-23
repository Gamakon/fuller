// Step 2 — selection and variation on the device. evolve/vary.rs is the CPU
// reference, written to be read beside this file; device.rs asserts the two
// agree bit for bit. mix64.wgsl is concatenated in front.
//
// Four kernels, one dispatch each per generation:
//   first_main      thread per row   -> stage1[row], the first tournament's winner
//                                       for that slot of the replaced deme
//   select_main     thread per row   -> parent[row], the second tournament's
//                                       winner among those slots (or an elite)
//   mutate_main     thread per row   -> the row cloned from its parent, mutated
//   crossover_main  thread per PAIR  -> the pair recombined in place
// A thread writes only its own row (or pair), so there is nothing to race on.

struct Gen {
    pop: u32,
    n_genes: u32,
    head: u32,
    tail: u32,
    n_rnc: u32,
    n_functions: u32,
    n_terminals: u32,
    n_islands: u32,
    seed: u32,
    generation: u32,
    rnc_lo: i32,
    rnc_span: u32,
    // The fourteen variation rates used to live here, one set for the whole
    // population. They are per-island now — see `Island` — so that two lanes can
    // breed under different rules in one dispatch.
    // VIRTUAL ALPS: cohorts at or past this label are one band (0 = cohorts off).
    cohort_merge: u32,
    rnc_id: u32,       // the "?" token, or NONE
    // The virtual head, already resolved by the host (never 0 here).
    vhead: u32,
}

// THE LANE'S OWN RULES. An island carries the fourteen variation rates it
// breeds under, so two islands in one population evolve differently — the swim
// lanes. They are read here exactly as `gp.<rate>` was read before; a run that
// gives every island the same row is the engine it was, byte for byte.
struct Island {
    lo: u32,
    hi: u32,
    elites: u32,
    tournsize: u32,
    // How many rows above the elites are rows the pump has just promoted. They
    // are NOT ranked: see `Island::arrivals` in vary.rs.
    arrivals: u32,
    mut_point: u32,
    invert: u32,
    is_transpose: u32,
    ris_transpose: u32,
    gene_transpose: u32,
    dc_point: u32,
    invert_dc: u32,
    transpose_dc: u32,
    rnc_point: u32,
    cx_one_point: u32,
    cx_two_point: u32,
    cx_gene: u32,
    cleanse: u32,
    cleanse_collapse: u32,
}

@group(0) @binding(0) var<uniform> gp: Gen;
@group(0) @binding(1) var<storage, read> sample_functions: array<u32>;
@group(0) @binding(2) var<storage, read> sample_terminals: array<u32>;
@group(0) @binding(3) var<storage, read> arity: array<u32>;
@group(0) @binding(4) var<storage, read> islands: array<Island>;
@group(0) @binding(5) var<storage, read> genome_now: array<u32>;
@group(0) @binding(6) var<storage, read> rnc_now: array<f32>;
@group(0) @binding(7) var<storage, read> fitness_now: array<f32>;
@group(0) @binding(8) var<storage, read> wrapper_now: array<u32>;
@group(0) @binding(9) var<storage, read_write> parent: array<u32>;
@group(0) @binding(10) var<storage, read_write> genome: array<u32>;
@group(0) @binding(11) var<storage, read_write> rnc: array<f32>;
@group(0) @binding(12) var<storage, read_write> fitness: array<f32>;
@group(0) @binding(13) var<storage, read_write> wrapper_id: array<u32>;
@group(0) @binding(14) var<storage, read_write> stage1: array<u32>;
// VIRTUAL ALPS: the cohort each row belongs to. A row drawn by the pump is
// stamped with the beat it arrived on; a child inherits its parent's stamp. It
// is a LINEAGE cohort, not an age: a fresh mutation of an elder is still an
// elder, which is what keeps young material from meeting one in a tournament.
@group(0) @binding(15) var<storage, read> cohort_now: array<u32>;
@group(0) @binding(16) var<storage, read_write> cohort: array<u32>;

const STREAM_SELECT_1: u32 = 6u;
const STREAM_SELECT_2: u32 = 7u;
const STREAM_MUT_HIT: u32 = 8u;
const STREAM_MUT_KIND: u32 = 9u;
const STREAM_MUT_SYMBOL: u32 = 10u;
const STREAM_OPERATOR: u32 = 11u;
const STREAM_INVERT: u32 = 12u;
const STREAM_IS: u32 = 13u;
const STREAM_RIS: u32 = 14u;
const STREAM_GENE_T: u32 = 15u;
const STREAM_DC_HIT: u32 = 16u;
const STREAM_DC_VALUE: u32 = 17u;
const STREAM_INVERT_DC: u32 = 18u;
const STREAM_TRANSPOSE_DC: u32 = 19u;
const STREAM_RNC_HIT: u32 = 20u;
const STREAM_RNC_VALUE: u32 = 21u;
const STREAM_CX_1P: u32 = 22u;
const STREAM_CX_2P: u32 = 23u;
const STREAM_CX_GENE: u32 = 24u;
const STREAM_CLEANSE: u32 = 25u;
// The champion an arrival breeds with: a UNIFORM draw over the island, not a
// tournament. See `Island::arrivals`.
const STREAM_ARRIVAL_MATE: u32 = 26u;

const OP_INVERT: u32 = 0u;
const OP_IS: u32 = 1u;
const OP_RIS: u32 = 2u;
const OP_GENE_T: u32 = 3u;
const OP_INVERT_DC: u32 = 4u;
const OP_TRANSPOSE_DC: u32 = 5u;
const OP_CX_1P: u32 = 6u;
const OP_CX_2P: u32 = 7u;
const OP_CX_GENE: u32 = 8u;
const OP_CLEANSE: u32 = 9u;

const NONE: u32 = 0xFFFFFFFFu;
const F32_MAX: f32 = 3.4028234e38;
const NAN_BITS: u32 = 0x7FC00000u;

const DOMAIN_LO: u32 = 0x79C19B14u;
const DOMAIN_HI: u32 = 0x6B86E41Au;

fn hash(row: u32, slot: u32, stream: u32) -> vec2<u32> {
    return mix64_3p(vec2<u32>(DOMAIN_LO ^ gp.seed, DOMAIN_HI ^ stream),
                    vec2<u32>(gp.generation, 0u), vec2<u32>(row, 0u), vec2<u32>(slot, 0u));
}

fn below(row: u32, slot: u32, stream: u32, n: u32) -> u32 {
    return mix64_modp(vec2<u32>(DOMAIN_LO ^ gp.seed, DOMAIN_HI ^ stream),
                      vec2<u32>(gp.generation, 0u), vec2<u32>(row, 0u), vec2<u32>(slot, 0u), n);
}

fn coin(row: u32, slot: u32, stream: u32) -> bool {
    return (hash(row, slot, stream).y >> 31u) == 1u;
}

fn chance(row: u32, slot: u32, stream: u32, thr: u32) -> bool {
    return thr == NONE || hash(row, slot, stream).y < thr;
}

// COUPLES FROM THE SAME CENTURY (Andrew). A tournament run by a row of cohort
// `mine` prefers a candidate of the SAME cohort over any other, however fit the
// other is — so a young line is never beaten by a converged elder that merely
// happened to be drawn. Measured on the engine as it was: a 105-row tournament
// had a 6.7e-11 chance of holding no elder, so young material was never
// sampled without one, and 294,000 pump-drawn individuals left no survivors.
//
// Rather than reject and redraw (a loop on the device), the cohort is the FIRST
// sort key and fitness the second: a mismatched candidate simply loses. When a
// tournament holds no match at all the best of the draw still wins, so a cohort
// that has died out costs nothing.
//
// `gp.cohort_merge` is the AGE at which cohorts stop being separate: a line
// older than it joins one band with every other elder, which is Hornby's
// unbounded top layer, while the young stay apart until they have earned their
// place. 0 = off, and then this is the engine as it was.
//
// The age is `generation - label`, not the label: the label says WHEN a line
// arrived and is fixed for ever, so banding on the label alone kept the OLDEST
// cohorts apart permanently and merged the youngest — the opposite of the
// design, and it meant the most refined lines never mingled at all.
fn band_of(row: u32) -> u32 {
    let label = cohort_now[row];
    // A label cannot be in the future, but a resumed or re-labelled row could
    // read as one; treat it as newborn rather than as older than the run.
    var age = 0u;
    if (gp.generation > label) {
        age = gp.generation - label;
    }
    if (age >= gp.cohort_merge) {
        return ELDERS;
    }
    return label;
}

// The band every elder shares. No cohort label can collide with it: a label is a
// generation, and a fit that reached this many generations would have merged
// long before.
const ELDERS: u32 = 0xFFFFFFFFu;

fn same_cohort(a: u32, b: u32) -> bool {
    if (gp.cohort_merge == 0u) {
        return true;
    }
    return band_of(a) == band_of(b);
}

// Fitness as a sort key: lower is fitter, an unevaluated row (NaN) is last. The
// NaN test is on the bits: a float comparison may be optimised away.
fn key(row: u32) -> f32 {
    let f = fitness_now[row];
    if ((bitcast<u32>(f) & 0x7FFFFFFFu) > 0x7F800000u) {
        return F32_MAX;
    }
    return min(f, F32_MAX);
}

// Is `c` a better mate for `me` than `w`? Same cohort first, then fitness — but
// ONLY on an intake island.
//
// THE CHAMPION ISLAND IS AN OPEN KNOCKOUT (Andrew: 'the champion island the no
// restriction open knockout island'). The cohort rule exists to keep a young
// line from meeting a converged elder before it has developed, and the intake is
// where that development happens. The champion island is the opposite: it is
// where a line that has already earned promotion proves it against everything
// else there, so banding it fragments the one place whose whole purpose is
// competition. Measured on strogatz_bacres1: cohort 0 held the best score for
// 1,558 generations, on both islands, because it never met a challenger on
// either.
//
// A promotion is a COPY, so a line that goes up stays in the intake too — it
// keeps its protected place while its copy takes its chances in the open.
fn better_mate(me: u32, c: u32, w: u32, open_fight: bool) -> bool {
    if (!open_fight) {
        let mc = same_cohort(me, c);
        let mw = same_cohort(me, w);
        if (mc != mw) {
            return mc;
        }
    }
    return key(c) < key(w);
}

// Pair p is islands 2p (intake) and 2p + 1 (champion), one pair after another —
// so an ODD island index is a champion island, and that is where the cohort rule
// is lifted.
fn is_champion(row: u32) -> bool {
    for (var i = 0u; i < gp.n_islands; i = i + 1u) {
        if (row < islands[i].hi) {
            return (i & 1u) == 1u;
        }
    }
    return (gp.n_islands - 1u & 1u) == 1u;
}

fn island_of(row: u32) -> Island {
    for (var i = 0u; i < gp.n_islands; i = i + 1u) {
        if (row < islands[i].hi) {
            return islands[i];
        }
    }
    return islands[gp.n_islands - 1u];
}

// The first tournament, one slot per thread. Materialised in its own pass so
// the second tournament reads a winner instead of replaying a tournament for
// every draw: O(n*k) work for the generation, not O(n*k*k).
@compute @workgroup_size(64)
fn first_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= gp.pop) {
        return;
    }
    let isl = island_of(row);
    let open_fight = is_champion(row);
    let n = isl.hi - isl.lo;
    var w = NONE;
    for (var i = 0u; i < isl.tournsize; i = i + 1u) {
        let c = isl.lo + below(row, i, STREAM_SELECT_1, n);
        if (w == NONE || better_mate(row, c, w, open_fight)) {
            w = c;
        }
    }
    stage1[row] = w;
}

@compute @workgroup_size(64)
fn select_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= gp.pop) {
        return;
    }
    let isl = island_of(row);
    let open_fight = is_champion(row);
    let n = isl.hi - isl.lo;
    if (row < isl.lo + isl.elites) {
        // The j-th fittest row of the island, ties to the lower row.
        let j = row - isl.lo;
        var taken: array<u32, 8>;
        var best = NONE;
        for (var e = 0u; e <= j; e = e + 1u) {
            best = NONE;
            for (var r = isl.lo; r < isl.hi; r = r + 1u) {
                var used = false;
                for (var q = 0u; q < e; q = q + 1u) {
                    if (taken[q] == r) {
                        used = true;
                    }
                }
                if (!used && (best == NONE || key(r) < key(best))) {
                    best = r;
                }
            }
            taken[e] = best;
        }
        parent[row] = best;
        return;
    }
    // THE ARRIVAL BAND, and it is not a tournament at all.
    //
    // The pump lands its promotions contiguously just above the elites. Those
    // rows and the rows beside them are paired by the crossover pass (which
    // takes `a = b - 1`), so laying them out in couples makes every arrival
    // breed with a champion by construction:
    //
    //   arrival, random champion, arrival, random champion, ...
    //
    // The arrival keeps ITSELF as its parent -- it is not ranked, because a
    // freshly promoted line loses every tournament it is drawn into and its
    // genes would never reach a crossover. Its partner is drawn UNIFORMLY over
    // the island rather than won, because the point is DIVERSITY: a tournament
    // returns the converged elder, and breeding the newcomer with it drags the
    // children back into the basin the island is already stuck in.
    let band = isl.lo + isl.elites;
    if (isl.arrivals > 0u && row >= band && row < band + isl.arrivals * 2u) {
        if (((row - band) & 1u) == 0u) {
            parent[row] = row;        // the arrival itself, unranked
        } else {
            parent[row] = isl.lo + below(row, 0u, STREAM_ARRIVAL_MATE, n);
        }
        return;
    }

    var w = NONE;
    for (var t = 0u; t < isl.tournsize; t = t + 1u) {
        let c = stage1[isl.lo + below(row, t, STREAM_SELECT_2, n)];
        if (w == NONE || better_mate(row, c, w, open_fight)) {
            w = c;
        }
    }
    parent[row] = w;
}

fn at(base: u32, g: u32, pos: u32) -> u32 {
    return base + g * (gp.head + 2u * gp.tail) + pos;
}

fn reverse(lo: u32, hi: u32) {
    // reverse genome[lo .. hi)
    var a = lo;
    var b = hi;
    loop {
        if (a + 1u >= b) {
            break;
        }
        b = b - 1u;
        let held = genome[a];
        genome[a] = genome[b];
        genome[b] = held;
        a = a + 1u;
    }
}

var<private> segment: array<u32, 512>;
var<private> before: array<u32, 512>;

// ---- the cleansing mutation (vary.rs :: cleanse_gene, statement for statement) ----
const MAX_CLEANSE: u32 = 128u;
const LEAF: u32 = 0xFFFFFFFEu;
var<private> cl_child: array<u32, 128>;
var<private> cl_ordinal: array<u32, 128>;
var<private> cl_queue: array<u32, 128>;
var<private> cl_tok: array<u32, 128>;
var<private> cl_dc: array<u32, 128>;

// `gene`: the index in `genome` of this gene's first token. The draws are the
// row's, slots 1..4 of STREAM_CLEANSE.
// `collapse_thr` is the row's OWN island rate: a helper cannot see `isl`.
fn cleanse_gene(row: u32, gene: u32, collapse_thr: u32) {
    let h = gp.vhead;
    let ht = gp.head + gp.tail;
    if (ht > MAX_CLEANSE) {
        return;
    }
    var need: i32 = 1;
    var n = 0u;
    loop {
        if (need <= 0 || n >= ht) {
            break;
        }
        need = need + i32(arity[genome[gene + n]]) - 1;
        n = n + 1u;
    }
    if (need > 0) {
        return;
    }
    var next_child = 1u;
    var n_rnc = 0u;
    var n_fn = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        let tok = genome[gene + i];
        cl_child[i] = next_child;
        next_child = next_child + arity[tok];
        cl_ordinal[i] = NONE;
        if (gp.rnc_id != NONE && tok == gp.rnc_id) {
            cl_ordinal[i] = n_rnc;
            n_rnc = n_rnc + 1u;
        }
        if (arity[tok] > 0u) {
            n_fn = n_fn + 1u;
        }
    }
    if (n_fn == 0u) {
        return;
    }
    let nth = below(row, 1u, STREAM_CLEANSE, n_fn);
    var p = 0u;
    var seen = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if (arity[genome[gene + i]] > 0u) {
            if (seen == nth) {
                p = i;
            }
            seen = seen + 1u;
        }
    }
    let collapse = gp.rnc_id != NONE && chance(row, 2u, STREAM_CLEANSE, collapse_thr);
    var q = LEAF;
    if (!collapse) {
        q = cl_child[p] + below(row, 3u, STREAM_CLEANSE, arity[genome[gene + p]]);
    }
    let value = below(row, 4u, STREAM_CLEANSE, gp.n_rnc);

    cl_queue[0] = select(0u, q, p == 0u);
    var head = 0u;
    var tail = 1u;
    var m = 0u;
    var k = 0u;
    loop {
        if (head >= tail) {
            break;
        }
        let i = cl_queue[head];
        head = head + 1u;
        if (i == LEAF) {
            cl_tok[m] = gp.rnc_id;
            cl_dc[k] = value;
            k = k + 1u;
            m = m + 1u;
            continue;
        }
        let tok = genome[gene + i];
        cl_tok[m] = tok;
        if (cl_ordinal[i] != NONE) {
            cl_dc[k] = select(0u, genome[gene + ht + cl_ordinal[i]], cl_ordinal[i] < gp.tail);
            k = k + 1u;
        }
        m = m + 1u;
        for (var j = 0u; j < arity[tok]; j = j + 1u) {
            let c = cl_child[i] + j;
            cl_queue[tail] = select(c, q, c == p);
            tail = tail + 1u;
        }
    }
    if (k > gp.tail) {
        return;
    }
    for (var x = h; x < m; x = x + 1u) {
        if (arity[cl_tok[x]] > 0u) {
            return;
        }
    }
    for (var x = 0u; x < m; x = x + 1u) {
        genome[gene + x] = cl_tok[x];
    }
    for (var x = 0u; x < k; x = x + 1u) {
        genome[gene + ht + x] = cl_dc[x];
    }
}

@compute @workgroup_size(64)
fn mutate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= gp.pop) {
        return;
    }
    let isl = island_of(row);
    let h = gp.head;
    let vh = gp.vhead;
    let t = gp.tail;
    let ht = h + t;
    let width = h + 2u * t;
    let row_w = gp.n_genes * width;
    let rnc_w = gp.n_genes * gp.n_rnc;
    let src = parent[row];
    let base = row * row_w;
    let rbase = row * rnc_w;
    for (var i = 0u; i < row_w; i = i + 1u) {
        genome[base + i] = genome_now[src * row_w + i];
    }
    for (var i = 0u; i < rnc_w; i = i + 1u) {
        rnc[rbase + i] = rnc_now[src * rnc_w + i];
    }
    wrapper_id[row] = wrapper_now[src];
    // A child IS of its parent's cohort: the label follows the line, not the
    // age, so a fresh mutation of an elder is still an elder.
    cohort[row] = cohort_now[src];
    if (row < isl.lo + isl.elites) {
        fitness[row] = fitness_now[src];
        return;
    }
    fitness[row] = bitcast<f32>(NAN_BITS);

    // 1. uniform point mutation
    for (var g = 0u; g < gp.n_genes; g = g + 1u) {
        for (var pos = 0u; pos < ht; pos = pos + 1u) {
            let slot = g * width + pos;
            if (!chance(row, slot, STREAM_MUT_HIT, isl.mut_point)) {
                continue;
            }
            if (pos < vh && coin(row, slot, STREAM_MUT_KIND)) {
                genome[at(base, g, pos)] = sample_functions[below(row, slot, STREAM_MUT_SYMBOL, gp.n_functions)];
            } else {
                genome[at(base, g, pos)] = sample_terminals[below(row, slot, STREAM_MUT_SYMBOL, gp.n_terminals)];
            }
        }
    }
    // 2. inversion, inside one head
    if (chance(row, OP_INVERT, STREAM_OPERATOR, isl.invert)) {
        let g = below(row, 0u, STREAM_INVERT, gp.n_genes);
        let len = 2u + below(row, 1u, STREAM_INVERT, vh - 1u);
        let start = below(row, 2u, STREAM_INVERT, vh - len + 1u);
        reverse(at(base, g, start), at(base, g, start + len));
    }
    // 3. IS transposition
    if (chance(row, OP_IS, STREAM_OPERATOR, isl.is_transpose)) {
        let donor = below(row, 0u, STREAM_IS, gp.n_genes);
        let donee = below(row, 1u, STREAM_IS, gp.n_genes);
        let len = 1u + below(row, 2u, STREAM_IS, vh - 1u);
        let start = below(row, 3u, STREAM_IS, ht - len + 1u);
        let ins = 1u + below(row, 4u, STREAM_IS, vh - len);
        for (var i = 0u; i < len; i = i + 1u) {
            segment[i] = genome[at(base, donor, start + i)];
        }
        for (var i = 0u; i < vh; i = i + 1u) {
            before[i] = genome[at(base, donee, i)];
        }
        for (var pos = ins + len; pos < vh; pos = pos + 1u) {
            genome[at(base, donee, pos)] = before[pos - len];
        }
        for (var i = 0u; i < len; i = i + 1u) {
            genome[at(base, donee, ins + i)] = segment[i];
        }
    }
    // 4. RIS transposition
    if (chance(row, OP_RIS, STREAM_OPERATOR, isl.ris_transpose)) {
        for (var trial = 0u; trial < 2u * gp.n_genes + 1u; trial = trial + 1u) {
            let donor = below(row, trial * 4u, STREAM_RIS, gp.n_genes);
            let donee = below(row, trial * 4u + 1u, STREAM_RIS, gp.n_genes);
            var n_fn = 0u;
            for (var pos = 0u; pos < vh; pos = pos + 1u) {
                if (arity[genome[at(base, donor, pos)]] > 0u) {
                    n_fn = n_fn + 1u;
                }
            }
            if (n_fn == 0u) {
                continue;
            }
            let pick = below(row, trial * 4u + 2u, STREAM_RIS, n_fn);
            var start = 0u;
            var seen = 0u;
            for (var pos = 0u; pos < vh; pos = pos + 1u) {
                if (arity[genome[at(base, donor, pos)]] > 0u) {
                    if (seen == pick) {
                        start = pos;
                    }
                    seen = seen + 1u;
                }
            }
            let len = 2u + below(row, trial * 4u + 3u, STREAM_RIS, min(vh, ht - start) - 1u);
            for (var i = 0u; i < len; i = i + 1u) {
                segment[i] = genome[at(base, donor, start + i)];
            }
            for (var i = 0u; i < vh; i = i + 1u) {
                before[i] = genome[at(base, donee, i)];
            }
            for (var pos = len; pos < vh; pos = pos + 1u) {
                genome[at(base, donee, pos)] = before[pos - len];
            }
            for (var i = 0u; i < len; i = i + 1u) {
                genome[at(base, donee, i)] = segment[i];
            }
            break;
        }
    }
    // 5. gene transposition — the constants go with the gene
    if (gp.n_genes > 1u && chance(row, OP_GENE_T, STREAM_OPERATOR, isl.gene_transpose)) {
        let source = 1u + below(row, 0u, STREAM_GENE_T, gp.n_genes - 1u);
        for (var pos = 0u; pos < width; pos = pos + 1u) {
            let held = genome[at(base, 0u, pos)];
            genome[at(base, 0u, pos)] = genome[at(base, source, pos)];
            genome[at(base, source, pos)] = held;
        }
        for (var k = 0u; k < gp.n_rnc; k = k + 1u) {
            let held = rnc[rbase + k];
            rnc[rbase + k] = rnc[rbase + source * gp.n_rnc + k];
            rnc[rbase + source * gp.n_rnc + k] = held;
        }
    }
    // 6. Dc point mutation
    for (var g = 0u; g < gp.n_genes; g = g + 1u) {
        for (var pos = ht; pos < width; pos = pos + 1u) {
            let slot = g * width + pos;
            if (chance(row, slot, STREAM_DC_HIT, isl.dc_point)) {
                genome[at(base, g, pos)] = below(row, slot, STREAM_DC_VALUE, gp.n_rnc);
            }
        }
    }
    // 7. Dc inversion: geppy chooses len elements and reverses len - 1
    if (t >= 2u && chance(row, OP_INVERT_DC, STREAM_OPERATOR, isl.invert_dc)) {
        let g = below(row, 0u, STREAM_INVERT_DC, gp.n_genes);
        let len = 2u + below(row, 1u, STREAM_INVERT_DC, t - 1u);
        let start = ht + below(row, 2u, STREAM_INVERT_DC, t - len + 1u);
        reverse(at(base, g, start), at(base, g, start + len - 1u));
    }
    // 8. Dc transposition
    if (chance(row, OP_TRANSPOSE_DC, STREAM_OPERATOR, isl.transpose_dc)) {
        let donor = below(row, 0u, STREAM_TRANSPOSE_DC, gp.n_genes);
        let donee = below(row, 1u, STREAM_TRANSPOSE_DC, gp.n_genes);
        let len = 1u + below(row, 2u, STREAM_TRANSPOSE_DC, t);
        let start = ht + below(row, 3u, STREAM_TRANSPOSE_DC, t - len + 1u);
        let ins = ht + below(row, 4u, STREAM_TRANSPOSE_DC, t - len + 1u);
        for (var i = 0u; i < len; i = i + 1u) {
            segment[i] = genome[at(base, donor, start + i)];
        }
        for (var i = 0u; i < t; i = i + 1u) {
            before[i] = genome[at(base, donee, ht + i)];
        }
        for (var pos = ins + len; pos < width; pos = pos + 1u) {
            genome[at(base, donee, pos)] = before[pos - len - ht];
        }
        for (var i = 0u; i < len; i = i + 1u) {
            genome[at(base, donee, ins + i)] = segment[i];
        }
    }
    // 9. the constants themselves
    for (var k = 0u; k < rnc_w; k = k + 1u) {
        if (chance(row, k, STREAM_RNC_HIT, isl.rnc_point)) {
            rnc[rbase + k] = f32(gp.rnc_lo + i32(below(row, k, STREAM_RNC_VALUE, gp.rnc_span)));
        }
    }
    // 10. cleanse: shrink one gene's expression
    if (chance(row, OP_CLEANSE, STREAM_OPERATOR, isl.cleanse)) {
        let g = below(row, 0u, STREAM_CLEANSE, gp.n_genes);
        cleanse_gene(row, base + g * width, isl.cleanse_collapse);
    }
}

fn swap_tokens(a: u32, b: u32, g: u32, from_pos: u32, to_pos: u32) {
    let width = gp.head + 2u * gp.tail;
    let row_w = gp.n_genes * width;
    for (var pos = from_pos; pos < to_pos; pos = pos + 1u) {
        let ia = a * row_w + g * width + pos;
        let ib = b * row_w + g * width + pos;
        let held = genome[ia];
        genome[ia] = genome[ib];
        genome[ib] = held;
    }
}

fn swap_consts(a: u32, ga: u32, b: u32, gb: u32) {
    let rnc_w = gp.n_genes * gp.n_rnc;
    for (var k = 0u; k < gp.n_rnc; k = k + 1u) {
        let ia = a * rnc_w + ga * gp.n_rnc + k;
        let ib = b * rnc_w + gb * gp.n_rnc + k;
        let held = rnc[ia];
        rnc[ia] = rnc[ib];
        rnc[ib] = held;
    }
}

// Thread per row; only the SECOND row of an offspring pair does any work.
@compute @workgroup_size(64)
fn crossover_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= gp.pop) {
        return;
    }
    let isl = island_of(b);
    let first = isl.lo + isl.elites;
    if (b <= first || ((b - first) & 1u) == 0u) {
        return;
    }
    let a = b - 1u;
    let width = gp.head + 2u * gp.tail;
    let row_w = gp.n_genes * width;
    // AN ARRIVAL PAIR ALWAYS CROSSES. The ordinary rates are 0.3 / 0.2 / 0.1 a
    // pair, so left to chance an arrival would breed with its champion only
    // about half the time and otherwise pass into the next generation as a
    // copy of itself -- which is the starvation this band exists to end. The
    // pump promoted this line on purpose; the splice is the whole point of
    // promoting it.
    let arrival_pair = isl.arrivals > 0u && b < first + isl.arrivals * 2u;
    if (arrival_pair || chance(b, OP_CX_1P, STREAM_OPERATOR, isl.cx_one_point)) {
        let g = below(b, 0u, STREAM_CX_1P, gp.n_genes);
        let point = below(b, 1u, STREAM_CX_1P, width);
        for (var whole = 0u; whole < g; whole = whole + 1u) {
            swap_tokens(a, b, whole, 0u, width);
            swap_consts(a, whole, b, whole);
        }
        swap_tokens(a, b, g, 0u, point + 1u);
    }
    if (chance(b, OP_CX_2P, STREAM_OPERATOR, isl.cx_two_point)) {
        let x = below(b, 0u, STREAM_CX_2P, gp.n_genes);
        let y = below(b, 1u, STREAM_CX_2P, gp.n_genes);
        let g1 = min(x, y);
        let g2 = max(x, y);
        let p1 = below(b, 2u, STREAM_CX_2P, width);
        let p2 = below(b, 3u, STREAM_CX_2P, width);
        if (g1 == g2) {
            swap_tokens(a, b, g1, min(p1, p2), max(p1, p2) + 1u);
        } else {
            for (var whole = g1 + 1u; whole < g2; whole = whole + 1u) {
                swap_tokens(a, b, whole, 0u, width);
                swap_consts(a, whole, b, whole);
            }
            swap_tokens(a, b, g1, p1, width);
            swap_tokens(a, b, g2, 0u, p2 + 1u);
        }
    }
    if (chance(b, OP_CX_GENE, STREAM_OPERATOR, isl.cx_gene)) {
        let ga = below(b, 0u, STREAM_CX_GENE, gp.n_genes);
        let gb = below(b, 1u, STREAM_CX_GENE, gp.n_genes);
        for (var pos = 0u; pos < width; pos = pos + 1u) {
            let ia = a * row_w + ga * width + pos;
            let ib = b * row_w + gb * width + pos;
            let held = genome[ia];
            genome[ia] = genome[ib];
            genome[ib] = held;
        }
        swap_consts(a, ga, b, gb);
    }
}
