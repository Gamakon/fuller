//! Physics-prior mutation GENERATOR — the sibling of denoise, NOT denoise.
//!
//! This is a PURE one-to-many generator. It receives one gene (a `Math`
//! s-expression) and emits a proliferation of physics-shaped candidate genes,
//! each a structural mutation that biases the form toward what real physical
//! laws look like (squared distances, axis-aligned coordinate pairs, inverse
//! powers, stripped wallpaper, ...). See docs/physics_prior_rules.md.
//!
//! It does NOT evaluate, score, or judge fit. No data rows, no tolerance, no
//! R^2. Generation is the only job; HFF + the extrapolation objective select
//! downstream. Mixing evaluation in here would smuggle selection into the
//! generator.
//!
//! Volume: internally we generate ALL distinct candidates (rules x match-sites
//! x composition, deduplicated to a fixpoint, with an internal safety cap).
//! Then, if there are more than the caller wants, we randomly sample `n` of
//! them (deterministic from `seed`). The full internal proliferation always
//! happens first; the cap only limits what is RETURNED.

/// A generated physics-prior candidate. No fitness — pure structure.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The mutated gene as a `Math` s-expression string.
    pub expr: String,
    /// Catalogue rule that produced this edit (e.g. "A1", "A2", "F").
    pub rule: String,
    /// True if this is a structural leap (changes what the gene computes) that
    /// the caller MUST gate on extrapolation, not holdout. False for edits that
    /// only reshape (e.g. axis re-pairing).
    pub speculative: bool,
}

/// Hard internal cap on how many distinct candidates we materialise before
/// sampling. Generation stops once this many distinct exprs exist — keeps a
/// pathological gene from blowing up memory/time. Tunable; well above any
/// caller's `n`.
const INTERNAL_CAP: usize = 2000;

/// How many extra `Math` nodes a generated candidate may add on top of the
/// input's node count, across ALL composed edits. This is the global
/// termination bound for composition (see `generate`): no candidate may exceed
/// `node_count(input) + COMPOSITION_BUDGET` nodes. Generous enough for a handful
/// of stacked physics edits; small enough to forbid runaway nesting towers.
const COMPOSITION_BUDGET: usize = 12;

/// Generate physics-prior candidates for `gene`, then randomly return up to `n`
/// of them.
///
/// * `gene` — a `Math` s-expression string.
/// * `paired_groups` — coordinate axes, e.g. `[["x1","x2"], ["y1","y2"]]`,
///   used by the distance-family rules.
/// * `n` — maximum candidates to RETURN (>= 1). If fewer are generated, all are
///   returned. If more, a uniform random `n` are sampled.
/// * `seed` — makes the random sample reproducible.
///
/// Returns the (sampled) candidates. Generation itself is exhaustive up to the
/// internal cap; the sample is the only thing `n` limits.
pub fn generate(
    gene: &str,
    paired_groups: &[Vec<String>],
    n: usize,
    seed: u64,
) -> Result<Vec<Candidate>, String> {
    let root = parse(gene).ok_or_else(|| format!("could not parse {gene:?}"))?;

    // GLOBAL termination bound. Individual rules guard against re-firing on their
    // OWN output, but distinct speculative structure-builders (A2, A3, C2, D1,
    // E1, and the trig identities) compose multiplicatively — one rule's output
    // is another's input — and can grow a tree without ever repeating an exact
    // string, which the `seen` set alone cannot stop. We therefore cap the
    // absolute node count of any generated form at the original size plus a fixed
    // composition budget. The set of Math trees with bounded node count over a
    // finite constructor alphabet is finite, so generation provably terminates
    // regardless of how rules interact. The budget is generous enough to allow
    // several stacked physics edits (e.g. square a diff, sum a 2nd axis, divide
    // by the squared norm) while ruling out runaway towers.
    let max_nodes = node_count(&root) + COMPOSITION_BUDGET;

    // BFS over reachable mutations. Start from the input; each pass applies
    // every rule at every matching site, feeding new distinct forms back in
    // (composition) until no new form appears, the size bound rejects further
    // growth, or the internal cap is hit.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(root.to_math());
    let mut frontier: Vec<Node> = vec![root];
    // Candidates carry the rule/speculative tag of the edit that *created* them.
    let mut candidates: Vec<Candidate> = Vec::new();

    while let Some(node) = frontier.pop() {
        if candidates.len() >= INTERNAL_CAP {
            break;
        }
        for (mutated, rule, speculative) in one_step_edits(&node, paired_groups) {
            if node_count(&mutated) > max_nodes {
                continue; // global size bound — keeps composition terminating
            }
            let s = mutated.to_math();
            if s == gene || !seen.insert(s.clone()) {
                continue; // skip the input itself and any form already seen
            }
            candidates.push(Candidate { expr: s, rule, speculative });
            frontier.push(mutated);
            if candidates.len() >= INTERNAL_CAP {
                break;
            }
        }
    }

    // Return all if within budget; otherwise uniformly sample `n` deterministically.
    let n = n.max(1);
    if candidates.len() <= n {
        return Ok(candidates);
    }
    Ok(sample(candidates, n, seed))
}

/// Uniform random sample of `n` items, deterministic from `seed`. Partial
/// Fisher–Yates over indices using an inline LCG (no rng crate dependency).
fn sample(mut items: Vec<Candidate>, n: usize, seed: u64) -> Vec<Candidate> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let len = items.len();
    for i in 0..n {
        let j = i + next() % (len - i);
        items.swap(i, j);
    }
    items.truncate(n);
    items
}

// ---------------------------------------------------------------------------
// Rules — each enumerates one-step edits at a single site. Composition is
// handled by the BFS in `generate` (feeding results back through these).
// ---------------------------------------------------------------------------

/// All one-step edits of `node`: for every rule, at every matching site,
/// produce `(mutated_tree, rule_name, speculative)`.
fn one_step_edits(node: &Node, groups: &[Vec<String>]) -> Vec<(Node, String, bool)> {
    let mut out = Vec::new();
    let sites = collect_sites(node);
    for path in &sites {
        let sub = at_path(node, path);
        // The parent constructor (if any) — lets rules avoid re-firing on their
        // own output (e.g. A2 must not square a Sub already inside a Pow2,
        // which would nest Pow2(Pow2(...)) without bound).
        let parent_op = parent_op_at(node, path);
        for (repl, rule, spec) in site_edits(sub, parent_op.as_deref(), groups) {
            out.push((replace_at_path(node, path, &repl), rule, spec));
        }
    }
    out
}

/// The constructor name of the parent of the node at `path`, or None for root.
fn parent_op_at(root: &Node, path: &[usize]) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let parent = at_path(root, &path[..path.len() - 1]);
    if let Node::App(op, _) = parent {
        Some(op.clone())
    } else {
        None
    }
}

/// The edits applicable to a single subtree `sub`, given its `parent_op`.
fn site_edits(sub: &Node, parent_op: Option<&str>, groups: &[Vec<String>]) -> Vec<(Node, String, bool)> {
    let mut out = Vec::new();
    if let Node::App(op, ch) = sub {
        match (op.as_str(), ch.len()) {
            // A1 — cross-coordinate -> axis-aligned pair promotion (reshape).
            ("Sub", 2) => {
                if let (Node::Var(a), Node::Var(b)) = (&ch[0], &ch[1]) {
                    if let (Some((ga, ia)), Some((gb, ib))) =
                        (locate(a, groups), locate(b, groups))
                    {
                        if ga != gb {
                            if let Some(p) = groups[ga].get(ib) {
                                out.push((
                                    Node::App("Sub".into(), vec![Node::Var(a.clone()), Node::Var(p.clone())]),
                                    "A1".into(),
                                    false,
                                ));
                            }
                            if let Some(p) = groups[gb].get(ia) {
                                out.push((
                                    Node::App("Sub".into(), vec![Node::Var(p.clone()), Node::Var(b.clone())]),
                                    "A1".into(),
                                    false,
                                ));
                            }
                        }
                    }
                }
                // A2 — square a difference (structural leap). Do NOT fire if
                // this Sub is already directly wrapped in Pow2, else
                // composition nests Pow2(Pow2(...)) without bound.
                if parent_op != Some("Pow2") {
                    out.push((Node::App("Pow2".into(), vec![sub.clone()]), "A2".into(), true));
                }
            }
            // C4 — additive-inverse fold: a + (Neg b) -> Sub a b. Catalogue C4
            // (Group C symmetry), IDENTITY (true reshape, not a leap). Output is
            // a Sub, which does not re-match this Add arm, so it cannot re-fire
            // on its own output — no nesting guard needed.
            ("Add", 2) => {
                if let Node::App(neg, nb) = &ch[1] {
                    if neg == "Neg" && nb.len() == 1 {
                        out.push((
                            Node::App("Sub".into(), vec![ch[0].clone(), nb[0].clone()]),
                            "C4".into(),
                            false,
                        ));
                    }
                }
                // A3 — sum-of-squared-diffs / Euclidean r^2 extension. Catalogue
                // A3 (Group A): given a + Pow2(Sub p q) where (p,q) is an axis
                // pair, extend the sum with the *next* axis pair's squared diff:
                //   a + Pow2(Sub p q) -> a + Pow2(Sub p q) + Pow2(Sub p' q')
                // SPECULATIVE (manufactures a new distance term). GROWS the tree,
                // so it MUST be bounded: it only fires when the matching next
                // axis pair (p',q') is not ALREADY present anywhere in the Add's
                // operand chain, which caps growth at the number of axis groups.
                if let Some((p2, q2)) = next_axis_pair(&ch[1], groups) {
                    let new_term = Node::App(
                        "Pow2".into(),
                        vec![Node::App("Sub".into(), vec![Node::Var(p2.clone()), Node::Var(q2)])],
                    );
                    if !contains_subtree(sub, &new_term) {
                        out.push((
                            Node::App("Add".into(), vec![sub.clone(), new_term]),
                            "A3".into(),
                            true,
                        ));
                    }
                }
                // D1 — reduced-mass / parallel-resistance template. Catalogue D1
                // (Group D): a sum of two SCALAR quantities a + b -> (a*b)/(a+b),
                // the harmonic/reduced form. SPECULATIVE. This GROWS the tree, so
                // it must not re-fire on its own output. Termination guard: D1
                // fires ONLY when both operands are scalars (Var/Num). Its output
                // is `Div(Mul(a,b), Add(a,b))`; the only Add inside is the
                // denominator `Add(a,b)`, but other rules (E1) may later wrap it
                // in Pow2 — so the scalar-operand restriction (not the parent_op
                // check alone) is what bounds it: a denominator built by other
                // rules is rarely a bare sum of two scalars, and even when it is,
                // the produced numerator `Mul(a,b)` is not a sum, so D1 cannot
                // chain on its own structural output. Physically this is also the
                // right shape: it pairs two simple resistances/masses.
                let both_scalar = matches!(&ch[0], Node::Var(_) | Node::Num(_))
                    && matches!(&ch[1], Node::Var(_) | Node::Num(_));
                let d1_denominator_ctx =
                    matches!(parent_op, Some("Div") | Some("Pow2") | Some("Sqrt"));
                if both_scalar && !d1_denominator_ctx {
                    out.push((
                        Node::App(
                            "Div".into(),
                            vec![
                                Node::App("Mul".into(), vec![ch[0].clone(), ch[1].clone()]),
                                sub.clone(),
                            ],
                        ),
                        "D1".into(),
                        true,
                    ));
                }
            }
            // E4 — power-law: wrap a multiplicative factor in a small power.
            ("Mul", 2) => {
                // F — strip a wallpaper factor (reshape).
                const WALL: &[&str] = &["Sqrt", "Sin", "Cos", "Tan", "Tanh", "Exp", "ProtectedSqrt"];
                if let Node::App(inner, _) = &ch[1] {
                    if WALL.contains(&inner.as_str()) {
                        out.push((ch[0].clone(), "F".into(), false));
                    }
                }
                if let Node::App(inner, _) = &ch[0] {
                    if WALL.contains(&inner.as_str()) {
                        out.push((ch[1].clone(), "F".into(), false));
                    }
                }
                // E1 — inverse-square a factor (leap): a*b -> a / Pow2(b). Do
                // NOT inverse-square a transcendental ENVELOPE factor (Exp/Cos/
                // Sin/...): that is never a physical distance, and composing with
                // the E-family envelope rules (which build Mul(x, Exp(..))) would
                // otherwise loop E6 -> E1 -> E6 unboundedly. We only inverse-square
                // a non-envelope right factor `b`.
                let b_is_envelope =
                    matches!(&ch[1], Node::App(o, _) if WALL.contains(&o.as_str()));
                if !b_is_envelope {
                    out.push((
                        Node::App(
                            "Div".into(),
                            vec![ch[0].clone(), Node::App("Pow2".into(), vec![ch[1].clone()])],
                        ),
                        "E1".into(),
                        true,
                    ));
                }
                // C1 — symmetrise a product of "charges/masses": Mul(Var v, s)
                // where v belongs to an axis pair -> Mul(Var v, Var v') with v'
                // the partner of v (e.g. m1*x -> m1*m2). Catalogue C1 (Group C),
                // SPECULATIVE (m1m2 symmetry). Fires on each side.
                //
                // Termination: C1 only replaces a *scalar* sibling — a bare Var
                // or Num — never a compound subtree. This is both physically
                // right (it pairs two scalar quantities) AND the nesting guard:
                // were C1 allowed to overwrite an arbitrary sibling, it could
                // clobber an E-family envelope factor (Cos x / Exp ..) back into
                // a partner Var, which then re-feeds E1/E-family and grows a
                // Div/Mul tower forever. Restricting to scalar siblings means the
                // output `Mul(Var v, Var p)` has only scalar children; C1 cannot
                // re-fire on it to introduce anything new (the partner already
                // matches), so it terminates.
                let scalar = |n: &Node| matches!(n, Node::Var(_) | Node::Num(_));
                if let Node::Var(v) = &ch[0] {
                    if scalar(&ch[1]) {
                        if let Some(p) = axis_partner(v, groups) {
                            if !matches!(&ch[1], Node::Var(w) if w == &p) {
                                out.push((
                                    Node::App("Mul".into(), vec![ch[0].clone(), Node::Var(p)]),
                                    "C1".into(),
                                    true,
                                ));
                            }
                        }
                    }
                }
                if let Node::Var(v) = &ch[1] {
                    if scalar(&ch[0]) {
                        if let Some(p) = axis_partner(v, groups) {
                            if !matches!(&ch[0], Node::Var(w) if w == &p) {
                                out.push((
                                    Node::App("Mul".into(), vec![Node::Var(p), ch[1].clone()]),
                                    "C1".into(),
                                    true,
                                ));
                            }
                        }
                    }
                }
            }
            // A2 also applies to Add as the (a+b) -> not squared; skip. Div
            // divisor inverse handled below.
            ("Div", 2) => {
                // E1 — promote divisor to its square (inverse-square leap):
                // a / b -> a / Pow2(b). Do NOT fire if the divisor is already
                // Pow2, else it nests a / Pow2(Pow2(...)) without bound.
                let divisor_is_pow2 = matches!(&ch[1], Node::App(op, _) if op == "Pow2");
                if !divisor_is_pow2 {
                    out.push((
                        Node::App("Div".into(), vec![ch[0].clone(), Node::App("Pow2".into(), vec![ch[1].clone()])]),
                        "E1".into(),
                        true,
                    ));
                }
                // A4 — inverse-distance (1/r) from a sum-of-squares denominator.
                // Catalogue A4 (Group A): when the divisor is a sum of squared
                // terms (an r^2 motif), also offer dividing by its Sqrt, i.e.
                // f / (r^2) -> f / Sqrt(r^2) = f / r. SPECULATIVE. Guard: only
                // when the divisor is a sum-of-squares and is NOT already wrapped
                // in Sqrt, so it cannot nest Sqrt(Sqrt(...)).
                let divisor_is_sqrt = matches!(&ch[1], Node::App(op, _) if op == "Sqrt");
                if is_sum_of_squares(&ch[1]) && !divisor_is_sqrt {
                    out.push((
                        Node::App(
                            "Div".into(),
                            vec![ch[0].clone(), Node::App("Sqrt".into(), vec![ch[1].clone()])],
                        ),
                        "A4".into(),
                        true,
                    ));
                }
            }
            // --- Trig identity templates (mined from SymPy sympy/simplify/fu.py) ---
            // These are EXACT trig identities re-expressed as one-step structural
            // mutations. As a generator we offer them in a single direction; the
            // BFS `seen` set already breaks any 2-cycle with the reverse form, so
            // these need no explicit guard UNLESS they monotonically grow the tree
            // in a way `seen` cannot catch (noted per rule).
            ("Sin", 1) => {
                // TR11 (fu.py): double-angle. Sin(2*x) -> 2*Sin(x)*Cos(x).
                // Reshape of an exact identity; the produced term has no Sin(2*x)
                // sub-pattern, so it cannot re-fire on itself.
                if let Some(x) = double_angle_arg(&ch[0]) {
                    out.push((
                        Node::App(
                            "Mul".into(),
                            vec![
                                Node::Num(2.0),
                                Node::App(
                                    "Mul".into(),
                                    vec![
                                        Node::App("Sin".into(), vec![x.clone()]),
                                        Node::App("Cos".into(), vec![x]),
                                    ],
                                ),
                            ],
                        ),
                        "TR11-sin".into(),
                        false,
                    ));
                }
                // TR10 (fu.py): angle-sum. Sin(a+b) -> Sin a Cos b + Cos a Sin b.
                // Exact identity. Could re-fire if a or b are themselves sums,
                // but each firing strictly shrinks the largest Sin/Cos argument's
                // Add-nesting, so it is bounded by argument depth.
                if let Node::App(add, ab) = &ch[0] {
                    if add == "Add" && ab.len() == 2 {
                        let (a, b) = (&ab[0], &ab[1]);
                        out.push((
                            Node::App(
                                "Add".into(),
                                vec![
                                    Node::App(
                                        "Mul".into(),
                                        vec![
                                            Node::App("Sin".into(), vec![a.clone()]),
                                            Node::App("Cos".into(), vec![b.clone()]),
                                        ],
                                    ),
                                    Node::App(
                                        "Mul".into(),
                                        vec![
                                            Node::App("Cos".into(), vec![a.clone()]),
                                            Node::App("Sin".into(), vec![b.clone()]),
                                        ],
                                    ),
                                ],
                            ),
                            "TR10-sin".into(),
                            false,
                        ));
                    }
                }
            }
            ("Cos", 1) => {
                // TR11 (fu.py): double-angle. Cos(2*x) -> Cos(x)^2 - Sin(x)^2.
                if let Some(x) = double_angle_arg(&ch[0]) {
                    out.push((
                        Node::App(
                            "Sub".into(),
                            vec![
                                Node::App("Pow2".into(), vec![Node::App("Cos".into(), vec![x.clone()])]),
                                Node::App("Pow2".into(), vec![Node::App("Sin".into(), vec![x])]),
                            ],
                        ),
                        "TR11-cos".into(),
                        false,
                    ));
                }
                // TR10 (fu.py): angle-sum. Cos(a+b) -> Cos a Cos b - Sin a Sin b.
                if let Node::App(add, ab) = &ch[0] {
                    if add == "Add" && ab.len() == 2 {
                        let (a, b) = (&ab[0], &ab[1]);
                        out.push((
                            Node::App(
                                "Sub".into(),
                                vec![
                                    Node::App(
                                        "Mul".into(),
                                        vec![
                                            Node::App("Cos".into(), vec![a.clone()]),
                                            Node::App("Cos".into(), vec![b.clone()]),
                                        ],
                                    ),
                                    Node::App(
                                        "Mul".into(),
                                        vec![
                                            Node::App("Sin".into(), vec![a.clone()]),
                                            Node::App("Sin".into(), vec![b.clone()]),
                                        ],
                                    ),
                                ],
                            ),
                            "TR10-cos".into(),
                            false,
                        ));
                    }
                }
            }
            // TR5 / TR6 (fu.py): Pythagorean rearrangement, applied to a SQUARED
            // sine or cosine. Pow2(Sin x) -> 1 - Pow2(Cos x)  and the dual
            // Pow2(Cos x) -> 1 - Pow2(Sin x). Exact identities. Together they
            // form a 2-cycle (sin^2 <-> cos^2 via two rewrites), which the BFS
            // `seen` set terminates: the round-trip reproduces the original form,
            // which is already seen and thus skipped. No explicit guard needed.
            ("Pow2", 1) => {
                if let Node::App(f, fa) = &ch[0] {
                    if fa.len() == 1 && (f == "Sin" || f == "Cos") {
                        let dual = if f == "Sin" { "Cos" } else { "Sin" };
                        out.push((
                            Node::App(
                                "Sub".into(),
                                vec![
                                    Node::Num(1.0),
                                    Node::App(
                                        "Pow2".into(),
                                        vec![Node::App(dual.into(), vec![fa[0].clone()])],
                                    ),
                                ],
                            ),
                            "TR5".into(),
                            false,
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    {
        // --- E-family modulation templates (modulate the WHOLE expression) ---
        // E2/E3/E6 wrap the entire function `f` in a physical envelope of one of
        // its variables: f -> f * env(x). Per the catalogue these apply to the
        // whole gene/residual, so they fire ONLY at the root (`parent_op` is
        // None), anchored on the first free variable of `f`. SPECULATIVE.
        //
        // Termination: firing at the root yields `Mul(f, env)`, whose new root is
        // a `Mul` whose right child is an `Exp`/`Cos` envelope. We refuse to fire
        // when `f` is ALREADY such a modulated product (root `Mul` with an
        // Exp/Cos right factor). So at most one envelope is stacked at the root,
        // and the rule cannot re-fire on its own output — bounded.
        if parent_op.is_none() {
            let already_modulated = matches!(
                sub,
                Node::App(o, c)
                    if o == "Mul" && c.len() == 2
                    && matches!(&c[1], Node::App(e, _) if e == "Exp" || e == "Cos")
            );
            if !already_modulated {
                if let Some(x) = first_free_var(sub) {
                    let xv = Node::Var(x);
                    // E2 — exponential decay: f -> f * Exp(Neg x).  Catalogue E2.
                    out.push((
                        Node::App(
                            "Mul".into(),
                            vec![
                                sub.clone(),
                                Node::App(
                                    "Exp".into(),
                                    vec![Node::App("Neg".into(), vec![xv.clone()])],
                                ),
                            ],
                        ),
                        "E2".into(),
                        true,
                    ));
                    // E3 — oscillator: f -> f * Cos x.  Catalogue E3.
                    out.push((
                        Node::App(
                            "Mul".into(),
                            vec![sub.clone(), Node::App("Cos".into(), vec![xv.clone()])],
                        ),
                        "E3".into(),
                        true,
                    ));
                    // E6 — Gaussian: f -> f * Exp(Neg(Pow2 x)).  Catalogue E6.
                    out.push((
                        Node::App(
                            "Mul".into(),
                            vec![
                                sub.clone(),
                                Node::App(
                                    "Exp".into(),
                                    vec![Node::App(
                                        "Neg".into(),
                                        vec![Node::App("Pow2".into(), vec![xv])],
                                    )],
                                ),
                            ],
                        ),
                        "E6".into(),
                        true,
                    ));
                }
            }
        }
        // C2 — even-power sign kill: wrap a sign-bearing DIFFERENCE in Abs
        // (sign symmetry). Catalogue C2 (Group C), SPECULATIVE. We target a `Sub`
        // (the canonical sign-bearing compound) rather than every bare Var, which
        // keeps the proposal physically pointed (|a - b|) and the candidate
        // volume modest. Termination: the output is `Abs(Sub ..)`; the inner Sub
        // now has parent `Abs`, and we refuse to fire when the Sub already sits
        // inside Abs/Pow2/Exp/Cos/Sin (sign already neutralised), so C2 cannot
        // nest `Abs(Abs(..))`.
        if !var_in_wrapper(parent_op) && matches!(sub, Node::App(o, _) if o == "Sub") {
            out.push((Node::App("Abs".into(), vec![sub.clone()]), "C2".into(), true));
        }
    }
    out
}

/// Which (group, index) a variable name belongs to in the paired groups.
fn locate(name: &str, groups: &[Vec<String>]) -> Option<(usize, usize)> {
    for (gi, g) in groups.iter().enumerate() {
        if let Some(idx) = g.iter().position(|v| v == name) {
            return Some((gi, idx));
        }
    }
    None
}

/// True if a leaf sitting directly under `parent_op` is already inside a
/// wrapper that the C2 (Abs) template builds, or a power/transcendental context
/// where its sign is already neutralised. Guards C2 against `Abs(..(Abs(..)))`
/// nesting and against pointlessly wrapping a value whose sign cannot matter.
fn var_in_wrapper(parent_op: Option<&str>) -> bool {
    matches!(
        parent_op,
        Some("Abs") | Some("Pow2") | Some("Exp") | Some("Cos") | Some("Sin")
    )
}

/// Total node count of a tree (every Num/Var/App node counts as 1). Used by the
/// global composition size bound in `generate`.
fn node_count(n: &Node) -> usize {
    match n {
        Node::Num(_) | Node::Var(_) => 1,
        Node::App(_, ch) => 1 + ch.iter().map(node_count).sum::<usize>(),
    }
}

/// The first free variable name encountered in pre-order, if any. Anchors the
/// E2/E3/E6 whole-expression modulation envelopes on an actual variable.
fn first_free_var(n: &Node) -> Option<String> {
    match n {
        Node::Var(v) => Some(v.clone()),
        Node::App(_, ch) => ch.iter().find_map(first_free_var),
        Node::Num(_) => None,
    }
}

/// The same-group partner of a variable in a 2-element axis pair (e.g. the
/// partner of `m1` is `m2`). `None` if the var is not in a 2-element group or
/// has no distinct partner. Used by C1 (mass/charge symmetrisation).
fn axis_partner(name: &str, groups: &[Vec<String>]) -> Option<String> {
    let (gi, idx) = locate(name, groups)?;
    let g = &groups[gi];
    if g.len() != 2 {
        return None;
    }
    let other = g[1 - idx].clone();
    if other == name {
        None
    } else {
        Some(other)
    }
}

/// Given a `Pow2(Sub p q)` term whose `(p, q)` are vars forming an axis pair in
/// the *same column index* of two different groups, return the next group's
/// pair `(p', q')` at those same column indices — the next Euclidean axis to
/// extend a sum-of-squared-differences with. `None` if the term is not a
/// squared difference of paired vars, or there is no further group. Used by A3.
fn next_axis_pair(term: &Node, groups: &[Vec<String>]) -> Option<(String, String)> {
    let Node::App(p2, p2c) = term else { return None };
    if p2 != "Pow2" || p2c.len() != 1 {
        return None;
    }
    let Node::App(sub, sc) = &p2c[0] else { return None };
    if sub != "Sub" || sc.len() != 2 {
        return None;
    }
    let (Node::Var(a), Node::Var(b)) = (&sc[0], &sc[1]) else { return None };
    let (ga, ia) = locate(a, groups)?;
    let (gb, ib) = locate(b, groups)?;
    // p and q must sit at the same column index across two groups (xi - yi).
    if ia != ib {
        return None;
    }
    // Next group after the larger of the two; wrap is not attempted (bounded).
    let next = ga.max(gb) + 1;
    let g = groups.get(next)?;
    let p = g.get(ia)?.clone();
    let q = g.get(ib)?.clone();
    Some((p, q))
}

/// True if `needle` occurs anywhere inside `hay` (structural equality).
fn contains_subtree(hay: &Node, needle: &Node) -> bool {
    if hay == needle {
        return true;
    }
    if let Node::App(_, ch) = hay {
        ch.iter().any(|c| contains_subtree(c, needle))
    } else {
        false
    }
}

/// True if `n` is a sum of squared terms — `Add(.., ..)` whose operands are all
/// `Pow2(..)` (recursing through nested `Add`s). Recognises the r^2 motif for
/// A4 (inverse-distance). A bare `Pow2(..)` alone does NOT count (needs >=2).
fn is_sum_of_squares(n: &Node) -> bool {
    fn all_sq(n: &Node) -> bool {
        match n {
            Node::App(op, ch) if op == "Add" && ch.len() == 2 => {
                all_sq(&ch[0]) && all_sq(&ch[1])
            }
            Node::App(op, ch) if op == "Pow2" && ch.len() == 1 => true,
            _ => false,
        }
    }
    matches!(n, Node::App(op, ch) if op == "Add" && ch.len() == 2) && all_sq(n)
}

/// If `arg` is a double angle `Mul(Num 2, x)` or `Mul(x, Num 2)`, return `x`.
/// Used by the TR11 double-angle templates.
fn double_angle_arg(arg: &Node) -> Option<Node> {
    if let Node::App(op, ch) = arg {
        if op == "Mul" && ch.len() == 2 {
            if matches!(&ch[0], Node::Num(v) if *v == 2.0) {
                return Some(ch[1].clone());
            }
            if matches!(&ch[1], Node::Num(v) if *v == 2.0) {
                return Some(ch[0].clone());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Math tree + path-based zipper (parse / serialise / site collection / replace)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(f64),
    Var(String),
    App(String, Vec<Node>),
}

impl Node {
    fn to_math(&self) -> String {
        match self {
            Node::Num(v) => format!("(Num {})", fmt_f64(*v)),
            Node::Var(n) => format!("(Var \"{n}\")"),
            Node::App(op, ch) => {
                let parts: Vec<String> = ch.iter().map(Node::to_math).collect();
                format!("({op} {})", parts.join(" "))
            }
        }
    }
}

fn fmt_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() { format!("{v:.1}") } else { format!("{v}") }
}

/// All node paths in pre-order (root = empty path).
fn collect_sites(root: &Node) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    fn go(n: &Node, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        out.push(path.clone());
        if let Node::App(_, ch) = n {
            for (i, c) in ch.iter().enumerate() {
                path.push(i);
                go(c, path, out);
                path.pop();
            }
        }
    }
    let mut p = Vec::new();
    go(root, &mut p, &mut out);
    out
}

/// The subtree at `path`.
fn at_path<'a>(root: &'a Node, path: &[usize]) -> &'a Node {
    let mut cur = root;
    for &i in path {
        if let Node::App(_, ch) = cur {
            cur = &ch[i];
        }
    }
    cur
}

/// A copy of `root` with the subtree at `path` replaced by `repl`.
fn replace_at_path(root: &Node, path: &[usize], repl: &Node) -> Node {
    if path.is_empty() {
        return repl.clone();
    }
    match root {
        Node::App(op, ch) => {
            let mut new_ch = ch.clone();
            new_ch[path[0]] = replace_at_path(&ch[path[0]], &path[1..], repl);
            Node::App(op.clone(), new_ch)
        }
        other => other.clone(),
    }
}

fn parse(s: &str) -> Option<Node> {
    let toks = tok(s);
    let mut pos = 0;
    let n = pparse(&toks, &mut pos, 0)?;
    if pos == toks.len() { Some(n) } else { None }
}

fn tok(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => { out.push(c.to_string()); chars.next(); }
            '"' => {
                let mut t = String::from("\"");
                chars.next();
                for c2 in chars.by_ref() { if c2 == '"' { break; } t.push(c2); }
                t.push('"');
                out.push(t);
            }
            c if c.is_whitespace() => { chars.next(); }
            _ => {
                let mut t = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() { break; }
                    t.push(c2);
                    chars.next();
                }
                out.push(t);
            }
        }
    }
    out
}

fn pparse(toks: &[String], pos: &mut usize, depth: usize) -> Option<Node> {
    // Depth cap: a pathologically nested gene must fail the parse (the caller
    // returns an Err), not overflow the stack — every later walk over the tree
    // (sites/replace/to_math/drop) inherits this bound.
    if depth > crate::MAX_EXPR_DEPTH {
        return None;
    }
    if toks.get(*pos)? != "(" { return None; }
    *pos += 1;
    let head = toks.get(*pos)?.clone();
    *pos += 1;
    let node = match head.as_str() {
        "Num" => {
            let v: f64 = toks.get(*pos)?.parse().ok()?;
            // "inf"/"NaN" parse as f64 but render to (Num ..) literals egglog
            // cannot read — refuse them so no candidate is unparseable.
            if !v.is_finite() { return None; }
            *pos += 1;
            Node::Num(v)
        }
        "Var" => { let n = toks.get(*pos)?.trim_matches('"').to_string(); *pos += 1; Node::Var(n) }
        ctor => {
            let mut ch = Vec::new();
            while *pos < toks.len() && toks[*pos] != ")" { ch.push(pparse(toks, pos, depth + 1)?); }
            Node::App(ctor.to_string(), ch)
        }
    };
    if toks.get(*pos)? != ")" { return None; }
    *pos += 1;
    Some(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups() -> Vec<Vec<String>> {
        vec![
            vec!["x1".into(), "x2".into()],
            vec!["y1".into(), "y2".into()],
            vec!["z1".into(), "z2".into()],
        ]
    }

    #[test]
    fn generates_many_from_one_gene() {
        // A modestly complex gene yields a proliferation of candidates.
        let gene = r#"(Div (Var "m1") (Sub (Var "x2") (Var "y1")))"#;
        let all = generate(gene, &groups(), 1000, 0).unwrap();
        assert!(all.len() > 3, "expected a proliferation, got {}", all.len());
        // never returns the input itself
        assert!(all.iter().all(|c| c.expr != gene));
        // all distinct
        let mut exprs: Vec<&str> = all.iter().map(|c| c.expr.as_str()).collect();
        exprs.sort();
        let n_before = exprs.len();
        exprs.dedup();
        assert_eq!(n_before, exprs.len(), "candidates must be distinct");
    }

    #[test]
    fn a1_axis_promotion_present() {
        let gene = r#"(Sub (Var "x2") (Var "y1"))"#;
        let all = generate(gene, &groups(), 1000, 0).unwrap();
        let a1: Vec<&str> = all.iter().filter(|c| c.rule == "A1").map(|c| c.expr.as_str()).collect();
        assert!(a1.contains(&r#"(Sub (Var "x2") (Var "x1"))"#), "{a1:?}");
        assert!(a1.contains(&r#"(Sub (Var "y2") (Var "y1"))"#), "{a1:?}");
    }

    #[test]
    fn a2_square_is_speculative() {
        let gene = r#"(Sub (Var "a") (Var "b"))"#;
        let all = generate(gene, &[], 1000, 0).unwrap();
        let a2 = all.iter().find(|c| c.rule == "A2").expect("A2");
        assert_eq!(a2.expr, r#"(Pow2 (Sub (Var "a") (Var "b")))"#);
        assert!(a2.speculative);
    }

    #[test]
    fn cap_limits_returned_count_not_generation() {
        let gene = r#"(Div (Mul (Var "m1") (Var "m2")) (Sub (Var "x2") (Var "y1")))"#;
        // Ask for 5; must return exactly 5 (generation produced more).
        let five = generate(gene, &groups(), 5, 7).unwrap();
        assert_eq!(five.len(), 5);
        // Deterministic in seed.
        let five_again = generate(gene, &groups(), 5, 7).unwrap();
        assert_eq!(five, five_again, "same seed -> same sample");
        // Different seed generally differs (not asserted strictly), but asking
        // for 1 must give exactly 1.
        assert_eq!(generate(gene, &groups(), 1, 0).unwrap().len(), 1);
    }

    /// Helper: the FULL distinct candidate set (n huge), filtered to one rule.
    fn rule_exprs(gene: &str, groups: &[Vec<String>], rule: &str) -> Vec<String> {
        generate(gene, groups, 1_000_000, 0)
            .unwrap()
            .into_iter()
            .filter(|c| c.rule == rule)
            .map(|c| c.expr)
            .collect()
    }

    fn has(v: &[String], s: &str) -> bool {
        v.iter().any(|e| e == s)
    }

    // --- Boundedness: every composing speculative rule stays well under the
    // internal cap. A runaway rule would flood to INTERNAL_CAP (2000); these
    // assert the generated set is finite and modest for genes that exercise the
    // full rule cross-product (A2/A3/A4/C1/C2/D1/E1/E2/E3/E6 all reachable).
    #[test]
    fn generation_is_bounded_well_under_cap() {
        let genes = [
            r#"(Mul (Var "x2") (Var "x1"))"#,
            r#"(Div (Mul (Var "m1") (Var "x2")) (Sub (Var "x2") (Var "y1")))"#,
            r#"(Add (Pow2 (Sub (Var "x1") (Var "x2"))) (Var "k"))"#,
            r#"(Sin (Add (Var "a") (Var "b")))"#,
            r#"(Cos (Mul (Num 2.0) (Var "a")))"#,
            r#"(Pow2 (Sin (Var "a")))"#,
        ];
        for gene in genes {
            let all = generate(gene, &groups(), 1000, 0).unwrap();
            // Ask for far more than any healthy gene yields: if a rule ran away
            // the INTERNAL generation would hit exactly INTERNAL_CAP distinct
            // forms. Assert we never reach the cap.
            let full = generate(gene, &groups(), 1_000_000, 0).unwrap();
            assert!(
                full.len() < INTERNAL_CAP,
                "gene {gene} flooded to {} (cap {INTERNAL_CAP}) — a rule is unbounded",
                full.len()
            );
            assert!(!all.is_empty());
        }
    }

    #[test]
    fn c4_additive_inverse_fold_reshape() {
        // C4: a + (Neg b) -> Sub a b. IDENTITY reshape (not speculative).
        let gene = r#"(Add (Var "a") (Neg (Var "b")))"#;
        let all = generate(gene, &[], 1000, 0).unwrap();
        let c4 = all.iter().find(|c| c.rule == "C4").expect("C4 present");
        assert_eq!(c4.expr, r#"(Sub (Var "a") (Var "b"))"#);
        assert!(!c4.speculative, "C4 is a reshape");
    }

    #[test]
    fn a3_extends_euclidean_sum_of_squares() {
        // A3: a + Pow2(Sub x1 x2) -> ... + Pow2(Sub <next-group@same-col>).
        // Column 0 of each group is the paired axis across groups.
        let g = vec![
            vec!["x1".to_string(), "y1".into()],
            vec!["x2".into(), "y2".into()],
            vec!["x3".into(), "y3".into()],
        ];
        let gene = r#"(Add (Var "k") (Pow2 (Sub (Var "x1") (Var "x2"))))"#;
        let a3 = rule_exprs(gene, &g, "A3");
        assert!(
            a3.iter().any(|e| e.contains("(Pow2 (Sub (Var \"x3\")")),
            "A3 should append the next axis pair's squared diff: {a3:?}"
        );
        let c = generate(gene, &g, 1_000_000, 0).unwrap();
        assert!(c.iter().find(|c| c.rule == "A3").unwrap().speculative);
    }

    #[test]
    fn a4_inverse_distance_from_sum_of_squares() {
        // A4: f / (Pow2 a + Pow2 b) -> f / Sqrt(Pow2 a + Pow2 b).
        let gene = r#"(Div (Var "q") (Add (Pow2 (Var "x")) (Pow2 (Var "y"))))"#;
        let a4 = rule_exprs(gene, &[], "A4");
        assert!(
            has(&a4, r#"(Div (Var "q") (Sqrt (Add (Pow2 (Var "x")) (Pow2 (Var "y")))))"#),
            "{a4:?}"
        );
    }

    #[test]
    fn c1_symmetrises_mass_product() {
        // C1: m1 * q -> m1 * m2 (partner of m1). Scalar-sibling only.
        let g = vec![vec!["m1".to_string(), "m2".into()]];
        let gene = r#"(Mul (Var "m1") (Var "q"))"#;
        let c1 = rule_exprs(gene, &g, "C1");
        assert!(has(&c1, r#"(Mul (Var "m1") (Var "m2"))"#), "{c1:?}");
        let spec = generate(gene, &g, 1_000_000, 0)
            .unwrap()
            .into_iter()
            .find(|c| c.rule == "C1")
            .unwrap();
        assert!(spec.speculative);
    }

    #[test]
    fn c1_does_not_clobber_compound_sibling() {
        // C1 must NOT overwrite a compound (non-scalar) sibling — that loop was
        // the source of an unbounded Div/Mul tower.
        let g = vec![vec!["m1".to_string(), "m2".into()]];
        let gene = r#"(Mul (Var "m1") (Cos (Var "m1")))"#;
        let c1 = rule_exprs(gene, &g, "C1");
        assert!(c1.is_empty(), "C1 fired on a Cos sibling: {c1:?}");
    }

    #[test]
    fn c2_abs_wraps_difference() {
        // C2: Sub a b -> Abs(Sub a b). SPECULATIVE; never nests Abs(Abs(..)).
        let gene = r#"(Sub (Var "a") (Var "b"))"#;
        let c2 = rule_exprs(gene, &[], "C2");
        assert!(has(&c2, r#"(Abs (Sub (Var "a") (Var "b")))"#), "{c2:?}");
        assert!(c2.iter().all(|e| !e.contains("(Abs (Abs")), "no Abs(Abs) nesting");
    }

    #[test]
    fn d1_reduced_mass_template() {
        // D1: a + b -> (a*b)/(a+b). SPECULATIVE.
        let gene = r#"(Add (Var "r1") (Var "r2"))"#;
        let d1 = rule_exprs(gene, &[], "D1");
        assert!(
            has(&d1, r#"(Div (Mul (Var "r1") (Var "r2")) (Add (Var "r1") (Var "r2")))"#),
            "{d1:?}"
        );
        // Bounded: must not nest ((a*b)/(a+b))/(a+b)... forever.
        let full = generate(gene, &[], 1_000_000, 0).unwrap();
        assert!(full.len() < INTERNAL_CAP);
    }

    #[test]
    fn e2_e3_e6_modulate_whole_expression() {
        // E2/E3/E6 fire at the ROOT only, anchored on a free variable.
        let gene = r#"(Var "x")"#;
        let e2 = rule_exprs(gene, &[], "E2");
        let e3 = rule_exprs(gene, &[], "E3");
        let e6 = rule_exprs(gene, &[], "E6");
        assert!(has(&e2, r#"(Mul (Var "x") (Exp (Neg (Var "x"))))"#), "{e2:?}");
        assert!(has(&e3, r#"(Mul (Var "x") (Cos (Var "x")))"#), "{e3:?}");
        assert!(has(&e6, r#"(Mul (Var "x") (Exp (Neg (Pow2 (Var "x")))))"#), "{e6:?}");
        let full = generate(gene, &[], 1_000_000, 0).unwrap();
        assert!(full.iter().filter(|c| c.rule == "E2").all(|c| c.speculative));
        assert!(full.len() < INTERNAL_CAP);
    }

    #[test]
    fn tr11_double_angle_sin_and_cos() {
        // TR11 (SymPy fu.py): Sin(2x) -> 2 Sin x Cos x ; Cos(2x) -> cos^2 - sin^2.
        let sgene = r#"(Sin (Mul (Num 2.0) (Var "a")))"#;
        let s = rule_exprs(sgene, &[], "TR11-sin");
        assert!(
            has(&s, r#"(Mul (Num 2.0) (Mul (Sin (Var "a")) (Cos (Var "a"))))"#),
            "{s:?}"
        );
        let cgene = r#"(Cos (Mul (Num 2.0) (Var "a")))"#;
        let c = rule_exprs(cgene, &[], "TR11-cos");
        assert!(
            has(&c, r#"(Sub (Pow2 (Cos (Var "a"))) (Pow2 (Sin (Var "a"))))"#),
            "{c:?}"
        );
        let any = generate(sgene, &[], 1_000_000, 0)
            .unwrap()
            .into_iter()
            .find(|c| c.rule == "TR11-sin")
            .unwrap();
        assert!(!any.speculative, "TR11 is an exact-identity reshape");
    }

    #[test]
    fn tr10_angle_sum_sin_and_cos() {
        // TR10 (SymPy fu.py): Sin(a+b) and Cos(a+b) expansions.
        let sgene = r#"(Sin (Add (Var "a") (Var "b")))"#;
        let s = rule_exprs(sgene, &[], "TR10-sin");
        assert!(
            has(
                &s,
                r#"(Add (Mul (Sin (Var "a")) (Cos (Var "b"))) (Mul (Cos (Var "a")) (Sin (Var "b"))))"#
            ),
            "{s:?}"
        );
        let cgene = r#"(Cos (Add (Var "a") (Var "b")))"#;
        let c = rule_exprs(cgene, &[], "TR10-cos");
        assert!(
            has(
                &c,
                r#"(Sub (Mul (Cos (Var "a")) (Cos (Var "b"))) (Mul (Sin (Var "a")) (Sin (Var "b"))))"#
            ),
            "{c:?}"
        );
        let full = generate(sgene, &[], 1_000_000, 0).unwrap();
        assert!(full.len() < INTERNAL_CAP);
    }

    #[test]
    fn tr5_pythagorean_squared_trig() {
        // TR5/TR6 (SymPy fu.py): Pow2(Sin x) -> 1 - Pow2(Cos x), and the dual.
        // The two directions form a 2-cycle terminated by the BFS `seen` set.
        let gene = r#"(Pow2 (Sin (Var "a")))"#;
        let tr5 = rule_exprs(gene, &[], "TR5");
        assert!(
            has(&tr5, r#"(Sub (Num 1.0) (Pow2 (Cos (Var "a"))))"#),
            "{tr5:?}"
        );
        let full = generate(gene, &[], 1_000_000, 0).unwrap();
        assert!(full.len() < INTERNAL_CAP);
        let any = full.into_iter().find(|c| c.rule == "TR5").unwrap();
        assert!(!any.speculative, "TR5 is an exact-identity reshape");
    }
}


