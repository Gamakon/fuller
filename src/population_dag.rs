//! The population DAG (`docs/PLAN_population_dag.md` §1, task 3): one
//! hash-consed DAG for a whole population's genes, maintained across
//! generations, with homeotic rows (the definitions the population
//! repeats) carrying memoised row values.
//!
//! Functional in the plan's sense: `fold` takes the DAG and a population
//! and returns the DAG grown by what the population needed (never
//! rebuilt; an empty DAG makes the first call a batch); `eval` takes the
//! DAG and the rows and computes every root, reading a homeotic node's
//! memo when its key matches and writing it when it does not. Nothing is
//! evicted (the plan's first version: let it grow, record use).
//!
//! A memo is keyed by the node (its structural hash is its index), the
//! data version, the numerical type of the node's root and the input
//! binding (which columns the rows carry), as the review required; a
//! memo whose key differs in any part is a miss. Execution follows the
//! level schedule (a node's level is one more than its deepest child),
//! never the use order. Determinism is the contract: a memo changes
//! whether a value is recomputed, never what it is, and the tests assert
//! bit-identical values with memos present and absent.
//!
//! Host-side, over the `Math` kingdom's s-expressions (`karva::parse_math`
//! and `eval::apply`): the machinery and its measurements; the device
//! arena is phylu's (task 4).

use std::collections::{BTreeMap, HashMap};

use crate::karva::{parse_math, MathNode};

/// One node of the DAG: a constructor over child ids, or a leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// The constructor (`Add`, `Sin`, …), `Var` or `Num`.
    pub ctor: String,
    pub kids: Vec<u32>,
    /// A `Var` leaf's name.
    pub var: Option<String>,
    /// A `Num` leaf's value.
    pub num: Option<f64>,
    /// Operators in the subtree (0 for a leaf).
    pub ops: u32,
    /// One more than the deepest child's level; 0 for a leaf.
    pub level: u32,
}

/// A memoised value vector for one node.
#[derive(Debug, Clone, PartialEq)]
pub struct Memo {
    pub data_version: u64,
    pub ty: &'static str,
    /// A hash of the input binding (the columns the rows carry, in order).
    pub binding: u64,
    pub values: Vec<f64>,
}

/// What one `eval` did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// Node evaluations performed (one per node per root walk, memo
    /// misses and non-homeotic nodes).
    pub evaluated: u64,
    /// Node evaluations avoided by a memo hit (the subtree below counted
    /// too: what the interpreter would have done).
    pub avoided: u64,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PopulationDag {
    pub nodes: Vec<Node>,
    /// Rendered text → node id: the hash-cons.
    index: HashMap<String, u32>,
    /// Occurrences of each node across the LAST folded population (a
    /// node's referrers, counting every path to it).
    pub uses: Vec<u32>,
    /// Memo per node, present for homeotic nodes once evaluated.
    pub memo: Vec<Option<Memo>>,
    /// How many generations each node has been present (since it was
    /// interned); the tail study's age.
    pub age: Vec<u32>,
    pub generation: u32,
}

/// Whether a node is a definition the population shares: repeated and
/// doing work.
pub fn is_homeotic(node: &Node, uses: u32) -> bool {
    uses >= 2 && node.ops >= 1
}

impl PopulationDag {
    pub fn new() -> Self {
        Self::default()
    }

    fn intern(&mut self, n: &MathNode) -> u32 {
        let text = render(n);
        if let Some(&id) = self.index.get(&text) {
            return id;
        }
        let node = match n {
            MathNode::Num(v) => Node { ctor: "Num".into(), kids: Vec::new(), var: None, num: Some(*v), ops: 0, level: 0 },
            MathNode::Var(name) => Node { ctor: "Var".into(), kids: Vec::new(), var: Some(name.clone()), num: None, ops: 0, level: 0 },
            MathNode::App(ctor, children) => {
                let kids: Vec<u32> = children.iter().map(|c| self.intern(c)).collect();
                let ops = 1 + kids.iter().map(|&k| self.nodes[k as usize].ops).sum::<u32>();
                let level = 1 + kids.iter().map(|&k| self.nodes[k as usize].level).max().unwrap_or(0);
                Node { ctor: ctor.clone(), kids, var: None, num: None, ops, level }
            }
        };
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.index.insert(text, id);
        self.uses.push(0);
        self.memo.push(None);
        self.age.push(0);
        id
    }

    /// Fold a population into the DAG: intern every gene (new subtrees
    /// are added, known ones found), recount the uses from this
    /// population, age every node by one generation. Returns the root id
    /// of every gene, in order.
    pub fn fold(&mut self, genes: &[String]) -> Result<Vec<u32>, String> {
        let trees: Vec<MathNode> = genes.iter().map(|g| parse_math(g)).collect::<Result<_, _>>()?;
        let roots: Vec<u32> = trees.iter().map(|t| self.intern(t)).collect();
        for u in &mut self.uses {
            *u = 0;
        }
        for &r in &roots {
            self.count_uses(r);
        }
        for a in &mut self.age {
            *a += 1;
        }
        self.generation += 1;
        Ok(roots)
    }

    fn count_uses(&mut self, id: u32) {
        self.uses[id as usize] += 1;
        let kids = self.nodes[id as usize].kids.clone();
        for k in kids {
            self.count_uses(k);
        }
    }

    /// The homeotic rows: the ids of every repeated node doing work, in
    /// level order (a definition after everything it references).
    pub fn homeotic(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..self.nodes.len() as u32).filter(|&i| is_homeotic(&self.nodes[i as usize], self.uses[i as usize])).collect();
        ids.sort_by_key(|&i| (self.nodes[i as usize].level, i));
        ids
    }

    /// Evaluate `roots` over `rows` (each row a binding of variable names
    /// to values; every row the same names, which is the input binding).
    /// A homeotic node reads its memo when the key matches and writes it
    /// otherwise; every other node is computed. Returns, per root, the
    /// values over the rows, and the counts.
    pub fn eval(&mut self, roots: &[u32], rows: &[Vec<(String, f64)>], data_version: u64) -> Result<(Vec<Vec<f64>>, EvalStats), String> {
        let binding = binding_hash(rows);
        let mut stats = EvalStats::default();
        let mut out = Vec::with_capacity(roots.len());
        let mut cache: HashMap<u32, Vec<f64>> = HashMap::new();
        for &r in roots {
            let v = self.eval_node(r, rows, data_version, binding, &mut cache, &mut stats)?;
            out.push(v);
        }
        Ok((out, stats))
    }

    fn eval_node(&mut self, id: u32, rows: &[Vec<(String, f64)>], data_version: u64, binding: u64, cache: &mut HashMap<u32, Vec<f64>>, stats: &mut EvalStats) -> Result<Vec<f64>, String> {
        if let Some(v) = cache.get(&id) {
            // Computed earlier in this very eval (a shared subtree): no
            // interpreter would redo it either, so it is neither a hit nor a miss.
            return Ok(v.clone());
        }
        let homeotic = is_homeotic(&self.nodes[id as usize], self.uses[id as usize]);
        if homeotic {
            if let Some(m) = &self.memo[id as usize] {
                if m.data_version == data_version && m.ty == "f64" && m.binding == binding {
                    stats.hits += 1;
                    stats.avoided += u64::from(self.nodes[id as usize].ops);
                    let v = m.values.clone();
                    cache.insert(id, v.clone());
                    return Ok(v);
                }
            }
            stats.misses += 1;
        }
        let node = self.nodes[id as usize].clone();
        let values: Vec<f64> = match node.ctor.as_str() {
            "Num" => vec![node.num.unwrap_or(f64::NAN); rows.len()],
            "Var" => {
                let name = node.var.clone().unwrap_or_default();
                rows.iter().map(|row| row.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap_or(f64::NAN)).collect()
            }
            ctor => {
                let kid_values: Vec<Vec<f64>> = node.kids.iter().map(|&k| self.eval_node(k, rows, data_version, binding, cache, stats)).collect::<Result<_, _>>()?;
                stats.evaluated += 1;
                (0..rows.len())
                    .map(|i| {
                        let args: Vec<f64> = kid_values.iter().map(|v| v[i]).collect();
                        crate::eval::apply(ctor, &args).ok_or_else(|| format!("population dag: no evaluator for {ctor}/{}", args.len()))
                    })
                    .collect::<Result<_, _>>()?
            }
        };
        if homeotic {
            self.memo[id as usize] = Some(Memo { data_version, ty: "f64", binding, values: values.clone() });
        }
        cache.insert(id, values.clone());
        Ok(values)
    }

    /// Memos present (the arena's occupancy).
    pub fn memos(&self) -> usize {
        self.memo.iter().filter(|m| m.is_some()).count()
    }
}

fn binding_hash(rows: &[Vec<(String, f64)>]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    if let Some(first) = rows.first() {
        for (name, _) in first {
            name.hash(&mut h);
        }
    }
    rows.len().hash(&mut h);
    h.finish()
}

fn render(n: &MathNode) -> String {
    match n {
        MathNode::Num(v) => format!("(Num {v:?})"),
        MathNode::Var(name) => format!("(Var {name:?})"),
        MathNode::App(ctor, children) => {
            let parts: Vec<String> = children.iter().map(render).collect();
            if parts.is_empty() {
                format!("({ctor})")
            } else {
                format!("({ctor} {})", parts.join(" "))
            }
        }
    }
}

/// The study's counts for one generation folded after another.
#[derive(Debug, Clone, Default)]
pub struct GenerationReport {
    pub nodes_total: usize,
    pub homeotic: usize,
    pub homeotic_new: usize,
    pub uses_histogram: BTreeMap<u32, usize>,
}

impl PopulationDag {
    /// What this generation's fold did to the tail: how many homeotic
    /// rows, how many of them interned this generation, and the use
    /// histogram.
    pub fn report(&self) -> GenerationReport {
        let mut r = GenerationReport { nodes_total: self.nodes.len(), ..Default::default() };
        for id in self.homeotic() {
            r.homeotic += 1;
            if self.age[id as usize] == 1 {
                r.homeotic_new += 1;
            }
            *r.uses_histogram.entry(self.uses[id as usize]).or_default() += 1;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<Vec<(String, f64)>> {
        (0..5).map(|i| vec![("x".to_string(), i as f64 * 0.5 - 1.0), ("y".to_string(), 2.0 - i as f64)]).collect()
    }

    #[test]
    fn a_population_folds_into_one_dag_with_shared_definitions_in_level_order() {
        let mut dag = PopulationDag::new();
        let genes = vec![
            r#"(Add (Mul (Var "x") (Var "y")) (Sin (Mul (Var "x") (Var "y"))))"#.to_string(),
            r#"(Sub (Sin (Mul (Var "x") (Var "y"))) (Num 1.0))"#.to_string(),
        ];
        let roots = dag.fold(&genes).unwrap();
        assert_eq!(roots.len(), 2);
        // x, y, x*y, sin(x*y), the two roots, Num 1.0: seven nodes, no duplicates.
        assert_eq!(dag.nodes.len(), 7);
        let h = dag.homeotic();
        let texts: Vec<&str> = h.iter().map(|&i| dag.nodes[i as usize].ctor.as_str()).collect();
        assert_eq!(texts, vec!["Mul", "Sin"], "x*y (used 3 times) then sin(x*y) (used twice), by level");
        assert!(dag.nodes[h[0] as usize].level < dag.nodes[h[1] as usize].level);
    }

    #[test]
    fn a_second_generation_adds_only_what_is_new_and_hits_the_memos_of_what_survived() {
        let mut dag = PopulationDag::new();
        let g1 = vec![
            r#"(Add (Mul (Var "x") (Var "y")) (Sin (Mul (Var "x") (Var "y"))))"#.to_string(),
            r#"(Sub (Sin (Mul (Var "x") (Var "y"))) (Num 1.0))"#.to_string(),
        ];
        let r1 = dag.fold(&g1).unwrap();
        let (v1, s1) = dag.eval(&r1, &rows(), 1).unwrap();
        assert_eq!((s1.hits, s1.misses), (0, 2), "first generation: every definition a miss");
        // The child keeps sin(x*y); the second gene is replaced by a new one.
        let g2 = vec![g1[0].clone(), r#"(Mul (Sin (Mul (Var "x") (Var "y"))) (Var "x"))"#.to_string()];
        let n_before = dag.nodes.len();
        let r2 = dag.fold(&g2).unwrap();
        assert_eq!(dag.nodes.len(), n_before + 1, "one new node (the new root)");
        let (v2, s2) = dag.eval(&r2, &rows(), 1).unwrap();
        assert_eq!((s2.hits, s2.misses), (2, 0), "both definitions survived and hit");
        assert!(s2.avoided > 0);
        // Determinism: the same values as a fresh evaluation without memos.
        let mut fresh = PopulationDag::new();
        let rf = fresh.fold(&g2).unwrap();
        let (vf, _) = fresh.eval(&rf, &rows(), 1).unwrap();
        for (a, b) in v2.iter().zip(&vf) {
            for (x, y) in a.iter().zip(b) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        assert_eq!(v1[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), v2[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn a_memo_is_a_miss_when_the_data_version_or_the_binding_differs() {
        let mut dag = PopulationDag::new();
        let g = vec![
            r#"(Add (Mul (Var "x") (Var "y")) (Num 1.0))"#.to_string(),
            r#"(Sub (Mul (Var "x") (Var "y")) (Num 1.0))"#.to_string(),
        ];
        let r = dag.fold(&g).unwrap();
        let (_, s) = dag.eval(&r, &rows(), 1).unwrap();
        assert_eq!(s.misses, 1);
        let (_, s) = dag.eval(&r, &rows(), 1).unwrap();
        assert_eq!((s.hits, s.misses), (1, 0));
        let (_, s) = dag.eval(&r, &rows(), 2).unwrap();
        assert_eq!((s.hits, s.misses), (0, 1), "a new data version");
        let other: Vec<Vec<(String, f64)>> = rows().into_iter().map(|r| r.into_iter().map(|(n, v)| (if n == "y" { "z".to_string() } else { n }, v)).collect()).collect();
        let (_, s) = dag.eval(&r, &other, 2).unwrap();
        assert_eq!((s.hits, s.misses), (0, 1), "a different input binding");
    }
}
