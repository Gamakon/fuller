use crate::termdag::{TermDag, TermId};
use crate::util::{HashMap, HashSet};
use crate::*;
use std::collections::VecDeque;

/// An interface for custom cost model.
///
/// To use it with the default extractor, the cost type must also satisfy `Ord + Eq + Clone + Debug`.
/// Additionally, the cost model should guarantee that a term has a no-smaller cost
/// than its subterms to avoid cycles in the extracted terms for common case usages.
/// For more niche usages, a term can have a cost less than its subterms.
/// As long as there is no negative cost cycle,
/// the default extractor is guaranteed to terminate in computing the costs.
/// However, the user needs to be careful to guarantee acyclicity in the extracted terms.
pub trait CostModel<C: Cost> {
    /// The total cost of a term given the cost of the root e-node and its immediate children's total costs.
    fn fold(&self, head: &str, children_cost: &[C], head_cost: C) -> C;

    /// The cost of an enode (without the cost of children)
    fn enode_cost(&self, egraph: &EGraph, func: &Function, row: &egglog_bridge::FunctionRow) -> C;

    /// The cost of a container value given the costs of its elements.
    ///
    /// The default cost for containers is just the sum of all the elements inside
    fn container_cost(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        element_costs: &[C],
    ) -> C {
        let _egraph = egraph;
        let _sort = sort;
        let _value = value;
        element_costs
            .iter()
            .fold(C::identity(), |s, c| s.combine(c))
    }

    /// Compute the cost of a (non-container) primitive value.
    ///
    /// The default cost for base values is the constant one
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C {
        let _egraph = egraph;
        let _sort = sort;
        let _value = value;
        C::unit()
    }
}

/// Requirements for a type to be usable as a cost by a [`CostModel`].
pub trait Cost {
    /// An identity element, usually zero.
    fn identity() -> Self;

    /// The default cost for a node with no children, usually one.
    fn unit() -> Self;

    /// A binary operation to combine costs, usually addition.
    /// This operation must NOT overflow or panic when given large values!
    fn combine(self, other: &Self) -> Self;
}

macro_rules! cost_impl_int {
    ($($cost:ty),*) => {$(
        impl Cost for $cost {
            fn identity() -> Self { 0 }
            fn unit()     -> Self { 1 }
            fn combine(self, other: &Self) -> Self {
                self.saturating_add(*other)
            }
        }
    )*};
}
cost_impl_int!(u8, u16, u32, u64, u128, usize);
cost_impl_int!(i8, i16, i32, i64, i128, isize);

macro_rules! cost_impl_num {
    ($($cost:ty),*) => {$(
        impl Cost for $cost {
            fn identity() -> Self {
                use num::Zero;
                Self::zero()
            }
            fn unit() -> Self {
                use num::One;
                Self::one()
            }
            fn combine(self, other: &Self) -> Self {
                self + other
            }
        }
    )*};
}
cost_impl_num!(num::BigInt, num::BigRational);
use ordered_float::OrderedFloat;
cost_impl_num!(f32, f64, OrderedFloat<f32>, OrderedFloat<f64>);

pub type DefaultCost = u64;

/// A cost model that computes the cost by summing the cost of each node.
#[derive(Default, Clone)]
pub struct TreeAdditiveCostModel {}

impl CostModel<DefaultCost> for TreeAdditiveCostModel {
    fn fold(
        &self,
        _head: &str,
        children_cost: &[DefaultCost],
        head_cost: DefaultCost,
    ) -> DefaultCost {
        children_cost.iter().fold(head_cost, |s, c| s.combine(c))
    }

    fn enode_cost(
        &self,
        _egraph: &EGraph,
        func: &Function,
        _row: &egglog_bridge::FunctionRow,
    ) -> DefaultCost {
        func.decl.cost.unwrap_or(DefaultCost::unit())
    }
}

/// The default, Bellman-Ford like extractor. This extractor is optimal for [`CostModel`].
///
/// Note that this assumes optimal substructure in the cost model, that is, a lower-cost
/// subterm should always lead to a non-worse superterm, to guarantee the extracted term
/// being optimal under the given cost model.
/// If this is not followed, the extractor may panic on reconstruction
pub struct Extractor<C: Cost + Ord + Eq + Clone + Debug> {
    rootsorts: Vec<ArcSort>,
    funcs: Vec<String>,
    cost_model: Box<dyn CostModel<C>>,
    costs: HashMap<String, HashMap<Value, C>>,
    topo_rnk_cnt: usize,
    topo_rnk: HashMap<String, HashMap<Value, usize>>,
    parent_edge: HashMap<String, HashMap<Value, (String, Vec<Value>)>>,
}

impl<C: Cost + Ord + Eq + Clone + Debug> Extractor<C> {
    /// Bulk of the computation happens at initialization time.
    /// The later extractions only reuses saved results.
    /// This means a new extractor must be created if the egraph changes.
    /// Holding a reference to the egraph would enforce this but prevents the extractor being reused.
    ///
    /// For convenience, if the rootsorts is `None`, it defaults to extract all extractable rootsorts.
    pub fn compute_costs_from_rootsorts(
        rootsorts: Option<Vec<ArcSort>>,
        egraph: &EGraph,
        cost_model: impl CostModel<C> + 'static,
    ) -> Self {
        // We filter out tables unreachable from the root sorts
        let extract_all_sorts = rootsorts.is_none();

        let mut rootsorts = rootsorts.unwrap_or_default();

        // Built a reverse index from output sort to function head symbols
        let mut rev_index: HashMap<String, Vec<String>> = Default::default();
        for func in egraph.functions.iter() {
            if !func.1.decl.unextractable {
                let func_name = func.0.clone();
                let output_sort_name = func.1.schema.output.name();
                if let Some(v) = rev_index.get_mut(output_sort_name) {
                    v.push(func_name);
                } else {
                    rev_index.insert(output_sort_name.to_owned(), vec![func_name]);
                    if extract_all_sorts {
                        rootsorts.push(func.1.schema.output.clone());
                    }
                }
            }
        }

        // Do a BFS to find reachable tables
        let mut q: VecDeque<ArcSort> = VecDeque::new();
        let mut seen: HashSet<String> = Default::default();
        for rootsort in rootsorts.iter() {
            q.push_back(rootsort.clone());
            seen.insert(rootsort.name().to_owned());
        }

        let mut funcs_set: HashSet<String> = Default::default();
        let mut funcs: Vec<String> = Vec::new();
        while !q.is_empty() {
            let sort = q.pop_front().unwrap();
            if sort.is_container_sort() {
                let inner_sorts = sort.inner_sorts();
                for s in inner_sorts {
                    if !seen.contains(s.name()) {
                        q.push_back(s.clone());
                        seen.insert(s.name().to_owned());
                    }
                }
            } else if sort.is_eq_sort() {
                if let Some(head_symbols) = rev_index.get(sort.name()) {
                    for h in head_symbols {
                        if !funcs_set.contains(h) {
                            let func = egraph.functions.get(h).unwrap();
                            for ch in &func.schema.input {
                                let ch_name = ch.name();
                                if !seen.contains(ch_name) {
                                    q.push_back(ch.clone());
                                    seen.insert(ch_name.to_owned());
                                }
                            }
                            funcs_set.insert(h.clone());
                            funcs.push(h.clone());
                        }
                    }
                }
            }
        }

        // Initialize the tables to have the reachable entries
        let mut costs: HashMap<String, HashMap<Value, C>> = Default::default();
        let mut topo_rnk: HashMap<String, HashMap<Value, usize>> = Default::default();
        let mut parent_edge: HashMap<String, HashMap<Value, (String, Vec<Value>)>> =
            Default::default();

        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            if !costs.contains_key(func.schema.output.name()) {
                debug_assert!(func.schema.output.is_eq_sort());
                costs.insert(func.schema.output.name().to_owned(), Default::default());
                topo_rnk.insert(func.schema.output.name().to_owned(), Default::default());
                parent_edge.insert(func.schema.output.name().to_owned(), Default::default());
            }
        }

        let mut extractor = Extractor {
            rootsorts,
            funcs,
            cost_model: Box::new(cost_model),
            costs,
            topo_rnk_cnt: 0,
            topo_rnk,
            parent_edge,
        };

        extractor.bellman_ford(egraph);

        extractor
    }

    /// Compute the cost of a single enode
    /// Recurse if container
    /// Returns None if contains an undefined eqsort term (potentially after unfolding)
    fn compute_cost_node(&self, egraph: &EGraph, value: Value, sort: &ArcSort) -> Option<C> {
        if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.container_values(), value);
            let mut ch_costs: Vec<C> = Vec::new();
            for ch in elements.iter() {
                ch_costs.push(self.compute_cost_node(egraph, ch.1, &ch.0)?);
            }
            Some(
                self.cost_model
                    .container_cost(egraph, sort, value, &ch_costs),
            )
        } else if sort.is_eq_sort() {
            self.costs.get(sort.name())?.get(&value).cloned()
        } else {
            // Primitive
            Some(self.cost_model.base_value_cost(egraph, sort, value))
        }
    }

    /// A row in a constructor table is a hyperedge from the set of input terms to the constructed output term.
    fn compute_cost_hyperedge(
        &self,
        egraph: &EGraph,
        row: &egglog_bridge::FunctionRow,
        func: &Function,
    ) -> Option<C> {
        let mut ch_costs: Vec<C> = Vec::new();
        let sorts = &func.schema.input;
        // Relying on .zip to truncate the values
        for (value, sort) in row.vals.iter().zip(sorts.iter()) {
            ch_costs.push(self.compute_cost_node(egraph, *value, sort)?);
        }
        Some(self.cost_model.fold(
            &func.decl.name,
            &ch_costs,
            self.cost_model.enode_cost(egraph, func, row),
        ))
    }

    fn compute_topo_rnk_node(&self, egraph: &EGraph, value: Value, sort: &ArcSort) -> usize {
        if sort.is_container_sort() {
            sort.inner_values(egraph.backend.container_values(), value)
                .iter()
                .fold(0, |ret, (sort, value)| {
                    usize::max(ret, self.compute_topo_rnk_node(egraph, *value, sort))
                })
        } else if sort.is_eq_sort() {
            if let Some(t) = self.topo_rnk.get(sort.name()) {
                *t.get(&value).unwrap_or(&usize::MAX)
            } else {
                usize::MAX
            }
        } else {
            0
        }
    }

    fn compute_topo_rnk_hyperedge(
        &self,
        egraph: &EGraph,
        row: &egglog_bridge::FunctionRow,
        func: &Function,
    ) -> usize {
        let sorts = &func.schema.input;
        row.vals
            .iter()
            .zip(sorts.iter())
            .fold(0, |ret, (value, sort)| {
                usize::max(ret, self.compute_topo_rnk_node(egraph, *value, sort))
            })
    }

    /// We use Bellman-Ford to compute the costs of the relevant eq sorts' terms
    /// [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) is a shortest path algorithm.
    /// The version implemented here computes the shortest path from any node in a set of sources to all the reachable nodes.
    /// Computing the minimum cost for terms is treated as a shortest path problem on a hypergraph here.
    /// In this hypergraph, the nodes corresponde to eclasses, the distances are the costs to extract a term of those eclasses,
    /// and each enode is a hyperedge that goes from the set of children eclasses to the enode's eclass.
    /// The sources are the eclasses with known costs from the cost model.
    /// Additionally, to avoid cycles in the extraction even when the cost model can assign an equal cost to a term and its subterm.
    /// It computes a topological rank for each eclass
    /// and only allows each eclass to have children of classes of strictly smaller ranks in the extraction.
    fn bellman_ford(&mut self, egraph: &EGraph) {
        let mut ensure_fixpoint = false;

        let funcs = self.funcs.clone();

        while !ensure_fixpoint {
            ensure_fixpoint = true;

            for func_name in funcs.iter() {
                let func = egraph.functions.get(func_name).unwrap();
                let target_sort = func.schema.output.clone();

                let relax_hyperedge = |row: egglog_bridge::FunctionRow| {
                    if !row.subsumed {
                        let target = row.vals.last().unwrap();
                        let mut updated = false;
                        if let Some(new_cost) = self.compute_cost_hyperedge(egraph, &row, func) {
                            match self
                                .costs
                                .get_mut(target_sort.name())
                                .unwrap()
                                .entry(*target)
                            {
                                HEntry::Vacant(e) => {
                                    updated = true;
                                    e.insert(new_cost);
                                }
                                HEntry::Occupied(mut e) => {
                                    if new_cost < *(e.get()) {
                                        updated = true;
                                        e.insert(new_cost);
                                    }
                                }
                            }
                        }
                        // record the chronological order of the updates
                        // which serves as a topological order that avoids cycles
                        // even when a term has a cost equal to its subterms
                        if updated {
                            ensure_fixpoint = false;
                            self.topo_rnk_cnt += 1;
                            self.topo_rnk
                                .get_mut(target_sort.name())
                                .unwrap()
                                .insert(*target, self.topo_rnk_cnt);
                        }
                    }
                };

                egraph.backend.for_each(func.backend_id, relax_hyperedge);
            }
        }

        // Save the edges for reconstruction
        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            let target_sort = func.schema.output.clone();

            let save_best_parent_edge = |row: egglog_bridge::FunctionRow| {
                if !row.subsumed {
                    let target = row.vals.last().unwrap();
                    if let Some(best_cost) = self.costs.get(target_sort.name()).unwrap().get(target)
                    {
                        if Some(best_cost.clone())
                            == self.compute_cost_hyperedge(egraph, &row, func)
                        {
                            // one of the possible best parent edges
                            let target_topo_rnk = *self
                                .topo_rnk
                                .get(target_sort.name())
                                .unwrap()
                                .get(target)
                                .unwrap();
                            if target_topo_rnk > self.compute_topo_rnk_hyperedge(egraph, &row, func)
                            {
                                // one of the parent edges that avoids cycles
                                if let HEntry::Vacant(e) = self
                                    .parent_edge
                                    .get_mut(target_sort.name())
                                    .unwrap()
                                    .entry(*target)
                                {
                                    e.insert((func.decl.name.clone(), row.vals.to_vec()));
                                }
                            }
                        }
                    }
                }
            };

            egraph
                .backend
                .for_each(func.backend_id, save_best_parent_edge);
        }
    }

    /// This recursively reconstruct the termdag that gives the minimum cost for eclass value.
    fn reconstruct_termdag_node(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
    ) -> TermId {
        self.reconstruct_termdag_node_helper(egraph, termdag, value, sort, &mut Default::default())
    }

    fn reconstruct_termdag_node_helper(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
        cache: &mut HashMap<(Value, String), TermId>,
    ) -> TermId {
        let key = (value, sort.name().to_owned());
        if let Some(term) = cache.get(&key) {
            return *term;
        }

        let term = if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.container_values(), value);
            let mut ch_terms: Vec<TermId> = Vec::new();
            for ch in elements.iter() {
                ch_terms.push(
                    self.reconstruct_termdag_node_helper(egraph, termdag, ch.1, &ch.0, cache),
                );
            }
            sort.reconstruct_termdag_container(
                egraph.backend.container_values(),
                value,
                termdag,
                ch_terms,
            )
        } else if sort.is_eq_sort() {
            let (func_name, hyperedge) = self
                .parent_edge
                .get(sort.name())
                .unwrap()
                .get(&value)
                .unwrap();
            let mut ch_terms: Vec<TermId> = Vec::new();
            let ch_sorts = &egraph.functions.get(func_name).unwrap().schema.input;
            for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()) {
                ch_terms.push(
                    self.reconstruct_termdag_node_helper(egraph, termdag, *value, sort, cache),
                );
            }
            termdag.app(func_name.clone(), ch_terms)
        } else {
            // Base value case
            sort.reconstruct_termdag_base(egraph.backend.base_values(), value, termdag)
        };

        cache.insert(key, term);
        term
    }

    /// Extract the best term of a value from a given sort.
    ///
    /// This function expects the sort to be already computed,
    /// which can be one of the rootsorts, or reachable from rootsorts, or primitives, or containers of computed sorts.
    pub fn extract_best_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: ArcSort,
    ) -> Option<(C, TermId)> {
        match self.compute_cost_node(egraph, value, &sort) {
            Some(best_cost) => {
                log::debug!("Best cost for the extract root: {:?}", best_cost);

                let term = self.reconstruct_termdag_node(egraph, termdag, value, &sort);

                Some((best_cost, term))
            }
            None => {
                log::error!("Unextractable root {:?} with sort {:?}", value, sort,);
                None
            }
        }
    }

    /// A convenience method for extraction.
    ///
    /// This expects the value to be of the unique sort the extractor has been initialized with
    pub fn extract_best(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
    ) -> Option<(C, TermId)> {
        assert!(
            self.rootsorts.len() == 1,
            "extract_best requires a single rootsort"
        );
        self.extract_best_with_sort(
            egraph,
            termdag,
            value,
            self.rootsorts.first().unwrap().clone(),
        )
    }

    /// Extract variants of an e-class.
    ///
    /// The variants are selected by first picking `nvairants` e-nodes with the lowest cost from the e-class
    /// and then extracting a term from each e-node.
    pub fn extract_variants_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: ArcSort,
    ) -> Vec<(C, TermId)> {
        debug_assert!(self.rootsorts.iter().any(|s| { s.name() == sort.name() }));

        if sort.is_eq_sort() {
            let mut root_variants: Vec<(C, String, Vec<Value>)> = Vec::new();

            let mut root_funcs: Vec<String> = Vec::new();

            for func_name in self.funcs.iter() {
                // Need an eq on sorts
                if sort.name()
                    == egraph
                        .functions
                        .get(func_name)
                        .unwrap()
                        .schema
                        .output
                        .name()
                {
                    root_funcs.push(func_name.clone());
                }
            }

            for func_name in root_funcs.iter() {
                let func = egraph.functions.get(func_name).unwrap();

                let find_root_variants = |row: egglog_bridge::FunctionRow| {
                    if !row.subsumed {
                        let target = row.vals.last().unwrap();
                        if *target == value {
                            let cost = self.compute_cost_hyperedge(egraph, &row, func).unwrap();
                            root_variants.push((cost, func_name.clone(), row.vals.to_vec()));
                        }
                    }
                };

                egraph.backend.for_each(func.backend_id, find_root_variants);
            }

            let mut res: Vec<(C, TermId)> = Vec::new();
            root_variants.sort();
            root_variants.truncate(nvariants);
            for (cost, func_name, hyperedge) in root_variants {
                let mut ch_terms: Vec<TermId> = Vec::new();
                let ch_sorts = &egraph.functions.get(&func_name).unwrap().schema.input;
                // zip truncates the row
                for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()) {
                    ch_terms.push(self.reconstruct_termdag_node(egraph, termdag, *value, sort));
                }
                res.push((cost, termdag.app(func_name, ch_terms)));
            }

            res
        } else {
            log::warn!(
                "extracting multiple variants for containers or primitives is not implemented, returning a single variant."
            );
            if let Some(res) = self.extract_best_with_sort(egraph, termdag, value, sort) {
                vec![res]
            } else {
                vec![]
            }
        }
    }

    /// A convenience method for extracting variants of a value.
    ///
    /// This expects the value to be of the unique sort the extractor has been initialized with.
    pub fn extract_variants(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
    ) -> Vec<(C, TermId)> {
        assert!(
            self.rootsorts.len() == 1,
            "extract_variants requires a single rootsort"
        );
        self.extract_variants_with_sort(
            egraph,
            termdag,
            value,
            nvariants,
            self.rootsorts.first().unwrap().clone(),
        )
    }
}

/// HFF extraction — the replacement for the scalar Bellman-Ford [`Extractor`].
///
/// egglog's stock extractor picks each e-class winner by a SCALAR cost that must
/// be MONOTONE (a term costs no less than its subterms) so the bottom-up walk
/// never reconsiders. That forces every selection criterion into one additive
/// number. A multi-objective, geometric criterion — like a hyperspherical-fitness
/// angle, where a larger term can legitimately score better — is non-monotone and
/// breaks that walk.
///
/// This function instead ENUMERATES each e-class's member terms (bounded) and
/// selects the one minimising a caller-supplied `score(&TermDag, TermId) -> f64`
/// evaluated on the WHOLE candidate subterm. No monotonicity is assumed: the
/// score is recomputed per whole candidate, so any objective expressible as a
/// function of the rendered term is admissible — including the CDF-corrected HFF
/// angle over a `/pattern/{measure}` vector. This is the "multi-objective by
/// design" tournament: the score need not be a single additive scalar.
///
/// `max_members_per_class` bounds the enumeration so a large e-class can't blow
/// up; cycles are cut by a visited set (an e-class referencing itself only sees
/// the members found so far). Returns the selected best `(score, TermId)` and the
/// full ranked member list for the class of `value`.
pub fn hff_extract(
    egraph: &EGraph,
    termdag: &mut TermDag,
    root_value: Value,
    root_sort: ArcSort,
    score: &dyn Fn(&TermDag, TermId) -> f64,
    max_members_per_class: usize,
) -> Vec<(f64, TermId)> {
    // Memoised members per (value, sort-name): the bounded set of whole terms
    // reachable for that e-class, each already its own subtree in `termdag`.
    let mut memo: HashMap<(Value, String), Vec<TermId>> = HashMap::default();
    let mut on_stack: HashSet<(Value, String)> = HashSet::default();
    let root_members =
        enumerate_members(egraph, termdag, root_value, &root_sort, max_members_per_class, &mut memo, &mut on_stack);
    let mut ranked: Vec<(f64, TermId)> =
        root_members.into_iter().map(|t| (score(termdag, t), t)).collect();
    // De-dup by rendered string, keep best score per distinct form.
    let mut best: HashMap<String, (f64, TermId)> = HashMap::default();
    for (s, t) in ranked.drain(..) {
        let key = termdag.to_string(t);
        match best.get(&key) {
            Some((bs, _)) if *bs <= s => {}
            _ => {
                best.insert(key, (s, t));
            }
        }
    }
    let mut out: Vec<(f64, TermId)> = best.into_values().collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Enumerate up to `cap` member terms of the e-class identified by `value`/`sort`.
/// Bottom-up: each e-node of the class contributes the cartesian product of its
/// children's member terms (capped). Containers and base values reconstruct as in
/// the stock extractor. Recursion into an e-class already on the stack returns the
/// members found so far (cycle cut), so self-referential classes stay finite.
fn enumerate_members(
    egraph: &EGraph,
    termdag: &mut TermDag,
    value: Value,
    sort: &ArcSort,
    cap: usize,
    memo: &mut HashMap<(Value, String), Vec<TermId>>,
    on_stack: &mut HashSet<(Value, String)>,
) -> Vec<TermId> {
    let key = (value, sort.name().to_owned());
    if let Some(m) = memo.get(&key) {
        return m.clone();
    }
    if on_stack.contains(&key) {
        // Cycle: this class is still being built; contribute nothing yet.
        return Vec::new();
    }
    on_stack.insert(key.clone());

    let members: Vec<TermId> = if sort.is_container_sort() {
        // Containers: one shape, children each reconstructed to their members'
        // first (cheapest-by-enumeration-order) term — containers are not the
        // multi-objective target, so a single representative suffices.
        let elements = sort.inner_values(egraph.backend.container_values(), value);
        let mut ch_terms: Vec<TermId> = Vec::new();
        for ch in elements.iter() {
            let ms = enumerate_members(egraph, termdag, ch.1, &ch.0, cap, memo, on_stack);
            ch_terms.push(ms.into_iter().next().unwrap_or_else(|| termdag.var("Unextractable".into())));
        }
        vec![sort.reconstruct_termdag_container(
            egraph.backend.container_values(),
            value,
            termdag,
            ch_terms,
        )]
    } else if sort.is_eq_sort() {
        // Every e-node whose output is this value, across all funcs of this sort.
        let mut rows: Vec<(String, Vec<Value>)> = Vec::new();
        for (func_name, func) in egraph.functions.iter() {
            if func.decl.unextractable || func.schema.output.name() != sort.name() {
                continue;
            }
            let fname = func_name.clone();
            let collected = std::cell::RefCell::new(Vec::new());
            egraph.backend.for_each(func.backend_id, |row: egglog_bridge::FunctionRow| {
                if !row.subsumed && row.vals.last() == Some(&value) {
                    collected.borrow_mut().push(row.vals[..row.vals.len() - 1].to_vec());
                }
            });
            for child_vals in collected.into_inner() {
                rows.push((fname.clone(), child_vals));
            }
        }
        let mut out: Vec<TermId> = Vec::new();
        for (func_name, child_vals) in rows {
            let ch_sorts = egraph.functions.get(&func_name).unwrap().schema.input.clone();
            // Members of each child, then a bounded cartesian product.
            let mut child_members: Vec<Vec<TermId>> = Vec::new();
            for (cv, cs) in child_vals.iter().zip(ch_sorts.iter()) {
                let ms = enumerate_members(egraph, termdag, *cv, cs, cap, memo, on_stack);
                if ms.is_empty() {
                    child_members.clear();
                    break;
                }
                child_members.push(ms);
            }
            if child_vals.is_empty() {
                out.push(termdag.app(func_name.clone(), vec![]));
            } else if !child_members.is_empty() && child_members.len() == child_vals.len() {
                for combo in bounded_product(&child_members, cap) {
                    out.push(termdag.app(func_name.clone(), combo));
                    if out.len() >= cap {
                        break;
                    }
                }
            }
            if out.len() >= cap {
                break;
            }
        }
        out
    } else {
        // Base value (a literal): one term.
        vec![sort.reconstruct_termdag_base(egraph.backend.base_values(), value, termdag)]
    };

    on_stack.remove(&key);
    memo.insert(key, members.clone());
    members
}

/// Bounded cartesian product of children's member lists: at most `cap` combos.
fn bounded_product(lists: &[Vec<TermId>], cap: usize) -> Vec<Vec<TermId>> {
    let mut acc: Vec<Vec<TermId>> = vec![Vec::new()];
    for list in lists {
        let mut next: Vec<Vec<TermId>> = Vec::new();
        for prefix in &acc {
            for &item in list {
                let mut p = prefix.clone();
                p.push(item);
                next.push(p);
                if next.len() >= cap {
                    break;
                }
            }
            if next.len() >= cap {
                break;
            }
        }
        acc = next;
    }
    acc
}
