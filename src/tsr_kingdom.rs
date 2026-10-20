//! The TSR kingdom's rows, loaded from the generated tables
//! (`kingdoms/tsr/<variant>/types.tsv` and `functions.tsv`, written by
//! `kingdoms/tsr/gen_tables.py`; docs/PLAN_tsr_typed_gep.md, step F2).
//!
//! Each row of `types.tsv` is a dual (`Ty::Tsr(index)`), the last of them the
//! `ROOT` dual. Each `function` row of `functions.tsv` is one operator over
//! one combination of argument duals, each `root` row its twin with output
//! `ROOT`. The `var` and `const` terminal rows are polymorphic: the loader
//! gives every dual one terminal per input column and one `?` placeholder, so
//! a leaf fits every slot (and every demanded dual has a terminal). `absent`
//! and `canon` rows are kept, named, so no operator is silently dropped.

use crate::geneframe::{Arity, Symbol, SymbolTable, Ty};

/// The kingdom's name in the symbol table.
pub const TSR: &str = "TSR";

/// One kingdom of the plan's arms: which limits the types carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TsrVariant {
    /// Neither limit: one dual `F`, which is also the root.
    Single,
    /// Stream order (arm B).
    Order,
    /// Transcendental depth (arm E).
    Depth,
    /// Both limits (arm C).
    OrderDepth,
    /// Both limits and the operator windows (arm F).
    OrderDepthWindows,
}

impl TsrVariant {
    pub const ALL: [TsrVariant; 5] =
        [TsrVariant::Single, TsrVariant::Order, TsrVariant::Depth, TsrVariant::OrderDepth, TsrVariant::OrderDepthWindows];

    /// The directory under `kingdoms/tsr`.
    pub fn dir(self) -> &'static str {
        match self {
            TsrVariant::Single => "single",
            TsrVariant::Order => "order",
            TsrVariant::Depth => "depth",
            TsrVariant::OrderDepth => "order_depth",
            TsrVariant::OrderDepthWindows => "order_depth_windows",
        }
    }

    fn types_tsv(self) -> &'static str {
        match self {
            TsrVariant::Single => include_str!("../kingdoms/tsr/single/types.tsv"),
            TsrVariant::Order => include_str!("../kingdoms/tsr/order/types.tsv"),
            TsrVariant::Depth => include_str!("../kingdoms/tsr/depth/types.tsv"),
            TsrVariant::OrderDepth => include_str!("../kingdoms/tsr/order_depth/types.tsv"),
            TsrVariant::OrderDepthWindows => include_str!("../kingdoms/tsr/order_depth_windows/types.tsv"),
        }
    }

    fn functions_tsv(self) -> &'static str {
        match self {
            TsrVariant::Single => include_str!("../kingdoms/tsr/single/functions.tsv"),
            TsrVariant::Order => include_str!("../kingdoms/tsr/order/functions.tsv"),
            TsrVariant::Depth => include_str!("../kingdoms/tsr/depth/functions.tsv"),
            TsrVariant::OrderDepth => include_str!("../kingdoms/tsr/order_depth/functions.tsv"),
            TsrVariant::OrderDepthWindows => include_str!("../kingdoms/tsr/order_depth_windows/functions.tsv"),
        }
    }
}

/// One dual of `types.tsv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TsrDual {
    /// `shape.level`, `ROOT`, or `F` in the single kingdom.
    pub dual: String,
    /// The root slot's dual.
    pub root: bool,
}

/// What a row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsrKind {
    /// An operator over argument duals.
    Function,
    /// The twin of a function row, output `ROOT`.
    RootTwin,
    /// The variable of input column `n`, at one dual.
    Var(usize),
    /// The constant placeholder `?`, at one dual.
    Const,
}

/// One typed row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TsrRow {
    /// Unique: `add@A12<-P1,P1`, `x2@P1`, `?@P1.T1`.
    pub name: String,
    /// What it computes: the table's `instance` (`add`, `var`, `const`).
    pub semantic_id: String,
    /// The evaluator operator (`Add`, `ProtectedDiv`, `Var`, `Num`).
    pub constructor: String,
    /// Ordered input duals, one per child slot.
    pub inputs: Vec<Ty>,
    pub output: Ty,
    pub kind: TsrKind,
}

impl TsrRow {
    pub fn terminal(&self) -> bool {
        matches!(self.kind, TsrKind::Var(_) | TsrKind::Const)
    }
}

/// An operator the table names but does not offer (`absent`), or spells as
/// another (`canon`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TsrNote {
    pub class: String,
    pub instance: String,
    pub note: String,
}

/// The loaded kingdom.
#[derive(Debug, Clone)]
pub struct TsrKingdom {
    pub variant: TsrVariant,
    pub n_inputs: usize,
    pub duals: Vec<TsrDual>,
    /// Functions, then root twins, then terminals, in table order.
    pub rows: Vec<TsrRow>,
    pub notes: Vec<TsrNote>,
}

fn data_lines(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect())
}

impl TsrKingdom {
    /// Load `variant` over `n_inputs` input columns (the `var` rows expand to
    /// one terminal per column at every dual).
    pub fn load(variant: TsrVariant, n_inputs: usize) -> Result<TsrKingdom, String> {
        let mut duals = Vec::new();
        for f in data_lines(variant.types_tsv()) {
            if f.len() < 9 {
                return Err(format!("{}: types.tsv row {f:?} has {} columns, wants 10", variant.dir(), f.len()));
            }
            duals.push(TsrDual { dual: f[1].to_string(), root: f[8] == "yes" });
        }
        if u16::try_from(duals.len()).is_err() {
            return Err(format!("{}: {} duals do not fit Ty::Tsr(u16)", variant.dir(), duals.len()));
        }
        let ty_of = |name: &str| -> Result<Ty, String> {
            duals
                .iter()
                .position(|d| d.dual == name)
                .map(|i| Ty::Tsr(i as u16))
                .ok_or_else(|| format!("{}: no dual {name}", variant.dir()))
        };

        let (mut functions, mut twins, mut terminals, mut notes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for f in data_lines(variant.functions_tsv()) {
            if f.len() < 7 {
                return Err(format!("{}: functions.tsv row {f:?} has {} columns, wants 8", variant.dir(), f.len()));
            }
            let (class, instance, constructor, ins, out) = (f[1], f[2], f[3], f[4], f[5]);
            match class {
                "function" | "root" => {
                    let inputs = ins.split(',').map(&ty_of).collect::<Result<Vec<_>, _>>()?;
                    let arity: usize = f[6].parse().map_err(|_| format!("{}: arity {:?}", variant.dir(), f[6]))?;
                    if inputs.len() != arity {
                        return Err(format!("{}: {instance} {ins} names {} inputs, arity {arity}", variant.dir(), inputs.len()));
                    }
                    let kind = if class == "function" { TsrKind::Function } else { TsrKind::RootTwin };
                    let name = format!("{instance}@{out}<-{ins}");
                    let row = TsrRow {
                        name,
                        semantic_id: instance.to_string(),
                        constructor: constructor.to_string(),
                        inputs,
                        output: ty_of(out)?,
                        kind,
                    };
                    if class == "function" { functions.push(row) } else { twins.push(row) }
                }
                "terminal" => {
                    let output = ty_of(out)?;
                    match instance {
                        "var" => {
                            for col in 0..n_inputs {
                                terminals.push(TsrRow {
                                    name: format!("x{col}@{out}"),
                                    semantic_id: instance.to_string(),
                                    constructor: constructor.to_string(),
                                    inputs: Vec::new(),
                                    output,
                                    kind: TsrKind::Var(col),
                                });
                            }
                        }
                        "const" => terminals.push(TsrRow {
                            name: format!("?@{out}"),
                            semantic_id: instance.to_string(),
                            constructor: constructor.to_string(),
                            inputs: Vec::new(),
                            output,
                            kind: TsrKind::Const,
                        }),
                        other => return Err(format!("{}: terminal {other} is neither var nor const", variant.dir())),
                    }
                }
                "absent" | "canon" => notes.push(TsrNote {
                    class: class.to_string(),
                    instance: instance.to_string(),
                    note: f.get(7).copied().unwrap_or("").to_string(),
                }),
                other => return Err(format!("{}: unknown class {other}", variant.dir())),
            }
        }
        let mut rows = functions;
        rows.extend(twins);
        rows.extend(terminals);
        Ok(TsrKingdom { variant, n_inputs, duals, rows, notes })
    }

    /// The dual's name, `None` for a type outside this kingdom.
    pub fn dual(&self, ty: Ty) -> Option<&TsrDual> {
        match ty {
            Ty::Tsr(i) => self.duals.get(i as usize),
            _ => None,
        }
    }

    /// The root dual.
    pub fn root(&self) -> Option<Ty> {
        self.duals.iter().position(|d| d.root).map(|i| Ty::Tsr(i as u16))
    }

    /// The rows as a `SymbolTable` kingdom: `symbol = i + 1` for a function
    /// (twins included), `-(i + 1)` for a terminal, many-hot arity keyed by
    /// `Ty::Tsr`.
    pub fn symbol_table(&self) -> SymbolTable {
        let mut t = SymbolTable::new();
        let (mut nf, mut nt) = (0i64, 0i64);
        for r in &self.rows {
            let mut arity = Arity::default();
            for i in &r.inputs {
                *arity.inputs.entry(*i).or_default() += 1;
            }
            arity.outputs.insert(r.output, 1);
            let symbol = if r.terminal() {
                nt += 1;
                -nt
            } else {
                nf += 1;
                nf
            };
            let alias = r.name.split_once('@').map_or(r.name.clone(), |(a, _)| a.to_string());
            t.push(Symbol {
                kingdom: TSR.to_string(),
                symbol,
                symbol_name: r.name.clone(),
                alias,
                semantic_id: r.semantic_id.clone(),
                arity,
            });
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_INPUTS: usize = 4;

    /// (variant, duals, function rows, root twins) from the generator's report.
    const EXPECTED: [(TsrVariant, usize, usize, usize); 5] = [
        (TsrVariant::Single, 1, 18, 0),
        (TsrVariant::Order, 12, 574, 574),
        (TsrVariant::Depth, 4, 68, 68),
        (TsrVariant::OrderDepth, 34, 4132, 4132),
        (TsrVariant::OrderDepthWindows, 34, 4047, 4047),
    ];

    fn count(k: &TsrKingdom, kind: fn(&TsrKind) -> bool) -> usize {
        k.rows.iter().filter(|r| kind(&r.kind)).count()
    }

    #[test]
    fn every_variant_loads_with_the_generators_counts() {
        for (variant, duals, functions, twins) in EXPECTED {
            let k = TsrKingdom::load(variant, N_INPUTS).unwrap();
            let d = variant.dir();
            assert_eq!(k.duals.len(), duals, "{d} duals");
            assert_eq!(count(&k, |t| *t == TsrKind::Function), functions, "{d} function rows");
            assert_eq!(count(&k, |t| *t == TsrKind::RootTwin), twins, "{d} root twins");
            let terminals = k.rows.iter().filter(|r| r.terminal()).count();
            assert_eq!(terminals, duals * (N_INPUTS + 1), "{d} terminals");
            assert_eq!(k.rows.len(), functions + twins + terminals, "{d} rows");
            assert_eq!(k.notes.len(), 11, "{d}: 9 absent and 2 canon");
            let st = k.symbol_table();
            assert_eq!(st.len(), k.rows.len(), "{d} symbols");
            assert_eq!(st.kingdoms(), vec![TSR.to_string()]);
            assert_eq!(st.kingdom(TSR).iter().filter(|s| s.symbol > 0).count(), functions + twins);
            assert_eq!(st.max_arity(TSR), 2);
        }
    }

    #[test]
    fn the_root_dual_is_last_and_alone_and_only_single_takes_it() {
        for variant in TsrVariant::ALL {
            let k = TsrKingdom::load(variant, N_INPUTS).unwrap();
            let roots: Vec<usize> = k.duals.iter().enumerate().filter(|(_, d)| d.root).map(|(i, _)| i).collect();
            assert_eq!(roots, vec![k.duals.len() - 1], "{}", variant.dir());
            let root = k.root().unwrap();
            // The single kingdom's one dual is the root and is taken by every row.
            if variant != TsrVariant::Single {
                assert!(k.rows.iter().all(|r| !r.inputs.contains(&root)), "{}: a row takes ROOT", variant.dir());
            }
            for r in k.rows.iter().filter(|r| r.kind == TsrKind::RootTwin) {
                assert_eq!(r.output, root, "{}", r.name);
            }
        }
    }

    #[test]
    fn a_twin_copies_a_function_row_but_its_output() {
        for variant in TsrVariant::ALL {
            let k = TsrKingdom::load(variant, N_INPUTS).unwrap();
            let functions: Vec<&TsrRow> = k.rows.iter().filter(|r| r.kind == TsrKind::Function).collect();
            let twins: Vec<&TsrRow> = k.rows.iter().filter(|r| r.kind == TsrKind::RootTwin).collect();
            for (f, t) in functions.iter().zip(twins.iter()) {
                assert_eq!((&f.semantic_id, &f.constructor, &f.inputs), (&t.semantic_id, &t.constructor, &t.inputs), "{}", t.name);
            }
        }
    }

    #[test]
    fn every_demanded_dual_has_a_variable_per_column_and_a_constant() {
        for variant in TsrVariant::ALL {
            let k = TsrKingdom::load(variant, N_INPUTS).unwrap();
            let d = variant.dir();
            let mut demanded: Vec<Ty> = k.rows.iter().flat_map(|r| r.inputs.iter().copied()).collect();
            demanded.push(k.root().unwrap());
            demanded.sort();
            demanded.dedup();
            for i in 0..k.duals.len() {
                let ty = Ty::Tsr(i as u16);
                for col in 0..N_INPUTS {
                    assert!(
                        k.rows.iter().any(|r| r.kind == TsrKind::Var(col) && r.output == ty),
                        "{d}: no variable {col} at {}",
                        k.duals[i].dual
                    );
                }
                assert!(k.rows.iter().any(|r| r.kind == TsrKind::Const && r.output == ty), "{d}: no ? at {}", k.duals[i].dual);
            }
            for ty in demanded {
                assert!(k.dual(ty).is_some(), "{d}: demanded {ty:?} is outside the kingdom");
            }
        }
    }

    #[test]
    fn symbol_names_are_unique_and_the_input_count_scales_the_terminals() {
        for variant in TsrVariant::ALL {
            let k = TsrKingdom::load(variant, N_INPUTS).unwrap();
            let mut names: Vec<&str> = k.rows.iter().map(|r| r.name.as_str()).collect();
            names.sort_unstable();
            let n = names.len();
            names.dedup();
            assert_eq!(names.len(), n, "{}: duplicate row names", variant.dir());
            let one = TsrKingdom::load(variant, 1).unwrap();
            assert_eq!(one.rows.iter().filter(|r| r.terminal()).count(), k.duals.len() * 2);
        }
    }
}
