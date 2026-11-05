//! THE GENE FRAME'S CONFIGURATION TABLES (design `docs/DESIGN_futhark_types_and_properties.md` section 3.3, after the
//! external review of 2026-11-05): three lists that name things (Symbol, DataType, the Property catalogue) and FOUR
//! KINDS OF ROW that say everything about them:
//!
//! | kind of row | what it holds |
//! |---|---|
//! | [`Argument`] | what the gene may hold: the only table the GPU's model is built from |
//! | [`Flow`] | a fact moved through the tree: a property up, an attribute or a variable down (a binding) |
//! | [`Guard`] | a condition, `must` (the decoder repairs it) or `advise` (the cost charges it), its repairs inline |
//! | [`Entry`] | the program's inputs (the root environment), their properties with their sources, the result |
//!
//! A kingdom is configured only by rows (Andrew Morgan, 2026-11-05). Arity is DERIVED from the Argument rows (design
//! 3.5), so Karva's frames are sized exactly as before; Flow and Guard rows are read only after Karva has expressed the
//! tree, on the host. A kingdom whose data types carry no property and that has no Flow or Guard row is today's kingdom.
//! The schema freezes when the Futhark kingdom and TSR both run end to end on it (design 3, freeze rule).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use super::formula::{is_identifier, Condition, PropertyExpr, Reference, Relation, SizeBounds};
use super::{Arity, Symbol, SymbolTable, Ty};

fn bug(kingdom: &str, what: String) -> String {
    format!("our software has a bug: kingdom {kingdom}: {what}")
}

// ---- the lists that name things -----------------------------------------------------------------------------------

/// What a symbol is to the gene (Symbol.role): an ordinary function or terminal, the constant placeholder `?_T` (a Dc
/// word), the variable terminal `var.T` (a Dv word), or a FORMULA OPERATOR's spelling in the target language (`=min`,
/// `=+`, `=group`, ...: never in the alphabet, no Argument row; the template engine's `{=formula}` reads it, code
/// review 2026-11-05 finding 3). A kingdom lists its formula rows after its alphabet, so the contract's symbol ids are
/// the tables' ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    Function,
    Constant,
    Variable,
    Formula,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::Function, Role::Constant, Role::Variable, Role::Formula];

    pub fn name(self) -> &'static str {
        match self {
            Role::Function => "function",
            Role::Constant => "constant",
            Role::Variable => "variable",
            Role::Formula => "formula",
        }
    }

    pub fn from_name(s: &str) -> Option<Role> {
        Role::ALL.into_iter().find(|r| r.name() == s)
    }
}

/// One Symbol: a function or terminal with its role, whether it is an adapter (an ordinary function evolution may place,
/// which a repair's template may also call; the cost charges it either way), its spellings by form (the empty form is
/// the default) and its prelude (text added once to a program that uses it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolRow {
    pub symbol: i64,
    pub symbol_name: String,
    pub alias: String,
    pub semantic_id: String,
    pub role: Role,
    pub adapter: bool,
    pub spellings: Vec<(String, String)>,
    pub prelude: String,
}

impl SymbolRow {
    /// A function named `name` (its alias and semantic id too), with no spelling.
    pub fn named(symbol: i64, name: &str) -> SymbolRow {
        SymbolRow { symbol, symbol_name: name.to_string(), alias: name.to_string(), semantic_id: name.to_string(), role: Role::Function, adapter: false, spellings: Vec::new(), prelude: String::new() }
    }

    /// The row of an existing [`Symbol`], its arity dropped, as a function.
    pub fn of(s: &Symbol) -> SymbolRow {
        SymbolRow { alias: s.alias.clone(), semantic_id: s.semantic_id.clone(), ..SymbolRow::named(s.symbol, &s.symbol_name) }
    }

    /// Its spelling in `form`, if it has one.
    pub fn spelling(&self, form: &str) -> Option<&str> {
        self.spellings.iter().find(|(f, _)| f == form).map(|(_, t)| t.as_str())
    }
}

/// What a data type is made of. The members are data types of the same table, by their [`Ty`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataKind {
    /// A value with no parts: `f32`, `i64`, a regex `Pattern`.
    Scalar,
    /// An array of `element`. Its length is a property, not a type.
    Array { element: Ty },
    /// A tuple of its components, in order.
    Tuple { components: Vec<Ty> },
    /// A function from its parameters, in order, to its result.
    Function { parameters: Vec<Ty>, result: Ty },
}

impl DataKind {
    /// The data types this kind names.
    pub fn members(&self) -> Vec<Ty> {
        match self {
            DataKind::Scalar => Vec::new(),
            DataKind::Array { element } => vec![*element],
            DataKind::Tuple { components } => components.clone(),
            DataKind::Function { parameters, result } => parameters.iter().copied().chain(std::iter::once(*result)).collect(),
        }
    }
}

/// A data type's constant: what a `var.T` gives with no variable of T in scope, and what a `{const}` substitute
/// renders as. Its spelling and its properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constant {
    pub spelling: String,
    pub properties: Vec<(String, PropertyExpr)>,
}

/// One DataType: as the target language spells it without sizes, the catalogue properties it carries, its constant
/// (none for Futhark's `size`), and, for the data type that holds sizes, the size names with their lower bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataType {
    pub id: Ty,
    pub name: String,
    pub kind: DataKind,
    pub properties: Vec<String>,
    pub constant: Option<Constant>,
    pub size_names: Vec<(String, i64)>,
}

impl DataType {
    /// A scalar data type with no property, no constant and no size name.
    pub fn scalar(id: Ty, name: &str) -> DataType {
        DataType { id, name: name.to_string(), kind: DataKind::Scalar, properties: Vec::new(), constant: None, size_names: Vec::new() }
    }
}

/// What kind of fact a property is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropertyKind {
    /// A count of elements: a length, a size value.
    Size,
    /// The exclusive upper bound of an index.
    Bound,
    /// An integer on an ordered scale: a depth, an order, a width.
    Range,
    /// One name from an unordered set: an address space, a grammatical number.
    Tag,
    /// True or false.
    Flag,
    /// An ordered list of one element kind (host only, design 3.4.5).
    List,
}

impl PropertyKind {
    pub const ALL: [PropertyKind; 6] = [PropertyKind::Size, PropertyKind::Bound, PropertyKind::Range, PropertyKind::Tag, PropertyKind::Flag, PropertyKind::List];

    pub fn name(self) -> &'static str {
        match self {
            PropertyKind::Size => "size",
            PropertyKind::Bound => "bound",
            PropertyKind::Range => "range",
            PropertyKind::Tag => "tag",
            PropertyKind::Flag => "flag",
            PropertyKind::List => "list",
        }
    }

    pub fn from_name(name: &str) -> Option<PropertyKind> {
        PropertyKind::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// One row of the Property catalogue: a property defined ONCE for every kingdom; an environment attribute (`trips`,
/// `uniform`) is one too, with its root value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Property {
    pub name: String,
    pub kind: PropertyKind,
    pub element: Option<PropertyKind>,
    pub values: Vec<String>,
    pub root: Option<PropertyExpr>,
    pub meaning: String,
}

/// The catalogue's text: `kingdoms/master/properties.tsv` (the one table of the master set with no `kingdom` column:
/// the catalogue is shared by every kingdom, design 3.3.1).
pub const CATALOGUE_TSV: &str = include_str!("../../kingdoms/master/properties.tsv");

/// Read a catalogue from its text (a header, then one property a line: name, kind, element, values, root, meaning).
pub fn parse_catalogue(text: &str) -> Result<Vec<Property>, String> {
    let bad = |what: String| format!("our software has a bug: the property catalogue: {what}");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    if header != ["name", "kind", "element", "values", "root", "meaning"] {
        return Err(bad(format!("the header is {header:?}")));
    }
    let mut out: Vec<Property> = Vec::new();
    for line in lines.filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 6 {
            return Err(bad(format!("the row {line:?} has {} columns, not 6", f.len())));
        }
        let kind = PropertyKind::from_name(f[1]).ok_or_else(|| bad(format!("{} is of the kind {:?}", f[0], f[1])))?;
        let element = if f[2].is_empty() { None } else { Some(PropertyKind::from_name(f[2]).ok_or_else(|| bad(format!("{}'s element kind {:?}", f[0], f[2])))?) };
        if (kind == PropertyKind::List) != element.is_some() || element == Some(PropertyKind::List) {
            return Err(bad(format!("{} is of the kind {} with the element {:?}: a list has one scalar element kind, nothing else has one", f[0], f[1], f[2])));
        }
        let values: Vec<String> = f[3].split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
        if kind != PropertyKind::Tag && element != Some(PropertyKind::Tag) && !values.is_empty() {
            return Err(bad(format!("{} lists values, and only a tag has them", f[0])));
        }
        let root = if f[4].is_empty() { None } else { Some(PropertyExpr::parse(f[4])?) };
        if !is_identifier(f[0]) || out.iter().any(|p| p.name == f[0]) {
            return Err(bad(format!("the name {:?} is not an identifier or is listed twice", f[0])));
        }
        out.push(Property { name: f[0].to_string(), kind, element, values, root, meaning: f[5].to_string() });
    }
    Ok(out)
}

/// THE PROPERTY CATALOGUE, one for every kingdom, read once from `kingdoms/master/properties.tsv` (its test reads it too, so a
/// malformed catalogue fails `cargo test`). A malformed catalogue is our software's bug and stops the process: an empty
/// one would read every property as undefined (code review 2026-11-05, finding 14).
pub fn catalogue() -> &'static [Property] {
    static CATALOGUE: OnceLock<Vec<Property>> = OnceLock::new();
    CATALOGUE.get_or_init(|| parse_catalogue(CATALOGUE_TSV).unwrap_or_else(|e| panic!("{e}")))
}

/// The catalogue's row for `name`.
pub fn property(name: &str) -> Option<&'static Property> {
    catalogue().iter().find(|p| p.name == name)
}

// ---- the four kinds of row ----------------------------------------------------------------------------------------

/// Whether an argument is an input or the output of its symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    In,
    Out,
}

/// ARGUMENT: one ordered input, or the output, of a symbol. A symbol's inputs are at positions `0..n` with no gap; its
/// one output is at position 0. `head` marks the input that receives a form asked of the node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub symbol: i64,
    pub position: u32,
    pub direction: Direction,
    pub data_type: Ty,
    pub head: bool,
}

/// Whose a Flow or Guard row is: a symbol's, or the entry's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Owner {
    Symbol(i64),
    Entry,
}

/// Where a Flow row acts: argument k's subtree (down), or the result (up; or the entry's root, down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Place {
    In(u32),
    Out,
}

/// The fact a Flow row moves.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FlowName {
    /// `p`: a property of the result (up).
    Property(String),
    /// `env.a`: an attribute of the argument's environment (down).
    Env(String),
    /// `var.T[/g]`: `formula`-many variables of T introduced into the argument's environment (down; a binding).
    Var { ty: Ty, group: u32 },
    /// `var.T[/g].p`: property p of each variable of that group; may read `each`, the variable's ordinal.
    VarProperty { ty: Ty, group: u32, property: String },
    /// `var.T[/g].spell`: the variables' spelling, a template (`v{depth}_{each}`).
    VarSpell { ty: Ty, group: u32 },
}

impl FlowName {
    fn text(&self) -> String {
        let var = |ty: &Ty, group: &u32| if *group == 0 { format!("var.{}", ty.code()) } else { format!("var.{}/{group}", ty.code()) };
        match self {
            FlowName::Property(p) => p.clone(),
            FlowName::Env(a) => format!("env.{a}"),
            FlowName::Var { ty, group } => var(ty, group),
            FlowName::VarProperty { ty, group, property } => format!("{}.{property}", var(ty, group)),
            FlowName::VarSpell { ty, group } => format!("{}.spell", var(ty, group)),
        }
    }

    fn parse(text: &str) -> Option<FlowName> {
        if let Some(a) = text.strip_prefix("env.") {
            return is_identifier(a).then(|| FlowName::Env(a.to_string()));
        }
        if let Some(rest) = text.strip_prefix("var.") {
            let (head, tail) = rest.split_once('.').map_or((rest, None), |(h, t)| (h, Some(t)));
            let (code, group) = head.split_once('/').map_or((head, "0"), |(c, g)| (c, g));
            let ty = code.parse::<u32>().ok().and_then(Ty::from_code)?;
            let group = group.parse().ok()?;
            return match tail {
                None => Some(FlowName::Var { ty, group }),
                Some("spell") => Some(FlowName::VarSpell { ty, group }),
                Some(p) if is_identifier(p) => Some(FlowName::VarProperty { ty, group, property: p.to_string() }),
                Some(_) => None,
            };
        }
        is_identifier(text).then(|| FlowName::Property(text.to_string()))
    }
}

/// A flow's direction: properties go up, environments and variables down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Way {
    Up,
    Down,
}

/// FLOW: one fact moved through the tree (design 3.3.3). Up, at `out`: the result's property `p` := formula over
/// `in<k>.p`, `out.q` (acyclic) and `env.a`. Down, at `in<k>`: an attribute of argument k's subtree, or the variables it
/// introduces there (a binding), with their properties and their spelling. Down, of the entry at `out`: an attribute's
/// root value. `template` holds a `var.T.spell` flow's template, `formula` every other flow's value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flow {
    pub owner: Owner,
    pub at: Place,
    pub name: FlowName,
    pub formula: PropertyExpr,
    pub template: String,
    pub way: Way,
}

impl Flow {
    /// An up flow of `symbol`: the result's `property` := `formula`.
    pub fn up(symbol: i64, property: &str, formula: PropertyExpr) -> Flow {
        Flow { owner: Owner::Symbol(symbol), at: Place::Out, name: FlowName::Property(property.to_string()), formula, template: String::new(), way: Way::Up }
    }

    /// A down flow of `symbol` into argument `k`.
    pub fn down(symbol: i64, k: u32, name: FlowName, formula: PropertyExpr) -> Flow {
        Flow { owner: Owner::Symbol(symbol), at: Place::In(k), name, formula, template: String::new(), way: Way::Down }
    }
}

/// A guard's strength, which picks its engine: `must` (the decoder repairs it), `advise` (the cost charges it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strength {
    Must,
    Advise,
}

impl Strength {
    pub fn name(self) -> &'static str {
        match self {
            Strength::Must => "must",
            Strength::Advise => "advise",
        }
    }

    pub fn from_name(s: &str) -> Option<Strength> {
        [Strength::Must, Strength::Advise].into_iter().find(|x| x.name() == s)
    }
}

/// Where a repair runs: at the node, at its operand k (whose environment the repair's guard reads), or at the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    Node,
    In(u32),
    Root,
}

impl std::fmt::Display for Site {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Site::Node => write!(f, "node"),
            Site::In(k) => write!(f, "in{k}"),
            Site::Root => write!(f, "root"),
        }
    }
}

impl Site {
    pub fn parse(s: &str) -> Option<Site> {
        match s {
            "node" => Some(Site::Node),
            "root" => Some(Site::Root),
            _ => s.strip_prefix("in").filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).and_then(|d| d.parse().ok()).map(Site::In),
        }
    }
}

/// One repair of a `must` guard, inline: its site, its template (a wrap, an operand rewrite, or one substitute that
/// replaces the node; the adapter symbols it calls are its `{call ..}` placeholders, each an adapter Symbol, whose
/// prelude is added once and which the cost charges), its `when` (choosing the text) and its guard (read in the site's
/// environment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repair {
    pub site: Site,
    pub template: String,
    pub when: Condition,
    pub guard: Condition,
}

/// The condition `true`: a repair with no `when` or no guard.
pub fn always() -> Condition {
    Condition::Flag(PropertyExpr::Flag(true))
}

impl Repair {
    /// A repair at `site` by `template`, with no `when` and no guard.
    pub fn new(site: Site, template: &str) -> Repair {
        Repair { site, template: template.to_string(), when: always(), guard: always() }
    }

    /// True when the repair always applies: its guard and its `when` are both `true`.
    pub fn unconditional(&self) -> bool {
        self.guard == always() && self.when == always()
    }
}

/// GUARD: a condition on a symbol's node (or, of the entry, on the result `out`), checked in `order`; `must` with its
/// ordered repairs (the first whose guard and `when` hold is used), or `advise` with none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guard {
    pub owner: Owner,
    pub order: u32,
    pub condition: Condition,
    pub strength: Strength,
    pub repairs: Vec<Repair>,
}

/// Where a property value came from: `declared` holds for every input; `measured on <case set>` only for the cases
/// seen, so it never makes a `must` guard hold.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    Declared,
    Measured(String),
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Declared => write!(f, "declared"),
            Source::Measured(cases) => write!(f, "measured on {cases}"),
        }
    }
}

/// A transform: how an alternative form is made from its raw parameter, once, at the program's head (`{0}` the raw's
/// name, `{name}` the form's), with its properties from the raw's (`in0.p`) and its cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transform {
    pub template: String,
    pub properties: Vec<(String, PropertyExpr)>,
    pub cost: PropertyExpr,
}

/// ENTRY, one row a parameter: its name, data type, properties (each with its source), and for an alternative form its
/// raw parameter and transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryParameter {
    pub name: String,
    pub data_type: Ty,
    pub properties: Vec<(String, PropertyExpr, Source)>,
    pub form_of: Option<(String, Transform)>,
}

impl EntryParameter {
    /// A raw parameter whose properties are all declared.
    pub fn declared(name: &str, data_type: Ty, properties: &[(&str, PropertyExpr)]) -> EntryParameter {
        EntryParameter { name: name.to_string(), data_type, properties: properties.iter().map(|(p, v)| (p.to_string(), v.clone(), Source::Declared)).collect(), form_of: None }
    }
}

/// THE ENTRY of one problem: the result's data type (its `out` row), the parameters (the root environment, raw values
/// and alternative forms, in order), the Guard rows of `entry` on the result, and its down flows (attribute values
/// overriding the catalogue's roots).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub problem: String,
    pub output: Ty,
    pub parameters: Vec<EntryParameter>,
    pub guards: Vec<Guard>,
    pub flows: Vec<Flow>,
}

impl Entry {
    /// An entry with no parameter, guard or flow.
    pub fn new(problem: &str, output: Ty) -> Entry {
        Entry { problem: problem.to_string(), output, parameters: Vec::new(), guards: Vec::new(), flows: Vec::new() }
    }
}

/// One kingdom's configuration: its symbols and data types, and its Argument, Flow and Guard rows. (Its entries are
/// per problem: [`Entry`].)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KingdomTables {
    pub kingdom: String,
    pub symbols: Vec<SymbolRow>,
    pub data_types: Vec<DataType>,
    pub arguments: Vec<Argument>,
    pub flows: Vec<Flow>,
    pub guards: Vec<Guard>,
}

impl Arity {
    /// THE DERIVATION (design 3.5): a symbol's many-hot arity is the count of its Argument rows of each data type,
    /// inputs and output apart.
    pub fn from_arguments<'a>(arguments: impl IntoIterator<Item = &'a Argument>) -> Arity {
        let mut a = Arity::default();
        for arg in arguments {
            let side = match arg.direction {
                Direction::In => &mut a.inputs,
                Direction::Out => &mut a.outputs,
            };
            *side.entry(arg.data_type).or_insert(0) += 1;
        }
        a
    }
}

// ---- the table check's kind rules (design 3.4.4) ------------------------------------------------------------------

/// The comparability classes of property kinds: a size and a bound both count elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Count,
    Range,
    Tag,
    Flag,
    List,
}

impl Class {
    fn of(k: PropertyKind) -> Class {
        match k {
            PropertyKind::Size | PropertyKind::Bound => Class::Count,
            PropertyKind::Range => Class::Range,
            PropertyKind::Tag => Class::Tag,
            PropertyKind::Flag => Class::Flag,
            PropertyKind::List => Class::List,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Class::Count => "size or bound",
            Class::Range => "range",
            Class::Tag => "tag",
            Class::Flag => "flag",
            Class::List => "list",
        }
    }
}

/// How the table check reads a reference in one place: the property it names, `Ok(None)` for one of no kind (an
/// index), or `Err` with why it may not be read there.
type Resolve<'a> = dyn Fn(&Reference) -> Result<Option<&'static Property>, String> + 'a;

/// The classes a formula's value carries (an integer literal, `?` and an index carry none).
fn classes(e: &PropertyExpr, resolve: &Resolve) -> BTreeSet<Class> {
    use PropertyExpr as P;
    let mut out = BTreeSet::new();
    let mut of = |r: Reference| {
        if let Ok(Some(p)) = resolve(&r) {
            out.insert(Class::of(p.kind));
        }
    };
    match e {
        P::Input { position, property } => of(Reference::Input { position: *position, property }),
        P::Output(p) => of(Reference::Output(p)),
        P::Env(a) => of(Reference::Env(a)),
        _ => {}
    }
    match e {
        P::Size(_) | P::Len(_) => {
            out.insert(Class::Count);
        }
        P::Tag(_) => {
            out.insert(Class::Tag);
        }
        P::Flag(_) | P::Cond(_) => {
            out.insert(Class::Flag);
        }
        P::List(_) | P::Concat(_) | P::For { .. } => {
            out.insert(Class::List);
        }
        P::Sum(xs) | P::Product(xs) | P::Min(xs) | P::Max(xs) => xs.iter().for_each(|x| out.extend(classes(x, resolve))),
        P::If { then, otherwise, .. } => {
            out.extend(classes(then, resolve));
            out.extend(classes(otherwise, resolve));
        }
        _ => {}
    }
    out
}

/// Every condition inside a formula (an `if`'s, a flag value's), at any depth.
fn conditions_in<'a>(e: &'a PropertyExpr, out: &mut Vec<&'a Condition>) {
    use PropertyExpr as P;
    match e {
        P::If { cond, then, otherwise } => {
            out.push(cond);
            conditions_in(then, out);
            conditions_in(otherwise, out);
        }
        P::Cond(c) => out.push(c),
        P::Sum(xs) | P::Product(xs) | P::Min(xs) | P::Max(xs) | P::List(xs) | P::Concat(xs) => xs.iter().for_each(|x| conditions_in(x, out)),
        P::Len(x) => conditions_in(x, out),
        P::At(a, b, c) => [a, b, c].into_iter().for_each(|x| conditions_in(x, out)),
        P::Find(a, b) => [a, b].into_iter().for_each(|x| conditions_in(x, out)),
        P::For { count, body, .. } => [count, body].into_iter().for_each(|x| conditions_in(x, out)),
        _ => {}
    }
}

/// The kind defects of every comparison in a condition (3.4.4): sides of different kinds; a tag, a flag or a list by a
/// relation other than `=`; a bound by a relation other than `<=`; a tag literal its property does not list.
fn comparison_defects(at: &str, c: &Condition, resolve: &Resolve) -> Vec<String> {
    let mut v = Vec::new();
    let named = |e: &PropertyExpr| -> Vec<&'static Property> {
        match e {
            PropertyExpr::Input { position, property } => resolve(&Reference::Input { position: *position, property }).ok().flatten().into_iter().collect(),
            PropertyExpr::Output(p) => resolve(&Reference::Output(p)).ok().flatten().into_iter().collect(),
            PropertyExpr::Env(a) => resolve(&Reference::Env(a)).ok().flatten().into_iter().collect(),
            _ => Vec::new(),
        }
    };
    match c {
        Condition::Compare { relation, left, right } => {
            let mut cl = classes(left, resolve);
            cl.extend(classes(right, resolve));
            if cl.len() > 1 {
                v.push(format!("{at} compares {left} with {right}, which mixes kinds ({})", cl.iter().map(|c| c.name()).collect::<Vec<_>>().join(", ")));
            }
            if *relation != Relation::Eq && cl.iter().any(|c| matches!(c, Class::Tag | Class::Flag | Class::List)) {
                v.push(format!("{at} compares {left} {} {right}, and a tag, a flag or a list compares by = only", relation.symbol()));
            }
            let kinds: Vec<&Property> = named(left).into_iter().chain(named(right)).collect();
            if *relation != Relation::Le && kinds.iter().any(|p| p.kind == PropertyKind::Bound) {
                v.push(format!("{at} compares {left} {} {right}, and a bound compares by <= only", relation.symbol()));
            }
            for (side, other) in [(left, right), (right, left)] {
                if let PropertyExpr::Tag(t) = side {
                    for p in named(other) {
                        if !p.values.is_empty() && !p.values.contains(t) {
                            v.push(format!("{at} compares {} with the tag '{t}', which it does not list", p.name));
                        }
                    }
                }
            }
            for x in [left, right] {
                let mut inner = Vec::new();
                conditions_in(x, &mut inner);
                inner.into_iter().for_each(|c| v.extend(comparison_defects(at, c, resolve)));
            }
        }
        Condition::And(cs) | Condition::Or(cs) => cs.iter().for_each(|c| v.extend(comparison_defects(at, c, resolve))),
        Condition::Not(c) => v.extend(comparison_defects(at, c, resolve)),
        Condition::All { body, .. } => v.extend(comparison_defects(at, body, resolve)),
        Condition::In { .. } | Condition::Known(_) | Condition::Flag(_) => {}
    }
    v
}

/// Every defect of a formula read in one place: a reference that may not be read there, a value not in its one spelling
/// or not reading back from its text, and the kind defects of its conditions.
fn formula_defects(at: &str, e: &PropertyExpr, resolve: &Resolve) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for r in e.references() {
        if let Err(why) = resolve(&r) {
            v.push(format!("{at} reads {r}, {why}"));
        }
    }
    v.extend(e.shape_defects().into_iter().map(|what| format!("{at} {what}")));
    if PropertyExpr::parse(&e.to_string()).as_ref() != Ok(e) {
        v.push(format!("{at} has {e}, which does not read back from its text"));
    }
    let mut conds = Vec::new();
    conditions_in(e, &mut conds);
    conds.into_iter().for_each(|c| v.extend(comparison_defects(at, c, resolve)));
    v
}

/// Every defect of a condition read in one place (as [`formula_defects`]).
fn condition_defects(at: &str, c: &Condition, resolve: &Resolve) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for r in c.references() {
        if let Err(why) = resolve(&r) {
            v.push(format!("{at} reads {r}, {why}"));
        }
    }
    v.extend(c.shape_defects().into_iter().map(|what| format!("{at} {what}")));
    if Condition::parse(&c.to_string()).as_ref() != Ok(c) {
        v.push(format!("{at} has the condition {c}, which does not read back from its text"));
    }
    v.extend(comparison_defects(at, c, resolve));
    v
}

// ---- the text form ------------------------------------------------------------------------------------------------

/// Text fields hold no tab and no line break; the text form is one row a line, its fields tab-separated.
fn field(kind: &str, s: &str) -> Result<String, String> {
    if s.contains(['\t', '\n', '\r']) {
        return Err(format!("our software has a bug: the {kind} {s:?} holds a tab or a line break, which the text form cannot carry"));
    }
    Ok(s.to_string())
}

fn codes(tys: &[Ty]) -> String {
    if tys.is_empty() {
        return "-".to_string();
    }
    tys.iter().map(|t| t.code().to_string()).collect::<Vec<_>>().join(",")
}

fn ty_of(text: &str) -> Result<Ty, String> {
    text.parse::<u32>().ok().and_then(Ty::from_code).ok_or_else(|| format!("our software has a bug: {text:?} is not a type code"))
}

fn tys_of(text: &str) -> Result<Vec<Ty>, String> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(',').map(ty_of).collect()
}

/// `p=e;q=f`, or `-` for none.
fn property_list(props: &[(String, PropertyExpr)]) -> String {
    if props.is_empty() {
        return "-".to_string();
    }
    props.iter().map(|(p, e)| format!("{p}={e}")).collect::<Vec<_>>().join(";")
}

fn parse_property_list(text: &str) -> Result<Vec<(String, PropertyExpr)>, String> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(';')
        .map(|pe| {
            let (p, e) = pe.split_once('=').ok_or_else(|| format!("our software has a bug: {pe:?} is not <property>=<formula>"))?;
            Ok((p.to_string(), PropertyExpr::parse(e)?))
        })
        .collect()
}

fn number<T: std::str::FromStr>(what: &str, text: &str) -> Result<T, String> {
    text.parse::<T>().map_err(|_| format!("our software has a bug: the {what} {text:?} is not a number"))
}

fn owner_text(o: Owner) -> String {
    match o {
        Owner::Symbol(s) => s.to_string(),
        Owner::Entry => "entry".to_string(),
    }
}

fn owner_of(text: &str) -> Result<Owner, String> {
    if text == "entry" {
        Ok(Owner::Entry)
    } else {
        number("symbol id", text).map(Owner::Symbol)
    }
}

/// The order of `n` items when each must come after those it depends on (`after[i]`), the smallest ready first; `None`
/// on a cycle.
fn topological(n: usize, after: &[BTreeSet<usize>]) -> Option<Vec<usize>> {
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut done = vec![false; n];
    while order.len() < n {
        let next = (0..n).find(|i| !done[*i] && after[*i].iter().all(|j| *j >= n || done[*j]))?;
        done[next] = true;
        order.push(next);
    }
    Some(order)
}

impl KingdomTables {
    /// The tables of a kingdom with no property rows, from its symbols with their ordered inputs and output: one
    /// Argument row a slot, and one scalar data type for each type that occurs, named by its `Debug` text.
    pub fn from_signatures(kingdom: &str, rows: Vec<(SymbolRow, Vec<Ty>, Ty)>) -> KingdomTables {
        let mut t = KingdomTables { kingdom: kingdom.to_string(), ..KingdomTables::default() };
        let mut seen: BTreeSet<Ty> = BTreeSet::new();
        for (row, ins, out) in rows {
            for (i, ty) in ins.iter().enumerate() {
                t.arguments.push(Argument { symbol: row.symbol, position: i as u32, direction: Direction::In, data_type: *ty, head: false });
            }
            t.arguments.push(Argument { symbol: row.symbol, position: 0, direction: Direction::Out, data_type: out, head: false });
            seen.extend(ins);
            seen.insert(out);
            t.symbols.push(row);
        }
        t.data_types = seen.into_iter().map(|ty| DataType::scalar(ty, &format!("{ty:?}"))).collect();
        t
    }

    /// True when the kingdom has any property, Flow or Guard row: false is today's kingdom.
    pub fn has_properties(&self) -> bool {
        !(self.data_types.iter().all(|d| d.properties.is_empty()) && self.flows.is_empty() && self.guards.is_empty())
    }

    /// The Argument rows grouped by symbol, inputs in position order then the output.
    fn arguments_by_symbol(&self) -> BTreeMap<i64, Vec<&Argument>> {
        let mut by: BTreeMap<i64, Vec<&Argument>> = BTreeMap::new();
        for a in &self.arguments {
            by.entry(a.symbol).or_default().push(a);
        }
        for args in by.values_mut() {
            args.sort_by_key(|a| (a.direction, a.position));
        }
        by
    }

    /// A symbol's input data types in argument order.
    pub fn inputs(&self, symbol: i64) -> Vec<Ty> {
        let mut ins: Vec<&Argument> = self.arguments.iter().filter(|a| a.symbol == symbol && a.direction == Direction::In).collect();
        ins.sort_by_key(|a| a.position);
        ins.into_iter().map(|a| a.data_type).collect()
    }

    /// A symbol's output data type.
    pub fn output(&self, symbol: i64) -> Option<Ty> {
        self.arguments.iter().find(|a| a.symbol == symbol && a.direction == Direction::Out).map(|a| a.data_type)
    }

    /// A symbol's head input, if it has one.
    pub fn head(&self, symbol: i64) -> Option<u32> {
        self.arguments.iter().find(|a| a.symbol == symbol && a.direction == Direction::In && a.head).map(|a| a.position)
    }

    /// A symbol's row.
    pub fn symbol(&self, symbol: i64) -> Option<&SymbolRow> {
        self.symbols.iter().find(|s| s.symbol == symbol)
    }

    /// A symbol's arity, derived from its Argument rows ([`Arity::from_arguments`]).
    pub fn arity(&self, symbol: i64) -> Arity {
        Arity::from_arguments(self.arguments.iter().filter(|a| a.symbol == symbol))
    }

    /// Every symbol's ordered signature, `(symbol, inputs in position order, output)`, in symbol order. A symbol with no
    /// output row is left out (the check reports it).
    pub fn signatures(&self) -> Vec<(i64, Vec<Ty>, Ty)> {
        let by = self.arguments_by_symbol();
        self.symbols
            .iter()
            .filter_map(|s| {
                let args = by.get(&s.symbol)?;
                let out = args.iter().find(|a| a.direction == Direction::Out)?.data_type;
                let ins = args.iter().filter(|a| a.direction == Direction::In).map(|a| a.data_type).collect();
                Some((s.symbol, ins, out))
            })
            .collect()
    }

    /// The kingdom as a [`SymbolTable`], each symbol's arity derived from its Argument rows, in symbol order.
    pub fn symbol_table(&self) -> SymbolTable {
        let by = self.arguments_by_symbol();
        let mut t = SymbolTable::new();
        for s in self.symbols.iter().filter(|s| s.role != Role::Formula) {
            t.push(Symbol {
                kingdom: self.kingdom.clone(),
                symbol: s.symbol,
                symbol_name: s.symbol_name.clone(),
                alias: s.alias.clone(),
                semantic_id: s.semantic_id.clone(),
                arity: Arity::from_arguments(by.get(&s.symbol).into_iter().flatten().copied()),
            });
        }
        t
    }

    /// A data type's row.
    pub fn data_type(&self, ty: Ty) -> Option<&DataType> {
        self.data_types.iter().find(|d| d.id == ty)
    }

    /// The catalogue properties `ty` carries, sorted by name.
    pub fn properties_of(&self, ty: Ty) -> Vec<&'static Property> {
        let mut v: Vec<&'static Property> = self.data_type(ty).into_iter().flat_map(|d| d.properties.iter()).filter_map(|p| property(p)).collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// The catalogue row of property `p` when `ty` carries it.
    pub fn carried(&self, ty: Ty, p: &str) -> Option<&'static Property> {
        self.data_type(ty).filter(|d| d.properties.iter().any(|x| x == p)).and_then(|_| property(p))
    }

    /// `ty`'s constant, if it has one.
    pub fn constant_of(&self, ty: Ty) -> Option<&Constant> {
        self.data_type(ty).and_then(|d| d.constant.as_ref())
    }

    /// The sizes' lower bounds: every data type's size names.
    pub fn bounds(&self) -> SizeBounds {
        self.data_types.iter().flat_map(|d| d.size_names.iter().cloned()).collect()
    }

    /// THE LEXICON the template engine spells with (code review finding 3): each formula row's spellings by form (its
    /// name without the `=`), and each adapter symbol's default spelling and prelude, which a `{call ..}` renders.
    pub fn lexicon(&self) -> Result<crate::typed_environment::template::Lexicon, String> {
        use crate::typed_environment::template::parse_template;
        let mut lex = crate::typed_environment::template::Lexicon::default();
        for s in &self.symbols {
            if s.role == Role::Formula {
                let op = s.symbol_name.strip_prefix('=').ok_or_else(|| bug(&self.kingdom, format!("the formula row {} is not named =<operator>", s.symbol_name)))?;
                for (form, t) in &s.spellings {
                    lex.formulas.insert((op.to_string(), form.clone()), parse_template(t)?);
                }
            } else if s.adapter {
                let t = s.spelling("").ok_or_else(|| bug(&self.kingdom, format!("the adapter {} has no default spelling", s.symbol_name)))?;
                lex.adapters.insert(s.symbol_name.clone(), (parse_template(t)?, s.prelude.clone()));
            }
        }
        Ok(lex)
    }

    /// The environment attributes of the kingdom: every `env.a` a Flow sets or a Flow or Guard reads, sorted.
    pub fn attributes(&self) -> Vec<String> {
        let mut out: BTreeSet<String> = BTreeSet::new();
        let mut refs: Vec<Reference> = Vec::new();
        for f in &self.flows {
            if let FlowName::Env(a) = &f.name {
                out.insert(a.clone());
            }
            refs.extend(f.formula.references());
        }
        for g in &self.guards {
            refs.extend(g.condition.references());
            for r in &g.repairs {
                refs.extend(r.guard.references());
                refs.extend(r.when.references());
            }
        }
        out.extend(refs.into_iter().filter_map(|r| if let Reference::Env(a) = r { Some(a.to_string()) } else { None }));
        out.into_iter().collect()
    }

    /// A symbol's down flows into argument `k`, in table order.
    pub fn down_flows(&self, symbol: i64, k: u32) -> Vec<&Flow> {
        self.flows.iter().filter(|f| f.owner == Owner::Symbol(symbol) && f.at == Place::In(k)).collect()
    }

    /// The order in which the decoder visits a symbol's inputs: each input after every input its down flows read
    /// (`tabulate` binds `i` with bound = in0.value, so in0 before in1), otherwise in position order. An error when a
    /// flow reads its own argument or the flows read each other's in a cycle.
    pub fn decode_order(&self, symbol: i64) -> Result<Vec<u32>, String> {
        let n = self.inputs(symbol).len();
        let mut after: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for f in self.flows.iter().filter(|f| f.owner == Owner::Symbol(symbol) && f.way == Way::Down) {
            let Place::In(pos) = f.at else { continue };
            for (j, p) in f.formula.inputs() {
                if j == pos {
                    return Err(bug(&self.kingdom, format!("symbol {symbol}'s down flow {} into argument {j} reads in{j}.{p}, its own argument", f.name.text())));
                }
                if let Some(s) = after.get_mut(pos as usize) {
                    s.insert(j as usize);
                }
            }
        }
        topological(n, &after).map(|o| o.into_iter().map(|i| i as u32).collect()).ok_or_else(|| bug(&self.kingdom, format!("symbol {symbol}'s down flows read each other's arguments in a cycle")))
    }

    /// The order in which the decoder evaluates a symbol's up flows: each after the flows of the `out.p` it reads,
    /// otherwise in table order. Indices into the tables' flows. An error on a cycle.
    pub fn up_order(&self, symbol: i64) -> Result<Vec<usize>, String> {
        let rows: Vec<usize> = (0..self.flows.len()).filter(|i| self.flows[*i].owner == Owner::Symbol(symbol) && self.flows[*i].way == Way::Up).collect();
        let name = |i: usize| match &self.flows[i].name {
            FlowName::Property(p) => p.as_str(),
            _ => "",
        };
        let at = |p: &str| rows.iter().position(|i| name(*i) == p);
        let after: Vec<BTreeSet<usize>> = rows.iter().map(|i| self.flows[*i].formula.references().into_iter().filter_map(|r| if let Reference::Output(p) = r { at(p) } else { None }).collect()).collect();
        topological(rows.len(), &after)
            .map(|o| o.into_iter().map(|k| rows[k]).collect())
            .ok_or_else(|| bug(&self.kingdom, format!("symbol {symbol}'s up flows read each other's out.p in a cycle")))
    }

    /// THE TABLE CHECK (design 8.14.1): every defect, each "our software has a bug: ...", before any run.
    pub fn check(&self) -> Result<(), Vec<String>> {
        let k = self.kingdom.as_str();
        let mut d: Vec<String> = Vec::new();
        let mut symbols: BTreeMap<i64, &SymbolRow> = BTreeMap::new();
        for s in &self.symbols {
            if symbols.insert(s.symbol, s).is_some() {
                d.push(bug(k, format!("symbol {} ({}) is listed twice", s.symbol, s.symbol_name)));
            }
            let mut forms: BTreeSet<&str> = BTreeSet::new();
            for (form, template) in &s.spellings {
                if !forms.insert(form) {
                    d.push(bug(k, format!("symbol {} has two spellings of the form {form:?}", s.symbol_name)));
                }
                if let Err(e) = crate::typed_environment::template::parse_template(template) {
                    d.push(bug(k, format!("symbol {}'s spelling {form:?}: {e}", s.symbol_name)));
                }
            }
        }
        let mut types: BTreeMap<Ty, &DataType> = BTreeMap::new();
        let mut type_names: BTreeSet<&str> = BTreeSet::new();
        let mut size_names: BTreeSet<&str> = BTreeSet::new();
        for t in &self.data_types {
            if types.insert(t.id, t).is_some() {
                d.push(bug(k, format!("data type {:?} is listed twice", t.id)));
            }
            if !type_names.insert(t.name.as_str()) {
                d.push(bug(k, format!("two data types are named {:?}", t.name)));
            }
            let mut carried: BTreeSet<&str> = BTreeSet::new();
            for p in &t.properties {
                if property(p).is_none() {
                    d.push(bug(k, format!("data type {} carries {p}, which the property catalogue does not define", t.name)));
                }
                if !carried.insert(p) {
                    d.push(bug(k, format!("data type {} carries {p} twice", t.name)));
                }
            }
            if let Some(c) = &t.constant {
                let mut set: BTreeSet<&str> = BTreeSet::new();
                for (p, e) in &c.properties {
                    if !carried.contains(p.as_str()) {
                        d.push(bug(k, format!("data type {}'s constant sets {p}, which it does not carry", t.name)));
                    }
                    if !set.insert(p) {
                        d.push(bug(k, format!("data type {}'s constant sets {p} twice", t.name)));
                    }
                    let ground = |r: &Reference| -> Result<Option<&'static Property>, String> { Err(format!("which a constant cannot read ({r})")) };
                    d.extend(formula_defects(&format!("data type {}'s constant's {p}", t.name), e, &ground).into_iter().map(|x| bug(k, x)));
                }
            }
            for (n, _) in &t.size_names {
                if !is_identifier(n) || !size_names.insert(n) {
                    d.push(bug(k, format!("the size name {n:?} of {} is not a name or is listed twice", t.name)));
                }
            }
        }
        for t in &self.data_types {
            for m in t.kind.members() {
                if !types.contains_key(&m) {
                    d.push(bug(k, format!("data type {} names {m:?}, which is not a data type", t.name)));
                }
            }
        }
        // Arguments.
        let by = self.arguments_by_symbol();
        let mut ins_of: BTreeMap<i64, Vec<Ty>> = BTreeMap::new();
        let mut out_of: BTreeMap<i64, Ty> = BTreeMap::new();
        for (sym, args) in &by {
            if !symbols.contains_key(sym) {
                d.push(bug(k, format!("an argument names symbol {sym}, which is not a symbol")));
            }
            let outs: Vec<&&Argument> = args.iter().filter(|a| a.direction == Direction::Out).collect();
            match outs.as_slice() {
                [o] if o.position == 0 && !o.head => {
                    out_of.insert(*sym, o.data_type);
                }
                [o] if o.head => d.push(bug(k, format!("symbol {sym}'s output is marked head, which only an input may be"))),
                [o] => d.push(bug(k, format!("symbol {sym}'s output is at position {}, not 0", o.position))),
                _ => d.push(bug(k, format!("symbol {sym} has {} outputs, not one", outs.len()))),
            }
            let ins: Vec<&&Argument> = args.iter().filter(|a| a.direction == Direction::In).collect();
            for (i, a) in ins.iter().enumerate() {
                if a.position != i as u32 {
                    d.push(bug(k, format!("symbol {sym}'s input {i} is at position {} (positions must be dense from 0)", a.position)));
                    break;
                }
            }
            if ins.iter().filter(|a| a.head).count() > 1 {
                d.push(bug(k, format!("symbol {sym} has more than one head input")));
            }
            ins_of.insert(*sym, ins.iter().map(|a| a.data_type).collect());
            for a in args {
                if !types.contains_key(&a.data_type) {
                    d.push(bug(k, format!("symbol {sym}'s argument takes {:?}, which is not a data type", a.data_type)));
                }
            }
        }
        let mut variable_of: BTreeMap<Ty, i64> = BTreeMap::new();
        let mut formulas = false;
        for s in &self.symbols {
            if s.role == Role::Formula {
                formulas = true;
                if by.contains_key(&s.symbol) || self.flows.iter().any(|f| f.owner == Owner::Symbol(s.symbol)) || self.guards.iter().any(|g| g.owner == Owner::Symbol(s.symbol)) || s.adapter || !s.prelude.is_empty() {
                    d.push(bug(k, format!("the formula row {} has an argument, a flow, a guard, an adapter mark or a prelude; a formula row is a spelling and nothing else", s.symbol_name)));
                }
                if !s.symbol_name.starts_with('=') {
                    d.push(bug(k, format!("the formula row {} is not named =<operator>", s.symbol_name)));
                }
                for (form, t) in &s.spellings {
                    if let Ok(t) = crate::typed_environment::template::parse_template(t) {
                        if t.iter().any(|p| !matches!(p, crate::typed_environment::template::Piece::Text(_) | crate::typed_environment::template::Piece::Operand { form: None, .. })) {
                            d.push(bug(k, format!("the formula row {}'s spelling {form:?} holds a placeholder other than an operand", s.symbol_name)));
                        }
                    }
                }
                continue;
            }
            if formulas {
                d.push(bug(k, format!("symbol {} ({}) comes after a formula row; formula rows come last, so the alphabet's ids are the contract's", s.symbol, s.symbol_name)));
            }
            if s.adapter {
                match s.spelling("").map(crate::typed_environment::template::parse_template) {
                    Some(Ok(t)) if t.iter().all(|p| matches!(p, crate::typed_environment::template::Piece::Text(_) | crate::typed_environment::template::Piece::Operand { form: None, .. })) => {}
                    _ => d.push(bug(k, format!("the adapter {} has no default spelling made of its operands alone, which a repair's call renders", s.symbol_name))),
                }
            }
            if !by.contains_key(&s.symbol) {
                d.push(bug(k, format!("symbol {} ({}) has no output", s.symbol, s.symbol_name)));
            }
            let n_in = ins_of.get(&s.symbol).map_or(0, Vec::len);
            if s.role != Role::Function && n_in > 0 {
                d.push(bug(k, format!("symbol {} ({}) is a {} and takes {n_in} inputs; a constant or a variable takes none", s.symbol, s.symbol_name, s.role.name())));
            }
            if s.role == Role::Variable {
                if self.flows.iter().any(|f| f.owner == Owner::Symbol(s.symbol)) {
                    d.push(bug(k, format!("the variable terminal {} has flows; its properties are its variable's", s.symbol_name)));
                }
                if let Some(other) = out_of.get(&s.symbol).and_then(|out| variable_of.insert(*out, s.symbol)) {
                    d.push(bug(k, format!("symbols {other} and {} are both the variable terminal of one data type", s.symbol)));
                }
            }
        }
        // References: what each place may read.
        let carried = |ty: Ty, p: &str| types.get(&ty).filter(|t| t.properties.iter().any(|x| x == p)).and_then(|_| property(p));
        let attribute = |a: &str| -> Result<Option<&'static Property>, String> {
            match property(a) {
                Some(p) if p.root.is_some() => Ok(Some(p)),
                _ => Err("which is not an environment attribute (a catalogue property with a root value)".to_string()),
            }
        };
        let ups: BTreeSet<(i64, &str)> = self.flows.iter().filter_map(|f| match (&f.owner, &f.name, f.way) {
            (Owner::Symbol(s), FlowName::Property(p), Way::Up) => Some((*s, p.as_str())),
            _ => None,
        }).collect();
        let node_ref = |sym: i64, outputs: bool, each: bool| {
            let ins = ins_of.get(&sym).cloned().unwrap_or_default();
            let out = out_of.get(&sym).copied();
            let ups = &ups;
            move |r: &Reference| -> Result<Option<&'static Property>, String> {
                match r {
                    Reference::Input { position, property } => match ins.get(*position as usize) {
                        None => Err(format!("but the symbol has {} inputs", ins.len())),
                        Some(ty) => carried(*ty, property).map(Some).ok_or_else(|| format!("which {ty:?} does not carry")),
                    },
                    Reference::Output(p) if !outputs => Err(format!("which cannot be read here (out.{p} is decoded after the subtrees)")),
                    Reference::Output(p) => match out.and_then(|ty| carried(ty, p)) {
                        None => Err(format!("which {} does not carry", out.map_or_else(|| "a symbol with no output".to_string(), |t| format!("{t:?}")))),
                        Some(_) if !ups.contains(&(sym, *p)) => Err("which no up flow of the symbol sets, so it is ? at every node".to_string()),
                        Some(x) => Ok(Some(x)),
                    },
                    Reference::Env(a) => attribute(a),
                    Reference::Index("each") if each => Ok(None),
                    Reference::Index(i) | Reference::Size(i) => Err(format!("but {i} is no index here")),
                }
            }
        };
        // Flows.
        let mut set: BTreeSet<(Owner, Place, &FlowName)> = BTreeSet::new();
        for f in &self.flows {
            let Owner::Symbol(s) = f.owner else {
                d.push(bug(k, format!("the flow {} is the entry's; an entry's flows are on its Entry", f.name.text())));
                continue;
            };
            let at = format!("symbol {s}'s {} flow {}", if f.way == Way::Up { "up" } else { "down" }, f.name.text());
            if !set.insert((f.owner, f.at, &f.name)) {
                d.push(bug(k, format!("{at} is given twice")));
            }
            let n_in = ins_of.get(&s).map_or(0, Vec::len);
            match (f.way, f.at, &f.name) {
                (Way::Up, Place::Out, FlowName::Property(p)) => {
                    if let Some(ty) = out_of.get(&s) {
                        if carried(*ty, p).is_none() {
                            d.push(bug(k, format!("{at} sets out.{p}, which {ty:?} does not carry")));
                        }
                    }
                }
                (Way::Down, Place::In(pos), name) if (pos as usize) < n_in => match name {
                    FlowName::Env(a) => {
                        if let Err(why) = attribute(a) {
                            d.push(bug(k, format!("{at} sets env.{a}, {why}")));
                        }
                    }
                    FlowName::Var { ty, .. } => {
                        if !types.contains_key(ty) {
                            d.push(bug(k, format!("{at} introduces variables of {ty:?}, which is not a data type")));
                        }
                    }
                    FlowName::VarProperty { ty, group, property } => {
                        if carried(*ty, property).is_none() {
                            d.push(bug(k, format!("{at} sets {property}, which {ty:?} does not carry")));
                        }
                        if !self.flows.iter().any(|g| g.owner == f.owner && g.at == f.at && g.name == (FlowName::Var { ty: *ty, group: *group })) {
                            d.push(bug(k, format!("{at} sets a property of variables no var flow introduces")));
                        }
                    }
                    FlowName::VarSpell { .. } => {
                        if let Err(e) = crate::typed_environment::template::parse_template(&f.template) {
                            d.push(bug(k, format!("{at}: {e}")));
                        }
                    }
                    FlowName::Property(_) => d.push(bug(k, format!("{at} is down and names a property, not env.a or var.T"))),
                },
                _ => d.push(bug(k, format!("{at} is at {:?}, which its direction and name do not allow (up at out; down at an argument the symbol has)", f.at))),
            }
            if !matches!(f.name, FlowName::VarSpell { .. }) {
                let each = matches!(f.name, FlowName::VarProperty { .. });
                d.extend(formula_defects(&at, &f.formula, &node_ref(s, f.way == Way::Up, each)).into_iter().map(|x| bug(k, x)));
            }
        }
        // Guards.
        let mut orders: BTreeSet<(Owner, u32)> = BTreeSet::new();
        for g in &self.guards {
            let Owner::Symbol(s) = g.owner else {
                d.push(bug(k, format!("the guard {} is the entry's; an entry's guards are on its Entry", g.condition)));
                continue;
            };
            let at = format!("symbol {s}'s guard {} ({})", g.order, g.condition);
            if !orders.insert((g.owner, g.order)) {
                d.push(bug(k, format!("{at}: the order is used twice")));
            }
            if !out_of.contains_key(&s) {
                d.push(bug(k, format!("{at} names a symbol with no output")));
            }
            d.extend(condition_defects(&at, &g.condition, &node_ref(s, true, false)).into_iter().map(|x| bug(k, x)));
            let (ins, out) = (ins_of.get(&s).cloned().unwrap_or_default(), out_of.get(&s).copied());
            d.extend(self.repair_defects(&at, g, &node_ref(s, true, false), ins.len(), &|sub| match sub {
                crate::typed_environment::template::Substitute::Input(i) => ins.get(i as usize).is_some_and(|t| Some(*t) == out),
                crate::typed_environment::template::Substitute::Own => ins.iter().any(|t| Some(*t) == out),
                crate::typed_environment::template::Substitute::Const => out.and_then(|t| types.get(&t)).is_some_and(|t| t.constant.is_some()),
            }));
        }
        for s in &self.symbols {
            if let Err(e) = self.decode_order(s.symbol) {
                d.push(e);
            }
            if let Err(e) = self.up_order(s.symbol) {
                d.push(e);
            }
        }
        if d.is_empty() {
            Ok(())
        } else {
            Err(d)
        }
    }

    /// The defects of a guard's strength and repairs: a `must` guard has a repair, an `advise` guard none; each repair's
    /// site exists, its template parses, every symbol it calls is an adapter Symbol (always, code review finding 6), its
    /// guard reads only the site's environment, a substitute replaces at the node; and a `must` guard's LAST repair
    /// always applies (guard and `when` both `true`) and is a wrap or a substitute the node always has, so no gene can
    /// meet a guard none of whose repairs applies (finding 5).
    fn repair_defects(&self, at: &str, g: &Guard, resolve: &Resolve, n_in: usize, has: &dyn Fn(crate::typed_environment::template::Substitute) -> bool) -> Vec<String> {
        let k = self.kingdom.as_str();
        let mut d = Vec::new();
        match (g.strength, g.repairs.is_empty()) {
            (Strength::Must, true) => d.push(bug(k, format!("{at} is must and has no repair"))),
            (Strength::Advise, false) => d.push(bug(k, format!("{at} is advise and has a repair, which nothing runs (the cost never repairs)"))),
            _ => {}
        }
        let env_only = |r: &Reference| -> Result<Option<&'static Property>, String> {
            match r {
                Reference::Env(a) => match property(a) {
                    Some(p) if p.root.is_some() => Ok(Some(p)),
                    _ => Err("which is not an environment attribute".to_string()),
                },
                _ => Err("and a repair's guard reads only the site's environment (env.a)".to_string()),
            }
        };
        for (i, r) in g.repairs.iter().enumerate() {
            let rat = format!("{at}'s repair {i}");
            let root = matches!(g.owner, Owner::Entry);
            match r.site {
                Site::Root if !root => d.push(bug(k, format!("{rat} runs at the root, which only the entry's guards do"))),
                Site::Node | Site::In(_) if root => d.push(bug(k, format!("{rat} runs at {}, and the entry's guards repair at the root", r.site))),
                Site::In(j) if j as usize >= n_in => d.push(bug(k, format!("{rat} runs at input {j}, which the symbol has not"))),
                _ => {}
            }
            let last = i + 1 == g.repairs.len();
            match crate::typed_environment::template::parse_template(&r.template) {
                Err(e) => d.push(bug(k, format!("{rat}: {e}"))),
                Ok(t) => {
                    if let Some(sub) = crate::typed_environment::template::substitute_of(&t) {
                        if r.site != Site::Node {
                            d.push(bug(k, format!("{rat} is a substitute at {}, and a substitute replaces at the node", r.site)));
                        }
                        if last && g.strength == Strength::Must && !has(sub) {
                            d.push(bug(k, format!("{rat} is the last and a substitute the node does not always have (no input of its own data type, no constant)")));
                        }
                    }
                    for c in crate::typed_environment::template::calls_of(&t) {
                        if !self.symbols.iter().any(|s| s.symbol_name == c && s.adapter) {
                            d.push(bug(k, format!("{rat} calls {c}, which is not an adapter symbol of the kingdom")));
                        }
                    }
                }
            }
            if last && g.strength == Strength::Must && !r.unconditional() {
                d.push(bug(k, format!("{rat} is the last repair of a must guard and has the guard {} and the when {}: the last must always apply", r.guard, r.when)));
            }
            d.extend(condition_defects(&format!("{rat}'s guard"), &r.guard, &env_only).into_iter().map(|x| bug(k, x)));
            d.extend(condition_defects(&format!("{rat}'s when"), &r.when, resolve).into_iter().map(|x| bug(k, x)));
        }
        d
    }

    /// THE ENTRY CHECK, against these tables. Every defect, each "our software has a bug: ...": the result's data type
    /// is the tables'; each parameter's name is an identifier given once, its data type the tables', its properties ones
    /// its data type carries, once each, by ground formulas whose size names are the tables'; an alternative form
    /// names an earlier raw parameter; each guard reads only `out.p` and `env.a` and repairs at the root; each flow sets
    /// an attribute's root value, once.
    pub fn entry_defects(&self, entry: &Entry) -> Vec<String> {
        let k = self.kingdom.as_str();
        let mut d: Vec<String> = Vec::new();
        let bounds = self.bounds();
        let is_type = |ty: Ty| self.data_type(ty).is_some();
        if !is_type(entry.output) {
            d.push(bug(k, format!("the entry returns {:?}, which is not a data type", entry.output)));
        }
        let ground = |r: &Reference| -> Result<Option<&'static Property>, String> { Err(format!("which an entry value cannot read ({r})")) };
        let size_check = |at: &str, e: &PropertyExpr, d: &mut Vec<String>| {
            for n in e.size_names() {
                if !bounds.contains_key(&n) {
                    d.push(bug(k, format!("{at} reads the size name {n}, which no data type's size_names declares")));
                }
            }
        };
        let mut names: BTreeMap<&str, &EntryParameter> = BTreeMap::new();
        for p in &entry.parameters {
            let name = p.name.as_str();
            if !is_identifier(name) {
                d.push(bug(k, format!("the entry's parameter name {name:?} is not an identifier")));
            }
            if !is_type(p.data_type) {
                d.push(bug(k, format!("the entry's parameter {name} is of {:?}, which is not a data type", p.data_type)));
            }
            if let Some((raw, t)) = &p.form_of {
                match names.get(raw.as_str()) {
                    None => d.push(bug(k, format!("the entry's parameter {name} is a form of {raw}, which is no earlier parameter"))),
                    Some(r) if r.form_of.is_some() => d.push(bug(k, format!("the entry's parameter {name} is a form of {raw}, which is itself a form"))),
                    Some(r) => {
                        let raw_ty = r.data_type;
                        let from_raw = |x: &Reference| -> Result<Option<&'static Property>, String> {
                            match x {
                                Reference::Input { position: 0, property } => self.carried(raw_ty, property).map(Some).ok_or_else(|| format!("which {raw_ty:?} does not carry")),
                                _ => Err("and a transform reads only its raw value (in0.p)".to_string()),
                            }
                        };
                        for (q, e) in &t.properties {
                            if self.carried(p.data_type, q).is_none() {
                                d.push(bug(k, format!("the entry's form {name}'s transform sets {q}, which {:?} does not carry", p.data_type)));
                            }
                            d.extend(formula_defects(&format!("the entry's form {name}'s {q}"), e, &from_raw).into_iter().map(|x| bug(k, x)));
                        }
                        d.extend(formula_defects(&format!("the entry's form {name}'s cost"), &t.cost, &from_raw).into_iter().map(|x| bug(k, x)));
                        if let Err(e) = crate::typed_environment::template::parse_template(&t.template) {
                            d.push(bug(k, format!("the entry's form {name}'s transform: {e}")));
                        }
                    }
                }
            }
            if names.insert(name, p).is_some() {
                d.push(bug(k, format!("the entry has two parameters named {name}")));
            }
            let mut set: BTreeSet<&str> = BTreeSet::new();
            for (q, e, _) in &p.properties {
                if self.carried(p.data_type, q).is_none() {
                    d.push(bug(k, format!("the entry's parameter {name} sets {q}, which {:?} does not carry", p.data_type)));
                }
                if !set.insert(q) {
                    d.push(bug(k, format!("the entry's parameter {name} sets {q} twice")));
                }
                let at = format!("the entry's parameter {name}'s {q}");
                d.extend(formula_defects(&at, e, &ground).into_iter().map(|x| bug(k, x)));
                size_check(&at, e, &mut d);
            }
        }
        let out = entry.output;
        let root_ref = |r: &Reference| -> Result<Option<&'static Property>, String> {
            match r {
                Reference::Output(p) => self.carried(out, p).map(Some).ok_or_else(|| format!("which {out:?} does not carry")),
                Reference::Env(a) => match property(a) {
                    Some(p) if p.root.is_some() => Ok(Some(p)),
                    _ => Err("which is not an environment attribute".to_string()),
                },
                _ => Err("and the entry's guard reads only out.p and env.a".to_string()),
            }
        };
        let mut orders: BTreeSet<u32> = BTreeSet::new();
        for g in &entry.guards {
            let at = format!("the entry's guard {} ({})", g.order, g.condition);
            if g.owner != Owner::Entry {
                d.push(bug(k, format!("{at} is a symbol's, on the entry")));
            }
            if !orders.insert(g.order) {
                d.push(bug(k, format!("{at}: the order is used twice")));
            }
            d.extend(condition_defects(&at, &g.condition, &root_ref).into_iter().map(|x| bug(k, x)));
            for n in g.condition.size_names() {
                if !bounds.contains_key(&n) {
                    d.push(bug(k, format!("{at} reads the size name {n}, which no data type's size_names declares")));
                }
            }
            d.extend(self.repair_defects(&at, g, &root_ref, 0, &|_| false));
        }
        let mut env: BTreeSet<String> = BTreeSet::new();
        for f in &entry.flows {
            match (&f.owner, f.at, &f.name, f.way) {
                (Owner::Entry, Place::Out, FlowName::Env(a), Way::Down) => {
                    if !matches!(property(a), Some(p) if p.root.is_some()) {
                        d.push(bug(k, format!("the entry sets env.{a}, which is not an environment attribute")));
                    }
                    if !env.insert(a.clone()) {
                        d.push(bug(k, format!("the entry sets env.{a} twice")));
                    }
                    d.extend(formula_defects(&format!("the entry's env.{a}"), &f.formula, &ground).into_iter().map(|x| bug(k, x)));
                }
                _ => d.push(bug(k, format!("the entry's flow {} is not a down flow of entry at out setting env.a", f.name.text()))),
            }
        }
        d
    }

    /// The tables as text: one row a line, its kind first and its fields tab-separated; types by their [`Ty::code`].
    /// [`KingdomTables::parse`] reads it back to the same tables. Tables that fail [`KingdomTables::check`] are an
    /// error, not text.
    pub fn to_text(&self) -> Result<String, String> {
        if let Err(d) = self.check() {
            return Err(format!("our software has a bug: kingdom {}: the tables fail their check ({} defects), so their text would not read back the same: {}", self.kingdom, d.len(), d.join("; ")));
        }
        let mut lines = vec![format!("kingdom\t{}", field("kingdom name", &self.kingdom)?)];
        for s in &self.symbols {
            lines.push(format!(
                "symbol\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                s.symbol,
                field("symbol name", &s.symbol_name)?,
                field("alias", &s.alias)?,
                field("semantic id", &s.semantic_id)?,
                s.role.name(),
                if s.adapter { "adapter" } else { "-" },
                field("prelude", &escape(&s.prelude))?
            ));
            for (form, t) in &s.spellings {
                lines.push(format!("spelling\t{}\t{}\t{}", s.symbol, field("form", form)?, field("spelling", t)?));
            }
        }
        for t in &self.data_types {
            let kind = match &t.kind {
                DataKind::Scalar => "scalar".to_string(),
                DataKind::Array { element } => format!("array\t{}", element.code()),
                DataKind::Tuple { components } => format!("tuple\t{}", codes(components)),
                DataKind::Function { parameters, result } => format!("function\t{}\t{}", codes(parameters), result.code()),
            };
            lines.push(format!("data_type\t{}\t{}\t{kind}", t.id.code(), field("data type name", &t.name)?));
            for p in &t.properties {
                lines.push(format!("carries\t{}\t{}", t.id.code(), field("property name", p)?));
            }
            if let Some(c) = &t.constant {
                lines.push(format!("constant\t{}\t{}\t{}", t.id.code(), field("constant", &c.spelling)?, property_list(&c.properties)));
            }
            for (n, lb) in &t.size_names {
                lines.push(format!("size_name\t{}\t{}\t{lb}", t.id.code(), field("size name", n)?));
            }
        }
        for a in &self.arguments {
            let dir = match a.direction {
                Direction::In => "in",
                Direction::Out => "out",
            };
            lines.push(format!("argument\t{}\t{}\t{dir}\t{}\t{}", a.symbol, a.position, a.data_type.code(), if a.head { "head" } else { "-" }));
        }
        for f in &self.flows {
            let at = match f.at {
                Place::In(k) => format!("in{k}"),
                Place::Out => "out".to_string(),
            };
            let value = if matches!(f.name, FlowName::VarSpell { .. }) { field("template", &f.template)? } else { f.formula.to_string() };
            lines.push(format!("flow\t{}\t{at}\t{}\t{}\t{value}", owner_text(f.owner), if f.way == Way::Up { "up" } else { "down" }, f.name.text()));
        }
        for g in &self.guards {
            lines.push(format!("guard\t{}\t{}\t{}\t{}", owner_text(g.owner), g.order, g.strength.name(), g.condition));
            for r in &g.repairs {
                lines.push(format!("repair\t{}\t{}\t{}\t{}\t{}\t{}", owner_text(g.owner), g.order, r.site, r.when, r.guard, field("template", &r.template)?));
            }
        }
        let mut text = lines.join("\n");
        text.push('\n');
        Ok(text)
    }

    /// Read the text [`KingdomTables::to_text`] writes.
    pub fn parse(text: &str) -> Result<KingdomTables, String> {
        let mut t = KingdomTables::default();
        let mut named = false;
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let bad = || format!("our software has a bug: line {} of the kingdom tables is not a row: {line:?}", n + 1);
            let dt = |t: &mut KingdomTables, code: &str| -> Result<usize, String> {
                let id = ty_of(code)?;
                t.data_types.iter().rposition(|d| d.id == id).ok_or_else(bad)
            };
            match (f[0], f.len()) {
                ("kingdom", 2) if named => return Err(format!("our software has a bug: line {} of the kingdom tables names a second kingdom: {line:?}", n + 1)),
                ("kingdom", 2) => {
                    t.kingdom = f[1].to_string();
                    named = true;
                }
                ("symbol", 8) => t.symbols.push(SymbolRow {
                    symbol: number("symbol id", f[1])?,
                    symbol_name: f[2].to_string(),
                    alias: f[3].to_string(),
                    semantic_id: f[4].to_string(),
                    role: Role::from_name(f[5]).ok_or_else(bad)?,
                    adapter: match f[6] {
                        "adapter" => true,
                        "-" => false,
                        _ => return Err(bad()),
                    },
                    spellings: Vec::new(),
                    prelude: unescape(f[7]),
                }),
                ("spelling", 4) => {
                    let s: i64 = number("symbol id", f[1])?;
                    t.symbols.iter_mut().rev().find(|x| x.symbol == s).ok_or_else(bad)?.spellings.push((f[2].to_string(), f[3].to_string()));
                }
                ("data_type", 4..=6) => {
                    let kind = match (f[3], f.len()) {
                        ("scalar", 4) => DataKind::Scalar,
                        ("array", 5) => DataKind::Array { element: ty_of(f[4])? },
                        ("tuple", 5) => DataKind::Tuple { components: tys_of(f[4])? },
                        ("function", 6) => DataKind::Function { parameters: tys_of(f[4])?, result: ty_of(f[5])? },
                        _ => return Err(bad()),
                    };
                    t.data_types.push(DataType { kind, ..DataType::scalar(ty_of(f[1])?, f[2]) });
                }
                ("carries", 3) => {
                    let i = dt(&mut t, f[1])?;
                    t.data_types[i].properties.push(f[2].to_string());
                }
                ("constant", 4) => {
                    let i = dt(&mut t, f[1])?;
                    t.data_types[i].constant = Some(Constant { spelling: f[2].to_string(), properties: parse_property_list(f[3])? });
                }
                ("size_name", 4) => {
                    let i = dt(&mut t, f[1])?;
                    t.data_types[i].size_names.push((f[2].to_string(), number("lower bound", f[3])?));
                }
                ("argument", 6) => t.arguments.push(Argument {
                    symbol: number("symbol id", f[1])?,
                    position: number("position", f[2])?,
                    direction: match f[3] {
                        "in" => Direction::In,
                        "out" => Direction::Out,
                        _ => return Err(bad()),
                    },
                    data_type: ty_of(f[4])?,
                    head: match f[5] {
                        "head" => true,
                        "-" => false,
                        _ => return Err(bad()),
                    },
                }),
                ("flow", 6) => {
                    let at = match f[2] {
                        "out" => Place::Out,
                        x => Place::In(x.strip_prefix("in").and_then(|d| d.parse().ok()).ok_or_else(bad)?),
                    };
                    let way = match f[3] {
                        "up" => Way::Up,
                        "down" => Way::Down,
                        _ => return Err(bad()),
                    };
                    let name = FlowName::parse(f[4]).ok_or_else(bad)?;
                    let (formula, template) = if matches!(name, FlowName::VarSpell { .. }) { (PropertyExpr::Unknown, f[5].to_string()) } else { (PropertyExpr::parse(f[5])?, String::new()) };
                    t.flows.push(Flow { owner: owner_of(f[1])?, at, name, formula, template, way });
                }
                ("guard", 5) => t.guards.push(Guard { owner: owner_of(f[1])?, order: number("order", f[2])?, strength: Strength::from_name(f[3]).ok_or_else(bad)?, condition: Condition::parse(f[4])?, repairs: Vec::new() }),
                ("repair", 7) => {
                    let (owner, order) = (owner_of(f[1])?, number::<u32>("order", f[2])?);
                    let g = t.guards.iter_mut().rev().find(|g| g.owner == owner && g.order == order).ok_or_else(bad)?;
                    g.repairs.push(Repair { site: Site::parse(f[3]).ok_or_else(bad)?, when: Condition::parse(f[4])?, guard: Condition::parse(f[5])?, template: f[6].to_string() });
                }
                _ => return Err(bad()),
            }
        }
        Ok(t)
    }
}

/// A text field on one line: a backslash as `\\`, a line break as `\n` ([`unescape`] reads it back).
pub fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('\n', "\\n")
}

/// A text field read back: `\\` is a backslash and `\n` a line break (a prelude is several lines of the target
/// language, on one line of a table).
pub fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

impl KingdomTables {
    /// THE NAMED TEXT FORM of a kingdom's four kinds of row and its entry, which a view generator writes (one row a
    /// line, its kind first, its fields tab-separated, every data type and symbol by its NAME, `-` for none):
    ///
    /// | line | fields |
    /// |---|---|
    /// | `data_type` | name, `scalar` / `array` element / `tuple` components (`;`) / `function` parameters (`;`) result |
    /// | `carries` | data type, property |
    /// | `constant` | data type, spelling, properties (`p=e;q=f`) |
    /// | `size_name` | data type, name, lower bound |
    /// | `symbol` | name, alias, role, `adapter` or `-`, prelude (`\n` a line break, `\\` a backslash) |
    /// | `spelling` | symbol, form (empty: the default), template |
    /// | `argument` | symbol, position, `in` / `out`, data type, `head` or `-` |
    /// | `flow` | symbol, `in<k>` / `out`, `up` / `down`, name (`p`, `env.a`, `var.<T>[/g][.p or .spell]`), formula or template |
    /// | `guard` | symbol or `entry`, order, `must` / `advise`, condition |
    /// | `repair` | symbol or `entry`, order, site, when, guard, template |
    /// | `entry` | problem, result's data type |
    /// | `parameter` | name, data type, properties (`p=e`, `p=e @ <case set>` when measured), raw, transform, its properties, cost |
    ///
    /// Data type `i` (in order) is `ty(i)`; symbol `i` is the i-th `symbol` line. A data type's name holds no `.` or
    /// `/`, so `var.<T>.p` reads back.
    pub fn parse_named(kingdom: &str, text: &str, ty: &dyn Fn(usize) -> Result<Ty, String>) -> Result<(KingdomTables, Entry), String> {
        let mut t = KingdomTables { kingdom: kingdom.to_string(), ..KingdomTables::default() };
        let mut entry: Option<Entry> = None;
        let mut types: BTreeMap<String, Ty> = BTreeMap::new();
        let mut symbols: BTreeMap<String, i64> = BTreeMap::new();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let bad = |why: &str| format!("our software has a bug: kingdom {kingdom}: line {} of its table {why}: {line:?}", n + 1);
            let dt = |name: &str| types.get(name).copied().ok_or_else(|| bad(&format!("names the data type {name:?}, which no earlier line declares")));
            let dts = |text: &str| -> Result<Vec<Ty>, String> { text.split(';').filter(|x| !x.is_empty()).map(dt).collect() };
            let sym = |name: &str| symbols.get(name).copied().ok_or_else(|| bad(&format!("names the symbol {name:?}, which no earlier line declares")));
            let owner = |name: &str| if name == "entry" { Ok(Owner::Entry) } else { sym(name).map(Owner::Symbol) };
            fn none(x: &str) -> &str {
                if x == "-" {
                    ""
                } else {
                    x
                }
            }
            match (f[0], f.len()) {
                ("data_type", 3..=5) => {
                    let id = ty(types.len())?;
                    let kind = match (f[2], f.len()) {
                        ("scalar", 3) => DataKind::Scalar,
                        ("array", 4) => DataKind::Array { element: dt(f[3])? },
                        ("tuple", 4) => DataKind::Tuple { components: dts(f[3])? },
                        ("function", 5) => DataKind::Function { parameters: dts(f[3])?, result: dt(f[4])? },
                        _ => return Err(bad("is not a data type of one of the four kinds")),
                    };
                    if f[1].contains(['.', '/']) || types.insert(f[1].to_string(), id).is_some() {
                        return Err(bad("declares a data type twice, or one whose name holds . or /"));
                    }
                    t.data_types.push(DataType { kind, ..DataType::scalar(id, f[1]) });
                }
                ("carries", 3) => {
                    let id = dt(f[1])?;
                    t.data_types.iter_mut().find(|d| d.id == id).ok_or_else(|| bad("names no data type"))?.properties.push(f[2].to_string());
                }
                ("constant", 4) => {
                    let id = dt(f[1])?;
                    t.data_types.iter_mut().find(|d| d.id == id).ok_or_else(|| bad("names no data type"))?.constant = Some(Constant { spelling: f[2].to_string(), properties: parse_property_list(f[3])? });
                }
                ("size_name", 4) => {
                    let id = dt(f[1])?;
                    let lb = number("lower bound", f[3])?;
                    t.data_types.iter_mut().find(|d| d.id == id).ok_or_else(|| bad("names no data type"))?.size_names.push((f[2].to_string(), lb));
                }
                ("symbol", 6) => {
                    let id = symbols.len() as i64;
                    if symbols.insert(f[1].to_string(), id).is_some() {
                        return Err(bad("declares a symbol twice"));
                    }
                    let adapter = match f[4] {
                        "adapter" => true,
                        "-" => false,
                        _ => return Err(bad("marks an adapter neither adapter nor -")),
                    };
                    let role = Role::from_name(f[3]).ok_or_else(|| bad("names no role"))?;
                    t.symbols.push(SymbolRow { symbol: id, symbol_name: f[1].to_string(), alias: f[2].to_string(), semantic_id: f[1].to_string(), role, adapter, spellings: Vec::new(), prelude: unescape(none(f[5])) });
                }
                ("spelling", 4) => {
                    let id = sym(f[1])?;
                    t.symbols.iter_mut().find(|s| s.symbol == id).ok_or_else(|| bad("names no symbol"))?.spellings.push((f[2].to_string(), f[3].to_string()));
                }
                ("argument", 6) => t.arguments.push(Argument {
                    symbol: sym(f[1])?,
                    position: number("position", f[2])?,
                    direction: match f[3] {
                        "in" => Direction::In,
                        "out" => Direction::Out,
                        _ => return Err(bad("has a direction neither in nor out")),
                    },
                    data_type: dt(f[4])?,
                    head: match f[5] {
                        "head" => true,
                        "-" => false,
                        _ => return Err(bad("marks a head neither head nor -")),
                    },
                }),
                ("flow", 6) => {
                    let at = match f[2] {
                        "out" => Place::Out,
                        x => Place::In(x.strip_prefix("in").and_then(|d| d.parse().ok()).ok_or_else(|| bad("acts at neither out nor in<k>"))?),
                    };
                    let way = match f[3] {
                        "up" => Way::Up,
                        "down" => Way::Down,
                        _ => return Err(bad("flows neither up nor down")),
                    };
                    let name = match f[4].strip_prefix("var.") {
                        Some(rest) => {
                            let end = rest.find(['.', '/']).unwrap_or(rest.len());
                            let coded = format!("var.{}{}", dt(&rest[..end])?.code(), &rest[end..]);
                            FlowName::parse(&coded).ok_or_else(|| bad("names no flow"))?
                        }
                        None => FlowName::parse(f[4]).ok_or_else(|| bad("names no flow"))?,
                    };
                    let (formula, template) = if matches!(name, FlowName::VarSpell { .. }) { (PropertyExpr::Unknown, f[5].to_string()) } else { (PropertyExpr::parse(f[5])?, String::new()) };
                    let flow = Flow { owner: owner(f[1])?, at, name, formula, template, way };
                    match flow.owner {
                        Owner::Entry => entry.as_mut().ok_or_else(|| bad("is the entry's before the entry line"))?.flows.push(flow),
                        Owner::Symbol(_) => t.flows.push(flow),
                    }
                }
                ("guard", 5) => {
                    let g = Guard { owner: owner(f[1])?, order: number("order", f[2])?, strength: Strength::from_name(f[3]).ok_or_else(|| bad("has a strength neither must nor advise"))?, condition: Condition::parse(f[4])?, repairs: Vec::new() };
                    match g.owner {
                        Owner::Entry => entry.as_mut().ok_or_else(|| bad("is the entry's before the entry line"))?.guards.push(g),
                        Owner::Symbol(_) => t.guards.push(g),
                    }
                }
                ("repair", 7) => {
                    let (o, order) = (owner(f[1])?, number::<u32>("order", f[2])?);
                    let r = Repair { site: Site::parse(f[3]).ok_or_else(|| bad("names no site"))?, when: Condition::parse(f[4])?, guard: Condition::parse(f[5])?, template: f[6].to_string() };
                    let guards = match o {
                        Owner::Entry => &mut entry.as_mut().ok_or_else(|| bad("is the entry's before the entry line"))?.guards,
                        Owner::Symbol(_) => &mut t.guards,
                    };
                    guards.iter_mut().rev().find(|g| g.owner == o && g.order == order).ok_or_else(|| bad("repairs no guard declared before it"))?.repairs.push(r);
                }
                ("entry", 3) => {
                    if entry.is_some() {
                        return Err(bad("declares a second entry"));
                    }
                    entry = Some(Entry::new(f[1], dt(f[2])?));
                }
                ("parameter", 8) => {
                    let mut properties = Vec::new();
                    for item in f[3].split(';').filter(|x| *x != "-" && !x.is_empty()) {
                        let (body, source) = match item.split_once(" @ ") {
                            Some((b, cases)) => (b, Source::Measured(cases.to_string())),
                            None => (item, Source::Declared),
                        };
                        let (q, v) = body.split_once('=').ok_or_else(|| bad("has a property that is not p=e"))?;
                        properties.push((q.to_string(), PropertyExpr::parse(v)?, source));
                    }
                    let form_of = if f[4] == "-" { None } else { Some((f[4].to_string(), Transform { template: f[5].to_string(), properties: parse_property_list(f[6])?, cost: PropertyExpr::parse(f[7])? })) };
                    let p = EntryParameter { name: f[1].to_string(), data_type: dt(f[2])?, properties, form_of };
                    entry.as_mut().ok_or_else(|| bad("is a parameter before the entry line"))?.parameters.push(p);
                }
                _ => return Err(bad("is not a row of the named text form")),
            }
        }
        let entry = entry.ok_or_else(|| format!("our software has a bug: kingdom {kingdom}: its table has no entry line"))?;
        Ok((t, entry))
    }
}

/// The flow's value for a `var.T.spell` flow: its template.
pub fn spell(symbol: i64, k: u32, ty: Ty, group: u32, template: &str) -> Flow {
    Flow { template: template.to_string(), ..Flow::down(symbol, k, FlowName::VarSpell { ty, group }, PropertyExpr::Unknown) }
}

#[cfg(test)]
mod tests;
