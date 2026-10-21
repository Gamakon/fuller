//! The true multi-typed WGSL kingdom: the generated symbol table of
//! `kingdoms/wgsl_true` (`gen_tables.py`, from `nucleotable/docs/
//! MultiTypedGEP_s9_WGSL_draft.md`) loaded through the kingdom contract of
//! `MultiTypedGEP.md` section 1, a typed initialiser over its layout, and the
//! renderer: a pure substitution of argument texts into each row's spelling.
//!
//! The table is data read at run time (`TrueKingdom::load`), not `include_str!`,
//! so a regenerated table needs no rebuild. A row is `(name, kind, output type,
//! ordered argument types, spelling)`; a constant row is the placeholder `?_T`
//! of its output type and `constants.tsv` is that type's append-only `C_T`.
//!
//! Spelling templates: `{N}` the text of argument N, `{N:F}` field F of that
//! text split on whitespace (a literal such as the grid shape `16 16`), `{v}`
//! the value text a constant reads from its `C_T`. Every other brace is WGSL.
//! The renderer does nothing else: no mask, no clamp, no fallback, no rewrite.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::geneframe::Ty;
use crate::kingdom_contract::{relations_from_rows, KingdomContract, KingdomRelations};
use crate::typed_gep::{GeneLayout, Tree};
use crate::wgsl::loader::{parse_duals, WgslDual};
use crate::wgsl::work_guard::Fuel;

/// The kingdom's name in its relations.
pub const TRUE_WGSL: &str = "TRUE_WGSL";

/// The directory of the generated table.
pub fn default_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kingdoms/wgsl_true")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Function,
    Terminal,
    /// The placeholder `?_T` of its output type.
    Constant,
}

#[derive(Debug, Clone)]
pub struct TrueRow {
    pub name: String,
    pub kind: RowKind,
    pub out: Ty,
    pub args: Vec<Ty>,
    pub spelling: String,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct TrueKingdom {
    pub dir: PathBuf,
    pub types: Vec<WgslDual>,
    pub rows: Vec<TrueRow>,
    /// `C_T`: the value texts of each constant-capable type.
    pub constants: BTreeMap<Ty, Vec<String>>,
    pub prologue: String,
    pub declarations: String,
    pub root: Ty,
    /// The interface's fuel budget `K`.
    pub fuel_budget: u32,
}

/// `H`: the sum of the head lengths `h_T` over the types `root` reaches, at program size `n`.
pub fn head_total_of(contract: &KingdomContract, root: Ty, n: usize) -> Result<u64, String> {
    let heads = contract.heads_from_program_size(root, n)?;
    let b = contract.bounds(root, &heads)?;
    Ok(b.types.iter().map(|t| t.h as u64).sum())
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn read(dir: &Path, file: &str) -> Result<String, String> {
    std::fs::read_to_string(dir.join(file)).map_err(|e| format!("our software has a bug: {}: {e}", dir.join(file).display()))
}

impl TrueKingdom {
    pub fn load(dir: &Path) -> Result<TrueKingdom, String> {
        let types = parse_duals(&read(dir, "types.tsv")?)?;
        let by_name: BTreeMap<&str, Ty> = types.iter().map(|d| (d.dual.as_str(), d.ty())).collect();
        let ty_of = |n: &str, what: &str| by_name.get(n).copied().ok_or_else(|| format!("our software has a bug: {what}: type {n} is not in types.tsv"));
        let text = read(dir, "functions.tsv")?;
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().ok_or("empty functions.tsv")?.split('\t').collect();
        let want = ["ref", "name", "kind", "out", "args", "spelling", "note"];
        if header != want {
            return Err(format!("functions.tsv header is {header:?}, want {want:?}"));
        }
        let mut rows = Vec::new();
        for (n, line) in lines.enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 7 {
                return Err(format!("functions.tsv line {}: {} fields, want 7", n + 2, f.len()));
            }
            let kind = match f[2] {
                "function" => RowKind::Function,
                "terminal" => RowKind::Terminal,
                "constant" => RowKind::Constant,
                o => return Err(format!("functions.tsv line {}: kind {o:?}", n + 2)),
            };
            let args = if f[4].is_empty() { Vec::new() } else { f[4].split(';').map(|a| ty_of(a, f[1])).collect::<Result<Vec<_>, _>>()? };
            rows.push(TrueRow { name: f[1].to_string(), kind, out: ty_of(f[3], f[1])?, args, spelling: unescape(f[5]), note: unescape(f[6]) });
        }
        let text = read(dir, "constants.tsv")?;
        let mut constants: BTreeMap<Ty, Vec<String>> = BTreeMap::new();
        for (n, line) in text.lines().skip(1).enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 3 {
                return Err(format!("constants.tsv line {}: {} fields, want 3", n + 2, f.len()));
            }
            let idx: usize = f[1].parse().map_err(|e| format!("constants.tsv line {}: {e}", n + 2))?;
            let v = constants.entry(ty_of(f[0], "constants.tsv")?).or_default();
            if idx != v.len() {
                return Err(format!("our software has a bug: constants.tsv line {}: index {idx} is not the next of {}", n + 2, f[0]));
            }
            v.push(unescape(f[2]));
        }
        let meta = read(dir, "interface.tsv")?;
        let get = |key: &str| meta.lines().filter_map(|l| l.split_once('\t')).find(|(k, _)| *k == key).map(|(_, v)| v.to_string()).ok_or_else(|| format!("interface.tsv has no {key}"));
        Ok(TrueKingdom {
            dir: dir.to_path_buf(),
            root: ty_of(&get("root")?, "interface.tsv root")?,
            fuel_budget: get("fuel_budget")?.parse().map_err(|e| format!("interface.tsv fuel_budget: {e}"))?,
            types,
            rows,
            constants,
            prologue: read(dir, "prologue.wgsl")?,
            declarations: read(dir, "declarations.wgsl")?,
        })
    }

    pub fn type_name(&self, t: Ty) -> String {
        match t {
            Ty::Wgsl(i) => self.types[i as usize].dual.clone(),
            o => format!("{o:?}"),
        }
    }

    /// The two relations of the kingdom contract; the symbol id is the row's position.
    pub fn relations(&self) -> KingdomRelations {
        let rows: Vec<(String, Vec<Ty>, Ty)> = self.rows.iter().map(|r| (r.name.clone(), r.args.clone(), r.out)).collect();
        let constants: Vec<&str> = self.rows.iter().filter(|r| r.kind == RowKind::Constant).map(|r| r.name.as_str()).collect();
        relations_from_rows(TRUE_WGSL, vec![self.root], &rows, &constants)
    }

    /// `|C_T|` for every constant-capable type.
    pub fn const_lens(&self) -> BTreeMap<Ty, u32> {
        self.constants.iter().map(|(t, v)| (*t, v.len() as u32)).collect()
    }

    /// `H`, the sum of the head lengths over the types the root reaches, at program size `n`.
    pub fn head_total(&self, contract: &KingdomContract, n: usize) -> Result<u64, String> {
        head_total_of(contract, self.root, n)
    }

    /// The work guard's fuel for kernels of this kingdom at program size `n`: `H` and the interface's `K`. Every
    /// device run of a kernel this kingdom produced passes it (`work_guard`).
    pub fn fuel(&self, contract: &KingdomContract, n: usize) -> Result<Fuel, String> {
        Ok(Fuel::Kingdom { h: self.head_total(contract, n)?, k: self.fuel_budget })
    }

    /// The bound table at program size `n` (design section 2): for every type the root reaches its head `h_T`, the
    /// demand `P_T`, the tail `t_T` and the Dc length `d_T`, the leaf bound `L` and the gene width; then the types
    /// the root cannot reach.
    pub fn bound_table(&self, contract: &KingdomContract, n: usize) -> Result<String, String> {
        let heads = contract.heads_from_program_size(self.root, n)?;
        let b = contract.bounds(self.root, &heads)?;
        let mut s = format!(
            "bound table at program size N = {n}: root {}, {} types reachable of {}, L = {}, gene width {} words (heads guarantee N = {})\n  {:<34} {:>5} {:>7} {:>7} {:>7}\n",
            self.type_name(self.root),
            b.types.len(),
            self.types.len(),
            b.l,
            b.width(),
            contract.guaranteed_program_size(self.root, &heads),
            "type",
            "h_T",
            "P_T",
            "t_T",
            "d_T"
        );
        let mut tys: Vec<_> = b.types.iter().collect();
        tys.sort_by_key(|t| (std::cmp::Reverse(t.h + t.t + if t.constant { t.d } else { 0 }), t.ty.code()));
        for t in &tys {
            s.push_str(&format!("  {:<34} {:>5} {:>7} {:>7} {:>7}{}\n", self.type_name(t.ty), t.h, t.p, t.t, t.d, if t.constant { "" } else { "  (no Dc)" }));
        }
        let reach: std::collections::BTreeSet<Ty> = b.types.iter().map(|t| t.ty).collect();
        let dead: Vec<String> = self.types.iter().map(|d| d.ty()).filter(|t| !reach.contains(t)).map(|t| self.type_name(t)).collect();
        s.push_str(&format!("  unreachable from the root ({}): {}\n", dead.len(), dead.join(" ")));
        Ok(s)
    }

    /// Spell an expressed tree: substitute each node's argument texts into its
    /// row's spelling, then put the interface's declarations in the entry point
    /// behind the prologue. Nothing else is done to the text.
    pub fn render(&self, tree: &Tree) -> Result<String, String> {
        let body = self.spell(tree)?;
        Ok(format!("{}{}", self.prologue, body.replace("<declarations>", self.declarations.trim_end_matches('\n'))))
    }

    fn spell(&self, node: &Tree) -> Result<String, String> {
        let row = self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
        if node.kids.len() != row.args.len() {
            return Err(format!("our software has a bug: {} has {} children, its signature {}", row.name, node.kids.len(), row.args.len()));
        }
        let kids: Vec<String> = node.kids.iter().map(|k| self.spell(k)).collect::<Result<_, _>>()?;
        let value = match (row.kind, node.const_index) {
            (RowKind::Constant, Some(i)) => Some(
                self.constants
                    .get(&row.out)
                    .and_then(|v| v.get(i as usize))
                    .ok_or_else(|| format!("our software has a bug: {}: constant index {i} is past C_T of {}", row.name, self.type_name(row.out)))?
                    .as_str(),
            ),
            (RowKind::Constant, None) => return Err(format!("our software has a bug: {}: a placeholder without a Dc index", row.name)),
            _ => None,
        };
        substitute(&row.spelling, &kids, value).map_err(|e| format!("our software has a bug: {}: {e}", row.name))
    }
}

/// `{N}`, `{N:F}` and `{v}` of a spelling; every other brace is copied.
pub fn substitute(spelling: &str, kids: &[String], value: Option<&str>) -> Result<String, String> {
    let b = spelling.as_bytes();
    let mut out = String::with_capacity(spelling.len() + kids.iter().map(String::len).sum::<usize>());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(close) = spelling[i..].find('}') {
                let inner = &spelling[i + 1..i + close];
                if inner == "v" {
                    out.push_str(value.ok_or("{v} in a row with no constant")?);
                    i += close + 1;
                    continue;
                }
                let (a, f) = match inner.split_once(':') {
                    Some((a, f)) => (a, Some(f)),
                    None => (inner, None),
                };
                if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && f.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit())) {
                    let n: usize = a.parse().map_err(|_| "argument index")?;
                    let text = kids.get(n).ok_or_else(|| format!("{{{inner}}} but only {} arguments", kids.len()))?;
                    match f {
                        None => out.push_str(text),
                        Some(f) => {
                            let f: usize = f.parse().map_err(|_| "field index")?;
                            out.push_str(text.split_whitespace().nth(f).ok_or_else(|| format!("{{{inner}}}: the argument has no field {f}: {text:?}"))?);
                        }
                    }
                    i += close + 1;
                    continue;
                }
            }
        }
        let ch = spelling[i..].chars().next().ok_or("utf8")?;
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// splitmix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How often a head cell draws a function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Share {
    /// A fair coin per cell.
    Half,
    /// In proportion to the functions against the terminals of the cell's type.
    Proportional,
}

/// The host typed initialiser (phylu `typed_init`, the same draws): a head cell
/// is a function or a terminal of its type, a tail cell a terminal of its type
/// drawn uniformly, a Dc cell an index below `|C_T|`; `root_function` makes the
/// root type's first head cell a function. Nothing is masked or redrawn.
pub fn random_gene(k: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut Rng, share: Share, root_function: bool) -> Vec<u32> {
    use crate::typed_gep::Domain;
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
        for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
            *cell = match s.domain {
                Domain::Head => {
                    let force = root_function && s.ty == layout.root && i == 0 && !functions.is_empty();
                    let p = match share {
                        Share::Half => 0.5,
                        Share::Proportional => functions.len() as f64 / (functions.len() + terminals.len()).max(1) as f64,
                    };
                    let function = !functions.is_empty() && (terminals.is_empty() || force || rng.unit() < p);
                    if function {
                        functions[rng.below(functions.len())]
                    } else {
                        terminals[rng.below(terminals.len())]
                    }
                }
                Domain::Tail => terminals[rng.below(terminals.len())],
                Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
            };
        }
    }
    gene
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kingdom() -> TrueKingdom {
        TrueKingdom::load(&default_dir()).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn substitution_copies_wgsl_braces_and_fills_arguments() {
        let kids = vec!["16 16".to_string(), "x".to_string()];
        assert_eq!(substitute("{ let r = {1}; w({0:1}) }", &kids, None).unwrap(), "{ let r = x; w(16) }");
        assert_eq!(substitute("lit_u32({v}u)", &[], Some("7")).unwrap(), "lit_u32(7u)");
        assert!(substitute("{2}", &kids, None).is_err());
        assert!(substitute("{v}", &[], None).is_err());
    }

    /// The kingdom check of rev 4 section 1 passes on the generated table: one output per symbol, ordered signatures,
    /// a terminal for every demanded type; the bound table at N = 8 is printed.
    #[test]
    fn the_generated_table_passes_the_kingdom_check_and_prints_its_bound_table() {
        let k = kingdom();
        let r = k.relations();
        let c = r.check().unwrap_or_else(|d| panic!("{} defects, first: {:?}", d.len(), &d[..d.len().min(5)]));
        assert_eq!(c.roots(), &[k.root]);
        eprintln!("{}", crate::kingdom_contract::report(&r, 8));
        eprintln!("{}", k.bound_table(&c, 8).unwrap());
        let heads = c.heads_from_program_size(k.root, 8).unwrap();
        let b = c.bounds(k.root, &heads).unwrap();
        assert!(b.width() > 0);
    }

    /// A small table by hand: one type R, a binary function and a terminal; `h = 1 + (N - 1) * 2`, so at N = 4,
    /// H = 7; with a second type T (R -> T -> R, a unary each) the sum is over both heads.
    #[test]
    fn the_head_total_is_the_sum_of_the_heads_of_a_small_table() {
        let (r, t) = (Ty::Wgsl(0), Ty::Wgsl(1));
        let one = relations_from_rows("one", vec![r], &[("add".into(), vec![r, r], r), ("x".into(), vec![], r)], &[]).check().expect("one type");
        assert_eq!(head_total_of(&one, r, 4).expect("H"), 7);
        let two = relations_from_rows("two", vec![r], &[("f".into(), vec![t], r), ("g".into(), vec![r], t), ("x".into(), vec![], r), ("y".into(), vec![], t)], &[]).check().expect("two types");
        let heads = two.heads_from_program_size(r, 4).expect("heads");
        assert_eq!(head_total_of(&two, r, 4).expect("H"), heads.values().sum::<usize>() as u64);
        assert!(head_total_of(&two, r, 4).expect("H") > 4, "more than N");
    }

    /// The crash configuration: K = 2 (2026-10-21 17:33), the Gram dispatch (the 128 x 64 rung, a 16 x 16
    /// workgroup) is over the cap, and so is even the unit-test dispatch (one workgroup); K = 0 at the unit-test
    /// dispatch is under it. The tables here are the K = 0 tables (a lower H than the K = 2 ones).
    #[test]
    fn the_crash_configuration_k2_with_the_gram_dispatch_is_refused_and_the_unit_test_at_k0_runs() {
        use crate::wgsl::gram::{band_layout, submit_invocations, RUNGS};
        use crate::wgsl::render::Grid;
        use crate::wgsl::work_guard::work_bound;
        let k = kingdom();
        assert_eq!(k.fuel_budget, 0, "interface.tsv is at fuel_budget 0 for the compiler and validity test");
        let c = k.relations().check().expect("check");
        let h = k.head_total(&c, 8).expect("H");
        let grid = Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false };
        let (d, m) = RUNGS[0];
        let bands = band_layout(d, m, 1 << 30);
        let rung = submit_invocations(grid, d, bands.band_rows.min(d), bands.item_bands);
        assert_eq!(rung, 64 * 256, "8 x 8 workgroups of 256");
        let crash = Fuel::Kingdom { h, k: 2 };
        let e = crash.check("the Gram rung", rung).expect_err("K = 2 at the Gram dispatch");
        eprintln!("H = {h}: K = 2 at the 128 x 64 rung: {}; K = 0 unit test: {}; {e}", work_bound(h, 2, rung), work_bound(h, 0, 256));
        assert!(k.fuel(&c, 8).expect("fuel").check("the unit test", 256).is_ok(), "K = 0 at the unit-test dispatch");
        assert!(k.fuel(&c, 8).expect("fuel").check("the Gram rung at K = 0", rung).is_ok(), "K = 0 even at the rung");
    }
}
