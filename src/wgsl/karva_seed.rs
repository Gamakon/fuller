//! ST4: a weak language model writes Karva symbol sequences for the true WGSL kingdom; every response decodes to a valid gene.
//!
//! The model is shown the gene's layout (per type: head length, tail length, reading order) and, for each type, the names
//! and spellings of its symbols. It writes one line per type, `TYPE: name name name ...`, the first rows of that type's
//! reading frame (`head_T` then `tail_T`). Nothing it writes can make a gene invalid, because a cell takes an INDEX into
//! its own alphabet:
//!
//! THE DECODING RULE (exact). A gene cell has an alphabet: a head cell of type `T` the symbols outputting `T` in
//! ascending symbol id (`GeneLayout::alphabet`), a tail cell the terminals of `T` in ascending symbol id, a Dc cell the
//! indices `0..|C_T|`. For the j-th token written on the line of type `T` (tokens split on whitespace and commas, with
//! surrounding backticks, quotes, asterisks and a trailing period removed) the cell is row `j` of `T`'s frame, and:
//! (1) if the token equals the name of a symbol in that cell's alphabet, the cell takes that symbol's index;
//! (2) otherwise the token's index is `FNV-1a-64(token bytes) mod |alphabet|`.
//!
//! A cell no token reached (rows past the end of the line, a type without a line) takes the index
//! `splitmix64(fill_seed + position * 0x9E3779B97F4A7C15) mod |alphabet|`, where for an unwritten HEAD cell the alphabet is
//! the type's terminals when it has any (so the tree ends where the model stopped writing) and the head alphabet otherwise;
//! a tail cell's is the terminals, a Dc cell's `0..|C_T|` (the model writes no constants). A line whose type name is not a framed type
//! is ignored, and a type named twice keeps its first line. A response with no usable line is a gene drawn wholly by the
//! fill. Every cell is an index into its own alphabet, so every response is a valid gene: there is no repair step.

use std::collections::BTreeMap;

use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::typed_gep::{Domain, GeneLayout};
use crate::wgsl::true_kingdom::TrueKingdom;

/// One framed type, with the alphabets of its cells.
#[derive(Debug, Clone)]
pub struct SeedFrame {
    pub ty: Ty,
    pub name: String,
    pub head_off: usize,
    pub head_len: usize,
    pub tail_off: usize,
    pub tail_len: usize,
    pub dc_off: Option<usize>,
    pub dc_len: usize,
    /// Symbols outputting the type, ascending symbol id.
    pub head_alpha: Vec<u32>,
    /// Terminals of the type, ascending symbol id.
    pub tail_alpha: Vec<u32>,
    /// `|C_T|`.
    pub const_len: u32,
}

/// What a decode did, for the report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecodeCounts {
    /// Tokens that named a symbol of their cell's alphabet.
    pub named: usize,
    /// Tokens mapped to an index by the hash.
    pub hashed: usize,
    /// Cells no token reached.
    pub filled: usize,
    /// Lines whose type name is not a framed type.
    pub ignored_lines: usize,
}

#[derive(Debug, Clone)]
pub struct SeedSpec<'a> {
    pub kingdom: &'a TrueKingdom,
    pub contract: &'a KingdomContract,
    pub layout: &'a GeneLayout,
    pub frames: Vec<SeedFrame>,
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A token as the model wrote it, less decoration.
fn clean(token: &str) -> &str {
    token.trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | '*' | '(' | ')' | '[' | ']')).trim_end_matches('.')
}

/// The first non-empty line of a spelling, whitespace collapsed, cut to `max` characters.
fn short_spelling(s: &str, max: usize) -> String {
    let line = s.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > max {
        let cut: String = collapsed.chars().take(max).collect();
        format!("{cut}...")
    } else {
        collapsed
    }
}

impl<'a> SeedSpec<'a> {
    pub fn new(kingdom: &'a TrueKingdom, contract: &'a KingdomContract, layout: &'a GeneLayout) -> SeedSpec<'a> {
        let const_len = kingdom.const_lens();
        let frames = layout
            .frames
            .iter()
            .map(|f| SeedFrame {
                ty: f.ty,
                name: kingdom.type_name(f.ty),
                head_off: f.head_off,
                head_len: f.head_len,
                tail_off: f.tail_off,
                tail_len: f.tail_len,
                dc_off: f.dc_off,
                dc_len: f.dc_len,
                head_alpha: (0..contract.n_symbols() as u32).filter(|s| contract.output(*s) == f.ty).collect(),
                tail_alpha: contract.terminals_of(f.ty).to_vec(),
                const_len: const_len.get(&f.ty).copied().unwrap_or(0),
            })
            .collect();
        SeedSpec { kingdom, contract, layout, frames }
    }

    /// The prompt: the task, the layout and reading rule, the output format, then every framed type with its symbols.
    /// `write_rows` is the most rows of a frame the model is asked to write; `phrase` is the seed phrase that makes the
    /// calls differ (it goes last, so the long common prefix stays identical between calls). A type's list is cut only
    /// when it has more than `max_list` symbols: then the first `max_list` by symbol id are shown and the line says how
    /// many are not (they remain reachable through the hash).
    pub fn prompt(&self, phrase: &str, write_rows: usize, max_list: usize) -> String {
        let mut p = String::new();
        p.push_str(
            "You are helping start a program search that builds WGSL compute kernels from fixed-size tables of symbols. \
You do not write a kernel: you write sequences of symbol names, and a fixed decoder turns each name into a table cell. \
A name that is not in a cell's list is not an error: it simply selects some other symbol of that cell's type. Your memory of what the kernel looks like is what we want.\n\n",
        );
        p.push_str(
            "TASK. The kernel computes the Gram matrix H = Phi^T Phi. Phi is an m x d matrix of f32 stored row-major in the storage buffer `phi` (element (k, i) at phi[k*d + i]); H is a d x d matrix of f32 in the buffer `h` \
(H[i][j] = sum over k of Phi[k][i] * Phi[k][j]); a uniform `cfg` holds d, m, r_rows and r_off. One invocation owns one cell (or a small block of cells) of H and writes only that. \
The kernel is a workgroup shape, a body that loops over k accumulating the products of the two columns, and an epilogue that stores the cell.\n\n",
        );
        p.push_str(&format!(
            "HOW THE TABLE READS. The root type is {}. Reading is breadth first from the root: each argument slot of a node asks for a TYPE, and the slot takes the next unused row of THAT TYPE's reading frame. \
A type's frame is its leading rows (any symbol of the type, functions and terminals) followed by its trailing rows (terminals only). The first rows of each frame are read first, so the first rows are the ones that matter. \
The root type's first row is read first, and each function you choose there asks, in order, for the argument types listed in parentheses after its name.\n\n",
            self.kingdom.type_name(self.layout.root)
        ));
        p.push_str(&format!(
            "OUTPUT. One line per type you want to write, exactly `TYPE: name name name ...` with the symbol names (the first word of each entry below, not the spelling) separated by spaces, in reading order, at most {write_rows} names per type. \
Follow the argument types: the root's function asks for its arguments, each of which is read from its own type's frame, and so on down; write a line for every type your tree will ask for, and write the rows in the order the slots will be asked (breadth first). \
You may first sketch the tree in a few plain lines that do not start with a type name; then the type lines. No code fences.\n\n"
        ));
        p.push_str("TYPES. Each header is `TYPE  leading rows + trailing rows`. An entry is `name(argument types, in order)  spelling` (spellings are cut to 30 characters, the first line only; `{0}`, `{1}` are the argument texts in order); `*` after a name marks a terminal (the only entries a trailing row takes).\n\n");
        for f in &self.frames {
            let shown = if f.head_alpha.len() > max_list { max_list } else { f.head_alpha.len() };
            p.push_str(&format!("[{}]  {} + {}\n", f.name, f.head_len, f.tail_len));
            for s in &f.head_alpha[..shown] {
                let row = &self.kingdom.rows[*s as usize];
                let mark = if self.contract.is_terminal(*s) { "*" } else { "" };
                let sig = if row.args.is_empty() { String::new() } else { format!("({})", row.args.iter().map(|t| self.kingdom.type_name(*t)).collect::<Vec<_>>().join(",")) };
                p.push_str(&format!("{}{}{}  {}\n", row.name, mark, sig, short_spelling(&row.spelling, 30)));
            }
            if shown < f.head_alpha.len() {
                p.push_str(&format!("... {} more symbols of this type are not shown (any name of this type you can recall still counts)\n", f.head_alpha.len() - shown));
            }
        }
        p.push_str(&format!("\nSEED PHRASE (it only varies the draw): {phrase}\nNow write the lines.\n"));
        p
    }

    /// Decode a response into a gene of `layout.width` words. See the module header for the rule.
    pub fn decode(&self, response: &str, fill_seed: u64) -> (Vec<u32>, DecodeCounts) {
        let mut counts = DecodeCounts::default();
        let mut written: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for line in response.lines() {
            let Some((left, right)) = line.split_once(':') else { continue };
            let name = left.trim().trim_start_matches(|c: char| matches!(c, '-' | '*' | '`' | '#' | '[' | ' ' | '0'..='9' | '.' | ')')).trim_end_matches(['`', ']', ' ']);
            match self.frames.iter().position(|f| f.name == name) {
                Some(i) => {
                    written.entry(i).or_insert_with(|| right.split(|c: char| c.is_whitespace() || c == ',').map(clean).filter(|t| !t.is_empty()).map(str::to_string).collect());
                }
                None => counts.ignored_lines += 1,
            }
        }
        let mut gene = vec![u32::MAX; self.layout.width];
        for (i, tokens) in &written {
            let f = &self.frames[*i];
            for (j, token) in tokens.iter().enumerate().take(f.head_len + f.tail_len) {
                let (pos, alpha) = if j < f.head_len { (f.head_off + j, &f.head_alpha) } else { (f.tail_off + j - f.head_len, &f.tail_alpha) };
                if alpha.is_empty() {
                    continue;
                }
                match alpha.iter().position(|s| self.kingdom.rows[*s as usize].name == *token) {
                    Some(k) => {
                        counts.named += 1;
                        gene[pos] = alpha[k];
                    }
                    None => {
                        counts.hashed += 1;
                        gene[pos] = alpha[(fnv1a64(token.as_bytes()) % alpha.len() as u64) as usize];
                    }
                }
            }
        }
        for s in self.layout.spans() {
            let f = self.frames.iter().find(|f| f.ty == s.ty).expect("a span's type is framed");
            for (pos, cell) in gene.iter_mut().enumerate().skip(s.off).take(s.len) {
                if *cell != u32::MAX {
                    continue;
                }
                counts.filled += 1;
                let r = splitmix64(fill_seed.wrapping_add((pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                *cell = match s.domain {
                    Domain::Head if !f.tail_alpha.is_empty() => f.tail_alpha[(r % f.tail_alpha.len() as u64) as usize],
                    Domain::Head => f.head_alpha[(r % f.head_alpha.len() as u64) as usize],
                    Domain::Tail => f.tail_alpha[(r % f.tail_alpha.len() as u64) as usize],
                    Domain::Dc => (r % u64::from(f.const_len.max(1))) as u32,
                };
            }
        }
        (gene, counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::{express, validate};
    use crate::wgsl::true_kingdom::default_dir;

    fn setup() -> (TrueKingdom, KingdomContract) {
        let k = TrueKingdom::load(&default_dir()).expect("table");
        let c = k.relations().check().expect("check");
        (k, c)
    }

    /// Whatever the model writes, every cell is an index into its own alphabet: empty text, noise, a foreign name, a
    /// name of the wrong type in a tail cell, far more tokens than rows, and a response naming real symbols all give
    /// genes that validate and express.
    #[test]
    fn every_response_decodes_to_a_valid_gene() {
        let (k, c) = setup();
        let layout = GeneLayout::from_program_size(&c, k.root, 8).expect("layout");
        let spec = SeedSpec::new(&k, &c, &layout);
        let const_len = k.const_lens();
        let first = &spec.frames[1];
        let real: String = first.head_alpha.iter().map(|s| k.rows[*s as usize].name.clone()).collect::<Vec<_>>().join(" ");
        let many = "x ".repeat(100_000);
        let mut lines = String::new();
        for f in &spec.frames {
            lines.push_str(&format!("{}: {} {} not_a_symbol 12 {{0}}\n", f.name, real, "kernel.entry_safe"));
        }
        let responses = ["", "no colon here at all", "kernel.safe: garbage", &format!("{}: {many}", spec.frames[0].name), &format!("{}: {real}", first.name), lines.as_str(), "unknown.type: a b c\n```\n"];
        for (i, r) in responses.iter().enumerate() {
            let (gene, _) = spec.decode(r, i as u64);
            assert_eq!(gene.len(), layout.width);
            validate(&c, &layout, &gene, &const_len).unwrap_or_else(|e| panic!("response {i}: {e}"));
            express(&c, &layout, &gene, &const_len).unwrap_or_else(|e| panic!("response {i}: {e}"));
        }
    }

    /// A name in the cell's alphabet takes its own symbol; a foreign text takes `fnv1a64 mod |alphabet|`; the fill is the
    /// documented splitmix of the position.
    #[test]
    fn the_rule_is_the_documented_one() {
        let (k, c) = setup();
        let layout = GeneLayout::from_program_size(&c, k.root, 8).expect("layout");
        let spec = SeedSpec::new(&k, &c, &layout);
        let f = spec.frames.iter().find(|f| f.head_alpha.len() > 3 && f.head_len > 2).expect("a frame");
        let want = f.head_alpha[2];
        let name = &k.rows[want as usize].name;
        let resp = format!("{}: {name} totally_foreign_text", f.name);
        let (gene, counts) = spec.decode(&resp, 5);
        assert_eq!(gene[f.head_off], want);
        assert_eq!(gene[f.head_off + 1], f.head_alpha[(fnv1a64(b"totally_foreign_text") % f.head_alpha.len() as u64) as usize]);
        assert_eq!((counts.named, counts.hashed), (1, 1));
        let pos = f.head_off + 2;
        let alpha = if f.tail_alpha.is_empty() { &f.head_alpha } else { &f.tail_alpha };
        assert_eq!(gene[pos], alpha[(splitmix64(5u64.wrapping_add((pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))) % alpha.len() as u64) as usize]);
    }

    #[test]
    fn the_prompt_names_every_type_and_ends_with_the_phrase() {
        let (k, c) = setup();
        let layout = GeneLayout::from_program_size(&c, k.root, 8).expect("layout");
        let spec = SeedSpec::new(&k, &c, &layout);
        let p = spec.prompt("alpha", 6, 200);
        for f in &spec.frames {
            assert!(p.contains(&format!("[{}]", f.name)));
        }
        assert!(p.contains("SEED PHRASE (it only varies the draw): alpha"));
    }
}
