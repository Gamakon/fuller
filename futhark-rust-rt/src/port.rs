//! THE WGSL PORT: Futhark's generated WGSL made acceptable to naga, the front end wgpu uses. A port of
//! `futhark-wgpu-shim/tools/port_futhark_c.py`'s `port_module` (the shim path ports the same modules with it, so the
//! two paths give naga the same text; a test of fuller compares them on the benchmarks' modules).
//!
//! Two constructs stop naga:
//!
//! 1. **Pointer arguments in the storage and workgroup address spaces** (`ptr<storage, array<T>, read_write>`):
//!    WGSL under the `unrestricted_pointer_parameters` language extension, which naga 30 does not implement. Each call
//!    whose pointer argument is `&name` or `&name[i]` is SPECIALISED into a copy of the helper that names the variable
//!    (`read_i32(&src_mem, i)` becomes `read_i32__src_mem(i)`), which means the same. A call with any other pointer
//!    argument is left as it is, and named in [`Ported::unported`]; naga then rejects the module by name.
//! 2. **Scalar-vector `bitcast` on f16**, a naga strictness: the functions using it are deleted; a kernel that needs
//!    one fails in naga by name.
//!
//! Every pointer helper left unspecialised is deleted (no caller needs it).

use std::collections::HashMap;

use regex::Regex;

use crate::Error;

/// A ported module and what the port did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ported {
    pub text: String,
    /// The functions deleted.
    pub removed: Vec<String>,
    /// The specialisations made: (helper, target, element).
    pub specialised: Vec<(String, String, bool)>,
    /// Calls left as they were (the pointer argument is neither `&name` nor `&name[i]`).
    pub unported: Vec<String>,
}

/// The name of the index argument an element specialisation takes.
const INDEX: &str = "futhark_port_index";

struct Res {
    fn_name: Regex,
    f16_bitcast: Regex,
    storage_pointer_arg: Regex,
    ident: Regex,
    element: Regex,
}

fn res() -> &'static Res {
    static R: std::sync::OnceLock<Res> = std::sync::OnceLock::new();
    R.get_or_init(|| Res {
        fn_name: Regex::new(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(").expect("a fixed regex"),
        f16_bitcast: Regex::new(r"(?i)bitcast<\s*vec2\s*<\s*f16\s*>\s*>|bitcast<\s*(?:i32|u32)\s*>\s*\(\s*vec2\s*<\s*f16\s*>").expect("a fixed regex"),
        storage_pointer_arg: Regex::new(r"(?i)ptr\s*<\s*(?:storage|workgroup)\b").expect("a fixed regex"),
        ident: Regex::new(r"^&\s*([A-Za-z_][A-Za-z0-9_]*)$").expect("a fixed regex"),
        element: Regex::new(r"(?s)^&\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]$").expect("a fixed regex"),
    })
}

/// `split_functions`: the line spans of top-level `fn` definitions, braces counted from the first `{`.
fn split_functions(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    let mut depth: i64 = 0;
    let mut seen_brace = false;
    let count = |l: &str| l.matches('{').count() as i64 - l.matches('}').count() as i64;
    for (i, line) in lines.iter().enumerate() {
        match start {
            None => {
                if line.starts_with("fn ") {
                    start = Some(i);
                    depth = count(line);
                    seen_brace = line.contains('{');
                    if seen_brace && depth <= 0 {
                        spans.push((i, i + 1));
                        start = None;
                    }
                }
            }
            Some(s) => {
                depth += count(line);
                seen_brace = seen_brace || line.contains('{');
                if seen_brace && depth <= 0 {
                    spans.push((s, i + 1));
                    start = None;
                }
            }
        }
    }
    spans
}

/// The text of a function up to its body's opening brace.
fn signature(text: &str) -> &str {
    text.split('{').next().unwrap_or(text)
}

/// `[(name, type)]` of a function signature.
fn params(sig: &str) -> Result<Vec<(String, String)>, Error> {
    let (a, b) = match (sig.find('('), sig.rfind(')')) {
        (Some(a), Some(b)) if a < b => (a, b),
        _ => return Err(Error::new(format!("the WGSL port found a signature it cannot read: {sig}"))),
    };
    let inner = &sig[a + 1..b];
    let mut out = Vec::new();
    let mut depth = 0i64;
    let mut cur = String::new();
    for c in inner.chars() {
        if c == '<' {
            depth += 1;
        } else if c == '>' {
            depth -= 1;
        }
        if c == ',' && depth == 0 {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.iter()
        .map(|p| match p.split_once(':') {
            Some((n, t)) => Ok((n.trim().to_string(), t.trim().to_string())),
            None => Err(Error::new(format!("the WGSL port found a parameter it cannot read: {p}"))),
        })
        .collect()
}

struct Helper {
    name: String,
    text: String,
    ptr_param: String,
}

impl Helper {
    fn new(name: &str, text: &str) -> Result<Helper, Error> {
        let r = res();
        let ps = params(signature(text))?;
        let first_is_ptr = ps.first().is_some_and(|p| r.storage_pointer_arg.is_match(&p.1));
        let others = ps.iter().skip(1).any(|p| r.storage_pointer_arg.is_match(&p.1));
        if !first_is_ptr || others {
            return Err(Error::new(format!("{name}: a pointer argument that is not the first, or more than one; not ported")));
        }
        Ok(Helper { name: name.to_string(), text: text.to_string(), ptr_param: ps[0].0.clone() })
    }

    /// This helper with its pointer argument fixed to `target` (or to `target[INDEX]` for an element).
    fn specialise(&self, target: &str, element: bool) -> Result<String, Error> {
        let p = &self.ptr_param;
        let (sig, body) = self.text.split_once('{').ok_or_else(|| Error::new(format!("{}: a helper with no body", self.name)))?;
        let mut kept: Vec<(String, String)> = params(sig)?.into_iter().skip(1).collect();
        let (place, name) = if element {
            kept.insert(0, (INDEX.to_string(), "i32".to_string()));
            (format!("{target}[{INDEX}]"), format!("{}__{target}__elem", self.name))
        } else {
            (target.to_string(), format!("{}__{target}", self.name))
        };
        let tail = &sig[sig.rfind(')').map(|i| i + 1).unwrap_or(sig.len())..];
        let new_sig = format!("fn {name}({}){tail}", kept.iter().map(|(n, t)| format!("{n}: {t}")).collect::<Vec<_>>().join(", "));
        let e = regex::escape(p);
        let deref_paren = Regex::new(&format!(r"\(\s*\*\s*{e}\s*\)")).map_err(|x| Error::new(x.to_string()))?;
        let deref = Regex::new(&format!(r"\*\s*{e}\b")).map_err(|x| Error::new(x.to_string()))?;
        let new_body = deref_paren.replace_all(body, regex::NoExpand(&place)).into_owned();
        let new_body = deref.replace_all(&new_body, regex::NoExpand(&place)).into_owned();
        // The pointer itself, passed on: `(?<![\w.&*])p\b`, by hand (no look-behind in the regex crate).
        let bare = Regex::new(&format!(r"{e}\b")).map_err(|x| Error::new(x.to_string()))?;
        let mut out = String::new();
        let mut last = 0;
        for m in bare.find_iter(&new_body) {
            let prev = new_body[..m.start()].chars().next_back();
            let blocked = prev.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '&' || c == '*');
            if blocked {
                continue;
            }
            out.push_str(&new_body[last..m.start()]);
            out.push('&');
            out.push_str(&place);
            last = m.end();
        }
        out.push_str(&new_body[last..]);
        Ok(format!("{new_sig}{{{out}"))
    }
}

/// `(the first argument's text, the index of the , or ) ending it)`.
fn first_argument(text: &str, i: usize) -> Result<(&str, usize), Error> {
    let b = text.as_bytes();
    let mut depth = 0i64;
    let mut j = i;
    while j < b.len() {
        match b[j] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => {
                if depth == 0 {
                    return Ok((&text[i..j], j));
                }
                depth -= 1;
            }
            b',' if depth == 0 => return Ok((&text[i..j], j)),
            _ => {}
        }
        j += 1;
    }
    Err(Error::new("an unterminated call in the WGSL"))
}

/// `((x))` -> `x`, only when the outer parentheses enclose the whole.
fn strip_parens(arg: &str) -> &str {
    let mut arg = arg.trim();
    while arg.starts_with('(') && arg.ends_with(')') {
        let mut depth = 0i64;
        let mut whole = true;
        for (k, c) in arg.char_indices() {
            if c == '(' {
                depth += 1;
            }
            if c == ')' {
                depth -= 1;
            }
            if depth == 0 && k != arg.len() - 1 {
                whole = false;
                break;
            }
        }
        if !whole {
            break;
        }
        arg = arg[1..arg.len() - 1].trim();
    }
    arg
}

/// For `&x[e]`: `(x, e)` when the final `]` closes the first `[`.
fn balanced_index(arg: &str) -> Option<(String, String)> {
    let c = res().element.captures(arg)?;
    let inner = c.get(2)?.as_str();
    let mut depth = 0i64;
    for ch in inner.chars() {
        if ch == '(' || ch == '[' {
            depth += 1;
        }
        if ch == ')' || ch == ']' {
            depth -= 1;
        }
        if depth < 0 {
            return None;
        }
    }
    (depth == 0).then(|| (c[1].to_string(), inner.to_string()))
}

struct Rewriter {
    helpers: HashMap<String, Helper>,
    call: Option<Regex>,
    made: HashMap<(String, String, bool), Option<String>>,
    order: Vec<(String, String, bool)>,
    unported: Vec<String>,
}

impl Rewriter {
    fn rewrite(&mut self, text: &str) -> Result<String, Error> {
        let Some(call) = self.call.clone() else {
            return Ok(text.to_string());
        };
        let mut out = String::new();
        let mut pos = 0;
        loop {
            let Some(c) = call.captures_at(text, pos) else {
                out.push_str(&text[pos..]);
                return Ok(out);
            };
            let m = c.get(0).ok_or_else(|| Error::new("our software has a bug: a match with no text"))?;
            let name = c[1].to_string();
            let (arg, end) = first_argument(text, m.end())?;
            let core = strip_parens(arg);
            let ident = res().ident.captures(core).map(|c| c[1].to_string());
            let elem = if ident.is_some() { None } else { balanced_index(core) };
            let target = match (&ident, &elem) {
                (Some(t), _) => t.clone(),
                (None, Some((t, _))) => t.clone(),
                (None, None) => {
                    let s = &text[m.start()..(m.start() + 100).min(text.len())];
                    let s = s.split('\n').next().unwrap_or(s);
                    self.unported.push(s.to_string());
                    out.push_str(&text[pos..m.end()]);
                    pos = m.end();
                    continue;
                }
            };
            let key = (name.clone(), target.clone(), elem.is_some());
            if !self.made.contains_key(&key) {
                self.made.insert(key.clone(), None);
                let spec = self.helpers[&name].specialise(&target, elem.is_some())?;
                let done = self.rewrite(&spec)?;
                self.made.insert(key.clone(), Some(done));
                self.order.push(key);
            }
            let new_name = format!("{name}__{target}{}", if elem.is_some() { "__elem" } else { "" });
            out.push_str(&text[pos..m.start()]);
            let next_is_comma = text.as_bytes()[end] == b',';
            match &elem {
                Some((_, index)) => {
                    let index = self.rewrite(index)?;
                    out.push_str(&format!("{new_name}({index}{}", if next_is_comma { ", " } else { ")" }));
                }
                None => out.push_str(&format!("{new_name}({}", if next_is_comma { "" } else { ")" })),
            }
            pos = end + 1;
        }
    }
}

/// Port one WGSL module.
pub fn port_module(wgsl: &str) -> Result<Ported, Error> {
    let r = res();
    let lines: Vec<&str> = wgsl.split('\n').collect();
    let spans = split_functions(&lines);
    let mut helpers = HashMap::new();
    let mut helper_order = Vec::new();
    let mut f16 = std::collections::HashSet::new();
    for &(start, end) in &spans {
        let text = lines[start..end].join("\n");
        let Some(c) = r.fn_name.captures(lines[start].trim()) else { continue };
        let name = c[1].to_string();
        if r.f16_bitcast.is_match(&text) {
            f16.insert(name);
        } else if r.storage_pointer_arg.is_match(signature(&text)) {
            helper_order.push(name.clone());
            helpers.insert(name.clone(), Helper::new(&name, &text)?);
        }
    }
    let mut removed = Vec::new();
    let mut drop = std::collections::HashSet::new();
    for &(start, end) in &spans {
        if let Some(c) = r.fn_name.captures(lines[start].trim()) {
            let n = &c[1];
            if helpers.contains_key(n) || f16.contains(n) {
                removed.push(n.to_string());
                drop.extend(start..end);
            }
        }
    }
    let rest: String = lines.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, l)| *l).collect::<Vec<_>>().join("\n");
    let call = if helpers.is_empty() {
        None
    } else {
        let alt = helper_order.iter().map(|h| regex::escape(h)).collect::<Vec<_>>().join("|");
        Some(Regex::new(&format!(r"\b({alt})\(")).map_err(|e| Error::new(e.to_string()))?)
    };
    let mut rw = Rewriter { helpers, call, made: HashMap::new(), order: Vec::new(), unported: Vec::new() };
    let rest = rw.rewrite(&rest)?;
    let specs: Vec<String> = rw.order.iter().map(|k| rw.made[k].clone().unwrap_or_default()).collect();
    let specs = specs.join("\n");
    let text = if specs.is_empty() { rest } else { format!("{rest}\n{specs}\n") };
    Ok(Ported { text, removed, specialised: rw.order, unported: rw.unported })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_storage_pointer_helper_is_specialised_at_each_call() {
        let src = "var<storage, read_write> xs: array<i32>;\n\
                   fn read_i32(buffer: ptr<storage, array<i32>, read_write>, i: i32) -> i32 {\n  return (*buffer)[i];\n}\n\
                   fn main() {\n  let a = read_i32(&xs, 3);\n  let b = atomic_add((&xs[2]), 1);\n}\n";
        let p = port_module(src).expect("the port");
        assert!(p.text.contains("read_i32__xs( 3)"), "{}", p.text);
        assert!(p.text.contains("fn read_i32__xs(i: i32) -> i32 {\n  return xs[i];"), "{}", p.text);
        assert_eq!(p.removed, vec!["read_i32".to_string()]);
        assert!(p.unported.is_empty());
    }
}
