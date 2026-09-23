//! `hff-chart` — recorded telemetry streams as a convergence figure for the paper.
//!
//! ```text
//! hff-chart --out <dir> [--x generation|elapsed] [--label <tag>=<text>] <stream.jsonl>...
//! ```
//!
//! A stream may be given as its file or as its run directory; a run card beside
//! it (`<dir>/card.json`, or `logs/cards/<tag>.json`) is picked up automatically
//! and is what lets a legend say "9 objectives" instead of a filename.
//!
//! WHY THE FIGURE IS WHAT IT IS — the argument is in [`fuller::evolve::chart`],
//! and it is worth the one line here: `log10 p` is DIMENSION-FREE, because the
//! objective count `m` is an argument to the incomplete beta that the CDF
//! consumes. So every run in the project goes on one set of axes and the
//! comparison is real, which an MSE trace could never be. And because a falling
//! p does NOT by itself mean a law was found, the figure is two stacked panels:
//! p above, `1 - R²` below, with both halves of the stop bar drawn.
//!
//! WHAT IT WRITES into `--out`:
//!
//! * `<tag>.dat` per series — whitespace columns, read by NAME.
//! * `convergence.tex` — the `\begin{figure}` fragment the paper `\input`s.
//! * `standalone.tex` — a wrapper so the figure compiles without the paper.
//!
//! The fragment carries no preamble: the document owns that, and the standalone
//! wrapper is how this compiles on its own while somebody else is editing the
//! paper.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fuller::evolve::chart::{figure, series, standalone, table, Series, Thinning, XAxis};

fn main() {
    if let Err(e) = run() {
        eprintln!("hff-chart: {e}");
        std::process::exit(1);
    }
}

const USAGE: &str = "hff-chart --out <dir> [--x generation|elapsed] [--label <tag>=<text>]\n              [--caption <text>] [--fig-label <text>] <stream.jsonl|run-dir>...";

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut out = PathBuf::from("chart");
    let mut x = XAxis::Generation;
    let mut inputs: Vec<String> = Vec::new();
    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    let mut caption = String::new();
    let mut fig_label = "fig:convergence".to_string();
    let mut thinning = Thinning::default();

    let mut i = 0;
    while i < argv.len() {
        let next = |i: usize, what: &str| -> Result<String, String> {
            argv.get(i + 1).cloned().ok_or_else(|| format!("{what} needs a value\n{USAGE}"))
        };
        match argv[i].as_str() {
            "--out" => {
                out = PathBuf::from(next(i, "--out")?);
                i += 2;
            }
            "--x" => {
                x = match next(i, "--x")?.as_str() {
                    "generation" | "gen" => XAxis::Generation,
                    "elapsed" | "elapsed_ms" | "time" => XAxis::ElapsedMs,
                    other => return Err(format!("--x is generation or elapsed, not {other}")),
                };
                i += 2;
            }
            "--label" => {
                let v = next(i, "--label")?;
                let (tag, text) = v.split_once('=').ok_or_else(|| format!("--label wants <tag>=<text>, got {v}"))?;
                labels.insert(tag.to_string(), text.to_string());
                i += 2;
            }
            "--caption" => {
                caption = next(i, "--caption")?;
                i += 2;
            }
            "--fig-label" => {
                fig_label = next(i, "--fig-label")?;
                i += 2;
            }
            "--max-gap" => {
                thinning.max_gap = next(i, "--max-gap")?.parse().map_err(|e| format!("--max-gap: {e}"))?;
                i += 2;
            }
            "--p-step" => {
                thinning.log10_p_step = next(i, "--p-step")?.parse().map_err(|e| format!("--p-step: {e}"))?;
                i += 2;
            }
            a if a.starts_with("--") => return Err(format!("unknown option {a}\n{USAGE}")),
            a => {
                inputs.push(a.to_string());
                i += 1;
            }
        }
    }
    if inputs.is_empty() {
        return Err(format!("no streams given\n{USAGE}"));
    }

    std::fs::create_dir_all(&out).map_err(|e| format!("output directory {}: {e}", out.display()))?;
    let mut built: Vec<Series> = Vec::new();
    for input in &inputs {
        let (stream_path, tag) = resolve(Path::new(input))?;
        let text = std::fs::read_to_string(&stream_path).map_err(|e| format!("{}: {e}", stream_path.display()))?;
        let card = find_card(&stream_path, &tag).and_then(|p| std::fs::read_to_string(p).ok());
        let mut s = series(&tag, &text, card.as_deref(), thinning)?;
        if let Some(text) = labels.get(&tag) {
            s.label = text.clone();
        }
        let dat = out.join(format!("{}.dat", s.tag));
        std::fs::write(&dat, table(&s, x)).map_err(|e| format!("{}: {e}", dat.display()))?;
        // WHETHER THE RUN EVER MET THE BAR, said on the way past. It is the one
        // question the figure exists to answer and an operator should not have
        // to read a PDF to learn that the answer was "no" for every series.
        let met = match s.stop_log10_p.zip(s.stop_one_minus_r2) {
            None => "   no bar recorded".to_string(),
            Some(_) => match s.first_met_bar() {
                Some(p) => format!("   MET BOTH HALVES at generation {}", p.generation),
                None => "   met neither half".to_string(),
            },
        };
        println!(
            "{:18} {:6} -> {:4} points   gen {} -> {}{}{}",
            s.tag,
            s.snapshots,
            s.points.len(),
            s.points.first().map_or(0, |p| p.generation),
            s.points.last().map_or(0, |p| p.generation),
            if s.saturated() { "   SATURATED (p = 0, off the scale)" } else { "" },
            met,
        );
        built.push(s);
    }

    let caption = if caption.is_empty() { default_caption(&built) } else { caption };
    let figure_text = figure(&built, x, &caption, &fig_label);
    let fig = out.join("convergence.tex");
    std::fs::write(&fig, &figure_text).map_err(|e| format!("{}: {e}", fig.display()))?;
    let wrapper = out.join("standalone.tex");
    std::fs::write(&wrapper, standalone("convergence.tex")).map_err(|e| format!("{}: {e}", wrapper.display()))?;
    println!("\nwrote {} and {}", fig.display(), wrapper.display());
    println!("compile it alone with:  cd {} && pdflatex standalone.tex", out.display());
    Ok(())
}

/// A caption that states what the figure is and what it must not be read as.
/// Written here rather than left to the author because the misreading it guards
/// against — a falling p taken for a law — has already cost real time.
fn default_caption(series: &[Series]) -> String {
    let mut c = String::from(
        "Convergence on the hyperspherical fitness function. $\\log_{10} p$ (top) is \
         $I_{\\sin^2\\theta}((m-1)/2, 1/2)$, in which the objective count $m$ is consumed by the \
         incomplete beta: $p$ is a probability on $[0,1]$, so $p = 10^{-19}$ means the same thing \
         at 6 objectives, at 9 and at 19{,}000, and runs under different objective sets are \
         directly comparable on this axis in a way no error metric allows. \
         A FALLING $p$ IS NOT BY ITSELF A RECOVERED LAW: the stop bar has two halves and both \
         must pass, so $1-R^2$ (bottom) is drawn beside it with the bar each run was actually \
         run under.",
    );
    if series.iter().any(Series::saturated) {
        c.push_str(
            " Markers on the floor of the top panel are $p$ underflowed to exactly zero \
             ($\\log_{10} p = -\\infty$) — the f32 angle saturated on the pole. They are off \
             the scale, not at a value, and they are not success: read the panel below them.",
        );
    }
    // A RUN WHOSE STREAM CARRIES NO BAR IS NAMED, not quietly drawn under
    // somebody else's rule: the rules belong to the runs that recorded them.
    let barless: Vec<String> = series
        .iter()
        .filter(|s| s.stop_log10_p.is_none() && s.stop_one_minus_r2.is_none())
        .map(|s| s.tag.replace('_', "\\_"))
        .collect();
    if !barless.is_empty() {
        c.push_str(&format!(
            " The dashed rules belong to the runs that recorded a bar; {} {} written before the \
             stream carried one, so no rule here is theirs.",
            barless.join(", "),
            if barless.len() == 1 { "was" } else { "were" },
        ));
    }
    c.push_str(
        " In the lower panel a SOLID line is train error and a DOTTED line of the same colour is \
         that run's third block (SMOGD/SMOTE synthetic rows), which enters the HFF angle like any \
         other objective. Validation error is measured and is in the data tables; it is left off \
         the panel only because it tracks train closely on every run drawn here.",
    );
    c.push_str(" Traces are thinned to their steps; no value is averaged and every point drawn is a beat the run held.");
    c
}

/// A stream given as its file, or as the run directory holding it.
fn resolve(input: &Path) -> Result<(PathBuf, String), String> {
    if input.is_dir() {
        let path = input.join("stream.jsonl");
        if !path.is_file() {
            return Err(format!("{} holds no stream.jsonl", input.display()));
        }
        let tag = input.file_name().and_then(|s| s.to_str()).unwrap_or("run").to_string();
        return Ok((path, tag));
    }
    if !input.is_file() {
        return Err(format!("{}: no such stream", input.display()));
    }
    // `logs/<tag>/stream.jsonl` names the run by its DIRECTORY; anything else is
    // named by its file stem.
    let tag = if input.file_name().and_then(|s| s.to_str()) == Some("stream.jsonl") {
        input.parent().and_then(Path::file_name).and_then(|s| s.to_str()).unwrap_or("run").to_string()
    } else {
        input.file_stem().and_then(|s| s.to_str()).unwrap_or("run").to_string()
    };
    Ok((input.to_path_buf(), tag))
}

/// The run card, at either of the two places one lives: beside the run, and in
/// the cards directory a run's `EVOLVE_CARD` writes to.
fn find_card(stream: &Path, tag: &str) -> Option<PathBuf> {
    let dir = stream.parent()?;
    let beside = dir.join("card.json");
    if beside.is_file() {
        return Some(beside);
    }
    let in_cards = dir.parent()?.join("cards").join(format!("{tag}.json"));
    in_cards.is_file().then_some(in_cards)
}
