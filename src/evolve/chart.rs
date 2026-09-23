//! THE CONVERGENCE CHART — a recorded telemetry stream as a figure for the paper.
//!
//! An evolutionary-computation paper draws error against generations. This draws
//! something else, and the reason is the whole point of the module.
//!
//! Andrew: *"We could put together our charts in terms of the hyperspherical
//! fitness function — the log of the p-value. What's nice about it is that it's
//! dimension free. So whatever fitness function we decide is important, it
//! doesn't matter. The patterns that we see across different schemes, those
//! patterns are directly comparable."*
//!
//! The property that buys it is in [`crate::evolve::engine::hff_p_value`]:
//! `I_{sin² θ}((m-1)/2, 1/2)`, the regularised incomplete beta, with the
//! dimension `m` as an ARGUMENT that the CDF consumes. What comes out is a
//! probability on [0, 1], so `p = 1e-19` means the same thing at 6 objectives,
//! at 9, and at 19,000. An MSE trace for a two-variable law and one for a
//! six-variable law share an axis but not a meaning and cannot honestly be
//! overlaid; a `log10 p` trace can. That is what makes one set of axes hold
//! every run in the project.
//!
//! # AND WHY A P-ONLY CHART WOULD BE A LIE
//!
//! `log10 p` falling does NOT mean the fit is approaching a law, and it has
//! misled in BOTH directions on real runs:
//!
//! | run | log10 p | train 1-R² | what actually happened |
//! |---|---|---|---|
//! | `bacres1_75long` | **-inf** | 2.6e-6 | the f32 angle SATURATED; the model is not exact |
//! | `bacres1_20k` | -13.65 | 2.6e-14 | the error was four orders PAST the bar and p still failed |
//!
//! The fit's stop bar has two halves and BOTH must pass ([`Series::stop_log10_p`]
//! and [`Series::stop_one_minus_r2`], read off the stream's own `run_start` —
//! a chart must never own a threshold the engine owns). So the figure this
//! module emits is TWO STACKED PANELS on a shared generation axis: `log10 p`
//! above, `1 - R²` below for train, validation and the third block. Not twin
//! y-axes — two scales overlaid on one frame leave a reader unable to say which
//! line owns which axis. Stacked, each panel carries its own half of the bar as
//! a dashed rule, and the cause of a p plateau is visible directly underneath
//! the plateau.
//!
//! # SATURATION IS A CLASSIFICATION, NOT A MISSING NUMBER
//!
//! `telemetry::finite` maps a non-finite metric to JSON `null`, so a p-value of
//! exactly 0 — `log10 p = -inf`, the f32 angle collapsed onto the pole — arrives
//! as `null`, which is the same token an unscored generation writes. They are
//! not the same fact and [`PointKind`] does not conflate them: a null `log10_p`
//! whose `best_hff` is exactly 0.0 is SATURATED, and a null `log10_p` with no
//! `best_hff` at all is UNMEASURED. A saturated point is plotted as a marker on
//! a floor rule with its own legend entry, and never as a finite number. The
//! `log10_p` column carries the literal `nan` for it, which pgfplots breaks the
//! line at under `unbounded coords=jump`.
//!
//! # THINNING KEEPS THE STAIRCASE
//!
//! A convergence trace is a staircase: long flats broken by steps. 1,626 points
//! is more than a figure needs and more than TikZ compiles happily, but an
//! average over a window would round the corners off the steps — which are the
//! only part of the shape that carries information. [`thin`] therefore emits a
//! point when a value MOVED (absolute on `log10 p`, relative on the R²s), plus
//! the first, the last, every change of [`PointKind`], and a forced point after
//! a maximum gap so a long flat still draws as a line. Nothing is averaged and
//! no emitted point is a value the run did not hold.

use std::collections::BTreeMap;

use super::telemetry::{parse_stream, Record};

/// What a point's `log10 p` IS, which is not answerable from the number alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointKind {
    /// A real measurement: `log10_p` is finite.
    Measured,
    /// THE ANGLE SATURATED. `log10_p` was `null` and `best_hff` was exactly 0.0:
    /// the f32 angle collapsed onto the pole, p underflowed to 0 and its log is
    /// -inf. This is NOT success — `bacres1_75long` sat here for 63,000
    /// generations with train 1-R² at 2.6e-6, four orders short of the bar.
    Saturated,
    /// No p at all: the generation produced no scored best. A silence, and a
    /// different fact from saturation.
    Unmeasured,
}

/// One beat of a run, reduced to what the figure draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub generation: u32,
    pub elapsed_ms: u64,
    /// `None` when the kind is not [`PointKind::Measured`]. A saturated point
    /// has no finite p and must never be given one.
    pub log10_p: Option<f64>,
    pub kind: PointKind,
    /// `1 - R²` for train, validation and the third block — the stop bar's other
    /// half, in the units the bar is written in. `None` where the stream had no
    /// R² for that block (a run with no third block has no third column).
    pub one_minus_r2_train: Option<f64>,
    pub one_minus_r2_val: Option<f64>,
    pub one_minus_r2_third: Option<f64>,
}

impl Point {
    /// Whether this beat met BOTH halves of the bar it was run under. The p half
    /// counts as met when the angle saturated: p = 0 is under any bar. Returns
    /// `None` when the stream did not carry a bar to judge against.
    pub fn meets_bar(&self, stop_log10_p: Option<f64>, stop_one_minus_r2: Option<f64>) -> Option<bool> {
        let (bar_p, bar_r2) = (stop_log10_p?, stop_one_minus_r2?);
        let p_half = match self.kind {
            PointKind::Saturated => true,
            PointKind::Measured => self.log10_p.is_some_and(|p| p <= bar_p),
            PointKind::Unmeasured => false,
        };
        let r2_half = [self.one_minus_r2_train, self.one_minus_r2_val]
            .iter()
            .all(|v| v.is_some_and(|v| v <= bar_r2));
        Some(p_half && r2_half)
    }
}

/// One run, as a chart draws it: its points and the facts a legend must tell the
/// truth with.
#[derive(Clone, Debug)]
pub struct Series {
    /// The run's tag — its log directory and its telemetry `run_id`.
    pub tag: String,
    /// What the legend says. From the run card's `derived.hff_objectives` when
    /// there is a card, else built from the `run_start` — and never guessed: the
    /// objective count is a function of five settings the stream does not carry
    /// (see `Engine::hff_dimensions`), so without a card the label says what the
    /// stream DOES know and stops there.
    pub label: String,
    pub points: Vec<Point>,
    /// THE BAR THIS RUN WAS ACTUALLY RUN UNDER, off its own `run_start`. `None`
    /// for a stream written before the field existed, and the chart then draws no
    /// rule rather than supplying one of its own — a figure carrying its own -19
    /// would colour one run's p against another run's bar.
    pub stop_log10_p: Option<f64>,
    pub stop_one_minus_r2: Option<f64>,
    /// How many snapshots the stream held before [`thin`] ran, so a caption can
    /// say what the figure is a thinning of.
    pub snapshots: usize,
}

impl Series {
    /// The first thinned point that met both halves of the bar, if any — the
    /// generation at which this run could honestly have stopped.
    pub fn first_met_bar(&self) -> Option<&Point> {
        self.points.iter().find(|p| p.meets_bar(self.stop_log10_p, self.stop_one_minus_r2) == Some(true))
    }

    /// Whether the run ever saturated its angle — which a caption must say,
    /// because a trace that stops drawing halfway down otherwise looks like a
    /// run that stopped.
    pub fn saturated(&self) -> bool {
        self.points.iter().any(|p| p.kind == PointKind::Saturated)
    }

    /// The floor a saturated marker sits on: below every finite p the series
    /// holds and below its own bar, so the marker is visibly OFF the scale
    /// rather than at a value the run reached.
    pub fn saturation_floor(&self) -> f64 {
        let lowest = self.points.iter().filter_map(|p| p.log10_p).fold(f64::INFINITY, f64::min);
        let bar = self.stop_log10_p.unwrap_or(f64::INFINITY);
        let deepest = lowest.min(bar);
        if deepest.is_finite() {
            deepest - 1.0
        } else {
            -1.0
        }
    }
}

/// How far a value must move before [`thin`] keeps the point.
#[derive(Clone, Copy, Debug)]
pub struct Thinning {
    /// ABSOLUTE movement in `log10 p`. It is already a log, so an absolute step
    /// here is a relative step in p.
    pub log10_p_step: f64,
    /// RELATIVE movement in `1 - R²`, as a fraction. `1 - R²` spans fourteen
    /// orders of magnitude across a run and an absolute threshold would keep
    /// every point at the top and none at the bottom.
    pub r2_relative_step: f64,
    /// The longest run of generations that may pass with nothing emitted. A flat
    /// stretch is real and must draw as a line, not as a gap.
    pub max_gap: u32,
}

impl Default for Thinning {
    fn default() -> Thinning {
        Thinning { log10_p_step: 0.01, r2_relative_step: 0.02, max_gap: 200 }
    }
}

/// Which quantity the x-axis carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XAxis {
    /// The fit's own clock — the axis an EC paper draws.
    Generation,
    /// Wall clock. The honest axis when schemes differ in cost per generation:
    /// 800+400 runs 27 ms a generation and 100k+100k runs 1,600 ms, so the same
    /// count of beats is sixty times the machine.
    ElapsedMs,
}

/// Read a stream and reduce it to a [`Series`].
///
/// `card` is the run card's JSON, when there is one. Only two fields are read
/// from it and they are read by name rather than by deserialising `Card`: that
/// type carries the whole `Config` and lives behind the `gpu` feature, and this
/// module must build on a machine with no adapter for the same reason the viewer
/// must.
pub fn series(tag: &str, stream: &str, card: Option<&str>, thinning: Thinning) -> Result<Series, String> {
    let (records, _bad) = parse_stream(stream);
    let start = records.iter().find_map(|r| match r {
        Record::RunStart(s) => Some(s.clone()),
        _ => None,
    });
    let mut all = Vec::new();
    for r in &records {
        if let Record::Snapshot(s) = r {
            let g = &s.global;
            let kind = match (g.log10_p, g.best_hff) {
                (Some(p), _) if p.is_finite() => PointKind::Measured,
                // p is null and the angle is EXACTLY zero: the f32 angle
                // collapsed onto the pole. p underflowed to 0, log10 p is -inf.
                (None, Some(0.0)) => PointKind::Saturated,
                _ => PointKind::Unmeasured,
            };
            all.push(Point {
                generation: s.header.generation,
                elapsed_ms: s.header.elapsed_ms,
                log10_p: if kind == PointKind::Measured { g.log10_p } else { None },
                kind,
                one_minus_r2_train: g.r2_train.map(|r| 1.0 - r),
                one_minus_r2_val: g.r2_val.map(|r| 1.0 - r),
                one_minus_r2_third: g.r2_third.map(|r| 1.0 - r),
            });
        }
    }
    if all.is_empty() {
        return Err(format!("{tag}: the stream holds no snapshots"));
    }
    let snapshots = all.len();
    let points = thin(&all, thinning);
    let label = label_for(tag, start.as_ref(), card);
    Ok(Series {
        tag: tag.to_string(),
        label,
        points,
        stop_log10_p: start.as_ref().and_then(|s| s.stop_log10_p),
        stop_one_minus_r2: start.as_ref().and_then(|s| s.stop_one_minus_r2),
        snapshots,
    })
}

/// What the legend says about a run, built from what is actually known.
///
/// THE OBJECTIVE COUNT COMES FROM THE CARD OR NOT AT ALL. It is
/// `hff_columns(...) + redundancy + tower` over five settings the telemetry
/// stream does not carry, so inferring "9 objectives" from `n_extrap > 0` would
/// be a guess printed as a fact. With no card the label says the population, the
/// pump and whether there is a third block, all of which the `run_start` holds.
fn label_for(tag: &str, start: Option<&super::telemetry::RunStart>, card: Option<&str>) -> String {
    let objectives = card
        .and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok())
        .and_then(|v| v.get("derived").and_then(|d| d.get("hff_objectives")).and_then(serde_json::Value::as_u64));
    let Some(s) = start else {
        return tag.to_string();
    };
    let mut parts = vec![format!("{}+{}", s.pop_intake, s.pop_champion)];
    if s.pump_every > 0 {
        parts.push(format!("pump {}", s.pump_every));
    }
    match objectives {
        Some(n) => parts.push(format!("{n} objectives")),
        None if s.n_extrap > 0 => parts.push("third block".to_string()),
        None => parts.push("no third block".to_string()),
    }
    format!("{tag} ({})", parts.join(", "))
}

/// Thin a trace to the points that carry its shape.
///
/// Kept: the first, the last, every point whose [`PointKind`] differs from the
/// one before it (so the entry into and exit from saturation is never lost),
/// every point where a drawn value MOVED by more than its threshold, and one
/// point per `max_gap` generations through a flat.
///
/// Not done: any averaging, smoothing or resampling. Every emitted point is a
/// beat the run actually had, with the numbers it actually held.
pub fn thin(points: &[Point], t: Thinning) -> Vec<Point> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let moved_rel = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() > t.r2_relative_step * a.abs().max(b.abs()).max(f64::MIN_POSITIVE),
        (None, None) => false,
        _ => true,
    };
    let mut out = vec![points[0]];
    let mut last = points[0];
    for p in &points[1..points.len() - 1] {
        let stepped = match (p.log10_p, last.log10_p) {
            (Some(a), Some(b)) => (a - b).abs() > t.log10_p_step,
            (None, None) => false,
            _ => true,
        };
        let keep = p.kind != last.kind
            || stepped
            || moved_rel(p.one_minus_r2_train, last.one_minus_r2_train)
            || moved_rel(p.one_minus_r2_val, last.one_minus_r2_val)
            || moved_rel(p.one_minus_r2_third, last.one_minus_r2_third)
            || p.generation.saturating_sub(last.generation) >= t.max_gap;
        if keep {
            out.push(*p);
            last = *p;
        }
    }
    out.push(points[points.len() - 1]);
    out
}

// ---------------------------------------------------------------------------
// The emitters
// ---------------------------------------------------------------------------

/// A series as a pgfplots-readable table: whitespace columns with a header row,
/// which `\addplot table[x=..,y=..]` reads by NAME, so adding a column later
/// cannot silently shift a plot onto the wrong one.
///
/// A value the run did not hold is written `nan`, which pgfplots skips under
/// `unbounded coords=jump` — it is never written as a number, and a saturated
/// p is never written as its floor in the `log10_p` column. The floor lives in
/// its own `sat` column so the marker and the line cannot be confused.
pub fn table(s: &Series, x: XAxis) -> String {
    let n = |v: Option<f64>| match v {
        Some(v) if v.is_finite() => format!("{v:.12e}"),
        _ => "nan".to_string(),
    };
    let floor = s.saturation_floor();
    let mut out = String::from("x generation elapsed_ms log10_p sat one_minus_r2_train one_minus_r2_val one_minus_r2_third\n");
    for p in &s.points {
        let xv = match x {
            XAxis::Generation => f64::from(p.generation),
            XAxis::ElapsedMs => p.elapsed_ms as f64,
        };
        let sat = if p.kind == PointKind::Saturated { format!("{floor:.6}") } else { "nan".to_string() };
        out.push_str(&format!(
            "{} {} {} {} {} {} {} {}\n",
            xv,
            p.generation,
            p.elapsed_ms,
            n(p.log10_p),
            sat,
            n(p.one_minus_r2_train),
            n(p.one_minus_r2_val),
            n(p.one_minus_r2_third),
        ));
    }
    out
}

/// The figure: a `groupplot` of two stacked axes on one shared x.
///
/// Emitted as a `\begin{figure}` fragment with no preamble in it, so the paper
/// `\input`s it and the figure restyles with the document. The paper needs one
/// line added to its own preamble, which [`PREAMBLE`] states.
///
/// The two halves of the stop bar are drawn from the SERIES' OWN values, and a
/// bar that differs between series is drawn once per distinct value and named
/// in the caption rather than averaged into one rule.
pub fn figure(series: &[Series], x: XAxis, caption: &str, label: &str) -> String {
    let xlabel = match x {
        XAxis::Generation => "generation",
        XAxis::ElapsedMs => "elapsed (ms)",
    };
    // One rule per DISTINCT bar. Sorted, so the output is deterministic whatever
    // order the series arrived in.
    let mut bars_p: BTreeMap<String, f64> = BTreeMap::new();
    let mut bars_r2: BTreeMap<String, f64> = BTreeMap::new();
    for s in series {
        if let Some(v) = s.stop_log10_p {
            bars_p.insert(format!("{v:.6}"), v);
        }
        if let Some(v) = s.stop_one_minus_r2 {
            bars_r2.insert(format!("{v:.3e}"), v);
        }
    }
    let colours = ["blue!70!black", "red!70!black", "green!45!black", "orange!80!black", "violet", "brown", "teal", "magenta!70!black"];
    let marks = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o", "square", "triangle"];

    let mut out = String::new();
    out.push_str("% THE CONVERGENCE CHART, emitted by `hff-chart` from recorded telemetry.\n");
    out.push_str("% Generated — do not edit by hand. Regenerate with the command in the caption.\n");
    // The preamble this needs is NAMED, not written: a \usepackage in a fragment
    // the paper \inputs is a package loaded after \begin{document}. The lines are
    // in `chart::PREAMBLE` and the standalone wrapper supplies them.
    out.push_str("% Requires pgfplots (compat=1.18) and its groupplots library in the\n");
    out.push_str("% DOCUMENT PREAMBLE. See `fuller::evolve::chart::PREAMBLE`, or compile\n");
    out.push_str("% the standalone.tex written beside this file, which supplies them.\n");
    out.push_str("\\begin{figure}[htbp]\n\\centering\n\\begin{tikzpicture}\n");
    out.push_str("\\begin{groupplot}[\n");
    out.push_str("  group style={group size=1 by 2, vertical sep=7mm, x descriptions at=edge bottom},\n");
    // Tall enough that the 1-R^2 panel's fourteen decades are readable, and a
    // legend BELOW the axis rather than inside it: with six runs an inside
    // legend covers the very plateau the figure exists to show, which is what
    // the first compile of this figure did.
    out.push_str("  width=0.94\\linewidth, height=62mm,\n");
    // A LOG x-AXIS, because the runs span 360 to 65,000 generations: on a linear
    // axis the 1,740-generation 100k-population run is a dot at the origin.
    out.push_str("  xmode=log, log basis x=10,\n");
    out.push_str(&format!("  xlabel={{{xlabel}}},\n"));
    // `nan` in a column is a BREAK IN THE LINE, not a point at zero. This is
    // what lets a saturated stretch leave a gap instead of a plunge to the
    // floor that would read as a measured value.
    out.push_str("  unbounded coords=jump,\n");
    out.push_str("  grid=both, grid style={gray!18},\n");
    out.push_str("  tick label style={font=\\small}, label style={font=\\small},\n");
    out.push_str("  legend style={font=\\scriptsize, draw=gray!40, at={(0.5,-0.28)}, anchor=north, cells={anchor=west}},\n");
    out.push_str("]\n\n");

    // ---- panel 1: log10 p, the dimension-free half ----------------------
    out.push_str("% Panel 1 --- log10 p. DIMENSION-FREE: p is I_{sin^2 theta}((m-1)/2, 1/2),\n");
    out.push_str("% so m is consumed by the CDF and p = 1e-19 means the same thing at any m.\n");
    out.push_str("% This is what makes runs with different objective counts overlayable.\n");
    // NO run legend on this panel: the colours are the same on both, and the
    // bottom panel carries the key. Only the saturation marker is named here,
    // because it appears nowhere else and a reader must not have to guess what a
    // row of dots on the floor is.
    // The legend sits in the panel's empty middle-left: the traces run along the
    // top and the saturation markers along the floor at the right, so this is
    // the one region of the panel that carries no data.
    out.push_str("\\nextgroupplot[ylabel={$\\log_{10} p$}, legend style={font=\\scriptsize, draw=gray!40, at={(0.02,0.42)}, anchor=west, cells={anchor=west}}]\n");
    for (i, s) in series.iter().enumerate() {
        let c = colours[i % colours.len()];
        out.push_str(&format!(
            "\\addplot[{c}, thick, mark=none, forget plot] table[x=x, y=log10_p] {{{}.dat}};\n",
            s.tag
        ));
    }
    for s in series.iter().filter(|s| s.saturated()) {
        let i = series.iter().position(|o| o.tag == s.tag).unwrap_or(0);
        let c = colours[i % colours.len()];
        let m = marks[i % marks.len()];
        out.push_str(&format!(
            "% SATURATED: p underflowed to 0 (log10 p = -inf). Drawn ON A FLOOR, off the\n% scale, NEVER as a finite value --- and it is not success: see the caption.\n\\addplot[{c}, only marks, mark={m}, mark size=1.1pt, opacity=0.7] table[x=x, y=sat] {{{}.dat}};\n\\addlegendentry{{{}: $p=0$ ($\\log_{{10}} p = -\\infty$, f32 saturated --- OFF SCALE, not a value)}}\n",
            s.tag,
            tex_escape(&s.tag)
        ));
    }
    for v in bars_p.values() {
        out.push_str(&format!(
            "\\draw[dashed, gray!70] ({{rel axis cs:0,0}}|-{{axis cs:1,{v}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:1,{v}}})\n  node[pos=0.13, above, font=\\scriptsize, gray!70] {{stop bar $\\log_{{10}} p \\le {v:.0}$}};\n"
        ));
    }

    // ---- panel 2: 1 - R^2, the other half of the bar --------------------
    out.push_str("\n% Panel 2 --- 1-R^2, THE OTHER HALF OF THE STOP BAR. A fit is a law only when\n");
    out.push_str("% BOTH halves pass, and each has failed alone on real runs: an angle can\n");
    out.push_str("% saturate at 1-R^2 = 2.6e-6 (no law), and 1-R^2 = 2.6e-14 can still fail p\n");
    out.push_str("% because the third block holds the angle open. Hence two panels, not one.\n");
    // THE KEY FOR BOTH PANELS lives here, under the figure, with the full label
    // off the run card. A SOLID line is train; a DOTTED line of the same colour
    // is that run's third block.
    //
    // `ymin` IS FORCED DOWN TO THE BAR, and this is not cosmetic. pgfplots
    // auto-ranges to the DATA and clips everything outside the axis, so a bar of
    // 1e-10 under traces that bottom out at 5e-8 is drawn and then thrown away:
    // the panel showed six traces converging with nothing on the page saying
    // they were three decades short of the thing they had to reach. The bar is
    // the reason this panel exists, so the panel is sized to include it. It
    // compresses the traces, and that compression IS the finding.
    let floor_r2 = bars_r2.values().copied().fold(f64::INFINITY, f64::min);
    let ymin = if floor_r2.is_finite() { format!(", ymin={:e}", floor_r2 / 5.0) } else { String::new() };
    out.push_str(&format!("\\nextgroupplot[ylabel={{$1-R^2$}}, ymode=log{ymin}, legend columns=2]\n"));
    for (i, s) in series.iter().enumerate() {
        let c = colours[i % colours.len()];
        out.push_str(&format!(
            "\\addplot[{c}, thick, mark=none] table[x=x, y=one_minus_r2_train] {{{}.dat}};\n\\addlegendentry{{{}}}\n",
            s.tag,
            tex_escape(&s.label)
        ));
        if s.points.iter().any(|p| p.one_minus_r2_third.is_some()) {
            out.push_str(&format!(
                "% The third block (SMOGD/SMOTE synthetic rows). It enters the HFF angle like\n% any other objective, so while it sits at ~7e-3 the angle cannot close and p\n% plateaus however far train falls --- which is the whole reading of this figure.\n\\addplot[{c}, densely dotted, thick, mark=none, forget plot] table[x=x, y=one_minus_r2_third] {{{}.dat}};\n",
                s.tag
            ));
        }
    }
    // One legend entry explaining the dotted style, drawn from no data.
    if series.iter().any(|s| s.points.iter().any(|p| p.one_minus_r2_third.is_some())) {
        out.push_str("\\addlegendimage{densely dotted, thick, gray!60!black}\n\\addlegendentry{(dotted, same colour) that run's third block}\n");
    }
    for v in bars_r2.values() {
        out.push_str(&format!(
            "\\draw[dashed, gray!70] ({{rel axis cs:0,0}}|-{{axis cs:1,{v:e}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:1,{v:e}}})\n  node[pos=0.13, above, font=\\scriptsize, gray!70] {{stop bar $1-R^2 \\le {}$}};\n",
            tex_scientific(*v)
        ));
    }
    out.push_str("\n\\end{groupplot}\n\\end{tikzpicture}\n");
    out.push_str(&format!("\\caption{{{}}}\n\\label{{{label}}}\n\\end{{figure}}\n", tex_escape_caption(caption)));
    out
}

/// The lines a document's preamble needs before it can `\input` the figure.
pub const PREAMBLE: &str = "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\usepgfplotslibrary{groupplots}\n";

/// A standalone document around the figure, so the chart compiles on its own
/// without touching the paper — which matters when somebody else is editing it.
pub fn standalone(figure_file: &str) -> String {
    format!(
        "% A wrapper so the figure compiles by itself: `pdflatex standalone.tex`.\n\\documentclass[11pt,a4paper]{{article}}\n\\usepackage[margin=1in]{{geometry}}\n\\usepackage{{amsmath}}\n{PREAMBLE}\\pagestyle{{empty}}\n\\begin{{document}}\n\\input{{{figure_file}}}\n\\end{{document}}\n"
    )
}

/// A number as MATHS, not as Rust's `{:e}`. `format!("{:e}", 1e-10)` is the
/// string `1e-10`, which TeX sets as the letter e between two numbers with a
/// minus sign — "1e − 10". A bar written that way is a bar nobody can read.
///
/// Returns the body of a maths expression (the caller supplies the `$…$`):
/// `10^{-10}` when the mantissa is 1, `2.5\times 10^{-3}` otherwise.
fn tex_scientific(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let exponent = v.abs().log10().floor() as i32;
    let mantissa = v / 10f64.powi(exponent);
    let sign = if v < 0.0 { "-" } else { "" };
    // Within a rounding of 1, the mantissa is not worth printing: 1e-10 is
    // `10^{-10}`, not `1\times 10^{-10}`.
    if (mantissa.abs() - 1.0).abs() < 1e-9 {
        format!("{sign}10^{{{exponent}}}")
    } else {
        format!("{:.3}\\times 10^{{{exponent}}}", mantissa)
    }
}

/// The characters a legend entry or a tag must not hand to TeX raw. Tags carry
/// underscores (`bacres1_75long`) and an unescaped one is a subscript, which
/// silently changes what the legend says.
fn tex_escape(s: &str) -> String {
    s.chars().fold(String::new(), |mut out, c| {
        match c {
            '_' | '%' | '$' | '&' | '#' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            '^' => out.push_str("\\textasciicircum{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '\\' => out.push_str("\\textbackslash{}"),
            _ => out.push(c),
        }
        out
    })
}

/// A caption is written with maths in it on purpose: `$`, `\`, `{}` and the `_`
/// of a subscript are the author's and must reach TeX untouched — escaping
/// `\log_{10} p` turns it into the literal `log_10p`, which is how the first
/// compile of this figure printed it.
///
/// So only the characters that are never deliberate in prose are escaped, and
/// only OUTSIDE maths: `%` would comment the rest of the caption away, and `&`
/// and `#` are errors in a caption. Inside `$…$` nothing is touched at all.
fn tex_escape_caption(s: &str) -> String {
    let mut maths = false;
    let mut escaped_backslash = false;
    s.chars().fold(String::new(), |mut out, c| {
        // A `\$` is a literal dollar and does not open maths.
        if c == '$' && !escaped_backslash {
            maths = !maths;
        }
        match c {
            '%' | '&' | '#' if !maths => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
        escaped_backslash = c == '\\' && !escaped_backslash;
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A REAL RECORDING OF THE RUN WHOSE ANGLE SATURATED — trimmed from
    /// `bacres1_75long`, the 800+400 fit that ran 65,264 generations, reported
    /// `log10 p -inf` and did NOT find the law: its train 1-R² sat at 2.6e-6,
    /// four orders short of the 1e-10 bar. It is the fixture that matters
    /// because it is the case a p-only chart reads as a triumph.
    const SATURATED: &str = include_str!("../../tests/fixtures/telemetry_v1_saturated.jsonl");

    /// The ordinary fixture: a fit whose p stayed finite throughout.
    const MEASURED: &str = include_str!("../../tests/fixtures/telemetry_v1.jsonl");

    /// SATURATION IS NOT A MISSING NUMBER. A null `log10_p` with `best_hff` of
    /// exactly 0.0 is the f32 angle on the pole; a null `log10_p` with no
    /// `best_hff` is a generation that did not score. The chart must tell them
    /// apart, because one of them is a marker on the floor and the other is a
    /// gap.
    #[test]
    fn a_saturated_angle_is_classified_and_never_given_a_number() {
        let s = series("bacres1_75long", SATURATED, None, Thinning::default()).expect("a series");
        assert!(s.saturated(), "the recording is of a run that saturated");
        let saturated: Vec<&Point> = s.points.iter().filter(|p| p.kind == PointKind::Saturated).collect();
        assert!(!saturated.is_empty());
        for p in &saturated {
            assert_eq!(p.log10_p, None, "a saturated point was given a finite p: {p:?}");
        }
        // And it is NOT success: the error half of the bar was nowhere near met.
        let last = s.points.last().expect("a last point");
        let err = last.one_minus_r2_train.expect("a train R²");
        assert!(err > 1e-10, "the fixture is the run that saturated WITHOUT finding the law: 1-R² = {err:e}");
        assert_eq!(last.meets_bar(s.stop_log10_p, s.stop_one_minus_r2), Some(false), "p = 0 alone must not read as a law");
        assert!(s.first_met_bar().is_none(), "no beat of this run met both halves");
    }

    /// THE SATURATED VALUE NEVER REACHES THE `log10_p` COLUMN. It is `nan` there
    /// — which pgfplots breaks the line at — and its floor is in its own `sat`
    /// column, so a marker off the scale can never be read as a value the run
    /// held.
    #[test]
    fn the_table_writes_nan_for_a_saturated_p_and_puts_its_floor_in_its_own_column() {
        let s = series("bacres1_75long", SATURATED, None, Thinning::default()).expect("a series");
        let text = table(&s, XAxis::Generation);
        let header: Vec<&str> = text.lines().next().expect("a header").split_whitespace().collect();
        let (p_at, sat_at) = (
            header.iter().position(|c| *c == "log10_p").expect("a log10_p column"),
            header.iter().position(|c| *c == "sat").expect("a sat column"),
        );
        let floor = s.saturation_floor();
        let mut saturated_rows = 0;
        for (line, point) in text.lines().skip(1).zip(&s.points) {
            let cells: Vec<&str> = line.split_whitespace().collect();
            if point.kind == PointKind::Saturated {
                saturated_rows += 1;
                assert_eq!(cells[p_at], "nan", "a saturated p was written as a number: {line}");
                let drawn: f64 = cells[sat_at].parse().expect("a floor");
                assert!((drawn - floor).abs() < 1e-6, "{drawn} is not the floor {floor}");
                assert!(drawn < s.stop_log10_p.unwrap_or(0.0), "the floor must be BELOW the bar, visibly off scale");
            } else {
                assert_eq!(cells[sat_at], "nan", "a measured point was given a saturation marker: {line}");
            }
        }
        assert!(saturated_rows > 0, "the fixture's saturated rows reached the table");
    }

    /// THINNING KEEPS THE STAIRCASE. The first and last points survive, every
    /// change of kind survives (so the entry into saturation is never lost), and
    /// no emitted point holds a value the run did not.
    #[test]
    fn thinning_keeps_the_ends_the_steps_and_invents_nothing() {
        let (records, _) = parse_stream(SATURATED);
        let all = series("t", SATURATED, None, Thinning { log10_p_step: 0.0, r2_relative_step: 0.0, max_gap: 1 })
            .expect("an unthinned series");
        let thinned = series("t", SATURATED, None, Thinning::default()).expect("a thinned series");
        assert_eq!(all.points.len(), all.snapshots, "a zero threshold keeps every snapshot");
        assert!(thinned.points.len() <= all.points.len());
        assert_eq!(thinned.points.first(), all.points.first(), "the first point must survive");
        assert_eq!(thinned.points.last(), all.points.last(), "the last point must survive");
        // Every point drawn is a point the run had — nothing is averaged.
        for p in &thinned.points {
            assert!(all.points.contains(p), "thinning invented a point: {p:?}");
        }
        // The transition into saturation is kept, whatever the thresholds.
        let kinds: Vec<PointKind> = all.points.iter().map(|p| p.kind).collect();
        if let Some(at) = kinds.windows(2).position(|w| w[0] != w[1]) {
            let changed = all.points[at + 1];
            assert!(thinned.points.contains(&changed), "a change of kind was thinned away: {changed:?}");
        }
        assert!(records.len() > 2);
    }

    /// A CHART MUST NOT OWN A THRESHOLD THE ENGINE OWNS. The bar is read off the
    /// stream's own `run_start`, and a stream written before the field existed
    /// gets NO rule rather than a supplied one.
    #[test]
    fn the_stop_bar_comes_from_the_stream_and_is_never_supplied() {
        let old = series("old", MEASURED, None, Thinning::default()).expect("a series");
        assert_eq!((old.stop_log10_p, old.stop_one_minus_r2), (None, None), "a stream with no bar must not acquire one");
        assert_eq!(old.points[0].meets_bar(None, None), None, "with no bar there is no verdict");
        let without = figure(&[old], XAxis::Generation, "c", "fig:x");
        assert!(!without.contains("stop bar"), "a rule was drawn for a run that carried no bar");

        let bar = series("bar", SATURATED, None, Thinning::default()).expect("a series");
        assert_eq!(bar.stop_log10_p, Some(-19.0));
        assert_eq!(bar.stop_one_minus_r2, Some(1e-10));
        let with = figure(&[bar], XAxis::Generation, "c", "fig:x");
        // BOTH halves, and on their OWN panels. A single `contains("stop bar")`
        // passed while the 1-R² rule was being CLIPPED AWAY by the axis: the
        // figure showed six traces converging with nothing saying they were
        // three decades short of the bar. So each panel is checked separately.
        let (top, bottom) = with.split_once("ylabel={$1-R^2$}").expect("the two panels");
        assert!(top.contains("stop bar $\\log_{10} p"), "the p half was not drawn on the p panel");
        assert!(bottom.contains("stop bar $1-R^2"), "the error half was not drawn on the error panel");
        // AND THE ERROR PANEL IS SIZED TO SHOW IT. pgfplots auto-ranges to the
        // data and clips outside the axis, so a rule below every trace is drawn
        // and then thrown away unless `ymin` reaches down to it.
        let ymin: f64 = bottom
            .lines()
            .next()
            .and_then(|l| l.split("ymin=").nth(1))
            .and_then(|r| r.split([',', ']']).next())
            .expect("an ymin on the error panel")
            .parse()
            .expect("a number");
        assert!(ymin < 1e-10, "ymin {ymin:e} does not reach the bar, so the rule is clipped away");
        // The bar reads as maths, not as Rust's `{:e}`: `1e-10` sets as
        // "1e - 10", which is not a number anybody can read.
        assert!(bottom.contains("10^{-10}"), "the bar was written in Rust's exponent form");
        assert!(!bottom.contains("\\le 1e-10"), "a raw `1e-10` reached the page");
    }

    /// A NUMBER ON THE PAGE IS MATHS. Rust's `{:e}` is a debugging format and
    /// TeX sets it as a letter between two numbers.
    #[test]
    fn an_exponent_reaches_the_page_as_maths() {
        assert_eq!(tex_scientific(1e-10), "10^{-10}");
        assert_eq!(tex_scientific(1e-19), "10^{-19}");
        assert_eq!(tex_scientific(1.0), "10^{0}");
        assert!(tex_scientific(2.5e-3).starts_with("2.500\\times 10^{-3}"), "{}", tex_scientific(2.5e-3));
        assert!(!tex_scientific(1e-10).contains('e'), "the letter e reached the maths");
    }

    /// THE LEGEND TELLS THE TRUTH ABOUT THE OBJECTIVE COUNT, which means it says
    /// nothing when there is no card: the count is a function of five settings
    /// the stream does not carry, so it cannot be inferred from the third block
    /// alone.
    #[test]
    fn the_label_names_the_objectives_only_when_a_card_says_so() {
        let bare = series("run", SATURATED, None, Thinning::default()).expect("a series");
        assert!(!bare.label.contains("objectives"), "an objective count was guessed: {}", bare.label);
        assert!(bare.label.contains("run"), "{}", bare.label);

        let card = r#"{"derived":{"hff_objectives":9,"population":1200}}"#;
        let carded = series("run", SATURATED, Some(card), Thinning::default()).expect("a series");
        assert!(carded.label.contains("9 objectives"), "the card's count did not reach the legend: {}", carded.label);
    }

    /// AN UNDERSCORE IN A TAG IS NOT A SUBSCRIPT. Every run in this project is
    /// named `bacres1_75long`, and an unescaped underscore silently rewrites the
    /// legend — or fails the compile outside maths mode.
    #[test]
    fn a_tag_reaches_the_legend_escaped() {
        let s = series("bacres1_75long", SATURATED, None, Thinning::default()).expect("a series");
        let drawn = figure(std::slice::from_ref(&s), XAxis::Generation, "100% of it", "fig:conv");
        assert!(drawn.contains("bacres1\\_75long"), "an underscore went to TeX raw");
        assert!(!drawn.contains("{bacres1_75long}"), "an unescaped tag reached a legend entry");
        assert!(drawn.contains("100\\%"), "a caption's percent sign went raw");
        // The data file it points at is the tag's own, unescaped — a filename is
        // not TeX.
        assert!(drawn.contains("{bacres1_75long.dat}"), "the table reference was escaped and will not be found");

        // A CAPTION'S MATHS IS THE AUTHOR'S. Escaping the `_` of a subscript
        // turned `\log_{10} p` into the literal `log_10p` on the first compile
        // of this figure — a caption that says something else than it was given.
        let maths = figure(&[s], XAxis::Generation, "$\\log_{10} p$ at 100% of it", "fig:x");
        assert!(maths.contains("$\\log_{10} p$"), "a subscript in the caption's maths was escaped away");
        assert!(maths.contains("100\\%"), "a percent sign outside maths must still be escaped");
    }

    /// BOTH HALVES OF THE BAR, OR IT IS NOT A LAW. A p under the bar with an
    /// error over it is not a pass, and neither is the reverse.
    #[test]
    fn a_verdict_needs_both_halves() {
        let point = |p: Option<f64>, kind: PointKind, train: f64| Point {
            generation: 1,
            elapsed_ms: 1,
            log10_p: p,
            kind,
            one_minus_r2_train: Some(train),
            one_minus_r2_val: Some(train),
            one_minus_r2_third: None,
        };
        let (bar_p, bar_r2) = (Some(-19.0), Some(1e-10));
        // The `bacres1_75long` shape: p saturated, error four orders short.
        assert_eq!(point(None, PointKind::Saturated, 2.6e-6).meets_bar(bar_p, bar_r2), Some(false));
        // The `bacres1_20k` shape: error four orders PAST the bar, p still over it.
        assert_eq!(point(Some(-13.65), PointKind::Measured, 2.6e-14).meets_bar(bar_p, bar_r2), Some(false));
        // Both halves met.
        assert_eq!(point(Some(-29.24), PointKind::Measured, 1e-14).meets_bar(bar_p, bar_r2), Some(true));
        // A saturated angle WITH the error met is a pass: p = 0 is under any bar.
        assert_eq!(point(None, PointKind::Saturated, 1e-14).meets_bar(bar_p, bar_r2), Some(true));
    }

    /// The figure is a fragment, not a document: the paper owns the preamble, and
    /// the standalone wrapper is what compiles it alone.
    #[test]
    fn the_figure_is_a_fragment_and_the_wrapper_is_the_document() {
        let s = series("t", SATURATED, None, Thinning::default()).expect("a series");
        let figure = figure(&[s], XAxis::ElapsedMs, "caption", "fig:conv");
        assert!(!figure.contains("\\documentclass"), "the fragment must not open a document");
        assert!(!figure.contains("\\usepackage{pgfplots}"), "the fragment must not load a package the paper loads");
        assert!(figure.contains("\\begin{figure}") && figure.contains("\\label{fig:conv}"));
        assert!(figure.contains("elapsed (ms)"), "the x-axis choice did not reach the figure");
        let doc = standalone("convergence.tex");
        assert!(doc.contains("\\documentclass") && doc.contains("\\input{convergence.tex}"));
        assert!(doc.contains("pgfplots"), "the wrapper must supply what the fragment needs");
    }

    /// A stream with no snapshots in it is an error with the tag in it, not a
    /// silent empty figure.
    #[test]
    fn a_stream_with_no_snapshots_says_so() {
        let e = series("empty", "", None, Thinning::default()).expect_err("an empty stream is not a series");
        assert!(e.contains("empty") && e.contains("no snapshots"), "{e}");
    }
}
