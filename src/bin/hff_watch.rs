//! `hff-watch` — an htop for a symbolic-regression fit.
//!
//! ```text
//! hff-watch --file   <path>   replay a recorded telemetry stream
//! hff-watch --follow <path>   tail a live one
//! ```
//!
//! It repaints the current state of a fit. It does not make an operator read an
//! ever-growing log, and it never sends anything to the search: a fit runs
//! exactly the same whether this is open, closed, resized or killed. The only
//! thing it touches is a file the engine appends to.
//!
//! **One parser, one state machine.** `--file` and `--follow` differ in exactly
//! one place — where the records come from — and meet immediately in
//! [`fuller::evolve::watch::WatchState`]. Everything that decides what is on
//! screen lives there, tested without a terminal; this file is the terminal.
//!
//! **HFF is lower-is-better**, in every bar, sort, colour and sparkline, and
//! improvement is drawn in LOG SPACE. A fit goes from 1e-1 to 1e-5; a linear bar
//! of that is empty and then full, with the whole run in the gap between.
//!
//! Keys: `Tab`/`Shift-Tab` island · `j`/`k`/`↑`/`↓` cohort · `s` sort ·
//! `/` filter · `g` global cohorts · `Enter` detail · `m` model · `Space` pause
//! the REDRAW ONLY · `q` quit the viewer (never the fit).

use std::io::Write;

use fuller::evolve::telemetry::{parse_stream, Tailer};
use fuller::evolve::watch::{gain_text, or_dash, Gain, Liveness, WatchState};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Clear, LineGauge, Paragraph, Row, Table, Wrap};
use ratatui::Frame;

/// The screen's redraw rate: the brief's 2–4 Hz. It is also how long a keypress
/// can wait, which is well inside what a hand notices, and it is what keeps the
/// viewer from spinning a core on a stream that is quiet.
const TICK: std::time::Duration = std::time::Duration::from_millis(250);

/// Below this the full screen does not fit and the 80x24 fallback is drawn.
const WIDE: (u16, u16) = (120, 35);
/// Below THIS there is no useful layout at all and the compact warning is drawn.
const MINIMUM: (u16, u16) = (70, 20);

/// The `m` overlay's tabs. The SIMPLIFIED tab is deliberately not a computation:
/// the fit's own final form costs a saturation and belongs at the end of a fit,
/// not on a viewer's keystroke. It says so rather than showing a form this
/// viewer invented.
const MODEL_TABS: [&str; 3] = ["protected", "plain", "simplified"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.iter().position(|a| a == "--file" || a == "--follow");
    let Some(at) = mode else {
        eprintln!("usage: hff-watch --file <path> | --follow <path>");
        eprintln!("  --file    replay a recorded telemetry stream");
        eprintln!("  --follow  tail a live one (the fit is never touched)");
        std::process::exit(2);
    };
    let following = args[at] == "--follow";
    let Some(path) = args.get(at + 1).cloned() else {
        eprintln!("{} needs a path", args[at]);
        std::process::exit(2);
    };
    let mut state = WatchState::new(following);
    // A REPLAY is read whole, up front: it is a recording, there is nothing to
    // wait for, and reading it in one go means the first frame is the last one
    // rather than an empty screen that fills in.
    let mut tailer = if following {
        Some(Tailer::new(&path))
    } else {
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let (records, bad) = parse_stream(&text);
                state.bad_lines = bad;
                for r in records {
                    state.apply_record(r);
                }
                None
            }
            Err(e) => {
                eprintln!("hff-watch: {path}: {e}");
                std::process::exit(1);
            }
        }
    };
    let mut ui = Ui::default();
    let mut terminal = ratatui::init();
    let outcome = loop {
        if let Some(t) = tailer.as_mut() {
            for parsed in t.poll() {
                state.apply(parsed);
            }
            state.rotations = t.rotations;
            state.bad_lines = t.bad_lines;
        }
        // PAUSE stops the REDRAW and nothing else: records keep arriving above,
        // so unpausing shows the present rather than replaying a backlog.
        if !state.paused {
            if let Err(e) = terminal.draw(|f| draw(f, &state, &ui)) {
                break Err(e);
            }
        }
        match event::poll(TICK) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                        break Ok(());
                    }
                    if press(&mut state, &mut ui, key.code, key.modifiers) {
                        break Ok(());
                    }
                }
                Ok(_) => {}
                Err(e) => break Err(e),
            },
            Ok(false) => {}
            Err(e) => break Err(e),
        }
    };
    ratatui::restore();
    if let Err(e) = outcome {
        eprintln!("hff-watch: {e}");
        std::process::exit(1);
    }
    // What was copied out, said AFTER the alternate screen is gone so the path
    // is still on the terminal when the viewer is.
    if let Some(path) = ui.copied {
        println!("model written to {path}");
    }
}

/// The bits of screen state that are the TERMINAL's, not the run's: which
/// overlay is open and where the panes are scrolled. They belong here rather
/// than in `WatchState` because they say nothing about the fit.
#[derive(Default)]
struct Ui {
    model_open: bool,
    model_tab: usize,
    /// `Enter`: the detail pane is expanded over the table on a narrow screen.
    detail_open: bool,
    /// Horizontal scroll in the model overlay, for a model too wide to wrap
    /// usefully. `w` toggles wrapping.
    model_scroll: u16,
    model_wrap: bool,
    copied: Option<String>,
}

/// One keypress. Returns true to quit — which quits the VIEWER. The fit is a
/// different process and has never heard of this one.
fn press(state: &mut WatchState, ui: &mut Ui, code: KeyCode, modifiers: KeyModifiers) -> bool {
    // The filter takes every printable key while it is open, so `q` and `s` type
    // rather than act.
    if state.filtering {
        match code {
            KeyCode::Esc => {
                state.filter.clear();
                state.filtering = false;
            }
            KeyCode::Enter => state.filtering = false,
            KeyCode::Backspace => {
                state.filter.pop();
            }
            KeyCode::Char(c) => state.filter.push(c),
            _ => {}
        }
        return false;
    }
    if ui.model_open {
        match code {
            KeyCode::Esc | KeyCode::Char('m') | KeyCode::Char('q') => ui.model_open = false,
            KeyCode::Tab | KeyCode::Right => ui.model_tab = (ui.model_tab + 1) % MODEL_TABS.len(),
            KeyCode::Left => ui.model_tab = (ui.model_tab + MODEL_TABS.len() - 1) % MODEL_TABS.len(),
            KeyCode::Char('w') => ui.model_wrap = !ui.model_wrap,
            KeyCode::Char('h') => ui.model_scroll = ui.model_scroll.saturating_sub(8),
            KeyCode::Char('l') => ui.model_scroll = ui.model_scroll.saturating_add(8),
            // COPY-TO-FILE: the brief asks for it, and a file is the honest form
            // of it — a terminal has no clipboard a program can rely on, and
            // pretending otherwise is a key that silently does nothing.
            KeyCode::Char('y') => ui.copied = copy_model(state),
            _ => {}
        }
        return false;
    }
    match code {
        KeyCode::Char('q') | KeyCode::Esc => return true,
        KeyCode::Char('j') | KeyCode::Down => state.move_selection(true),
        KeyCode::Char('k') | KeyCode::Up => state.move_selection(false),
        KeyCode::Char('s') => state.sort = state.sort.next(),
        KeyCode::Char('/') => {
            state.filtering = true;
            state.filter.clear();
        }
        // `g`: back to the GLOBAL cohorts — the table off an island's own split,
        // the filter cleared. The "show me everything" key.
        KeyCode::Char('g') => {
            state.filter.clear();
            state.island_focus = false;
        }
        KeyCode::Char(' ') => state.paused = !state.paused,
        KeyCode::Char('m') => ui.model_open = true,
        KeyCode::Enter => ui.detail_open = !ui.detail_open,
        // Tab moves the island AND focuses the table on it: the brief's
        // "selection focuses an island". `g` puts it back to the global totals.
        KeyCode::Tab => {
            state.move_island(!modifiers.contains(KeyModifiers::SHIFT));
            state.island_focus = true;
        }
        KeyCode::BackTab => {
            state.move_island(false);
            state.island_focus = true;
        }
        _ => {}
    }
    false
}

/// Write the model out, all three forms, and return where it went.
fn copy_model(state: &WatchState) -> Option<String> {
    let m = state.model.as_ref()?;
    let run = state.start.as_ref().map_or("run", |s| s.header.run_id.as_str());
    let path = std::env::temp_dir().join(format!("hff-watch-{run}-gen{}.txt", m.found_generation));
    let mut file = std::fs::File::create(&path).ok()?;
    writeln!(file, "# {run}, found at generation {}, HFF {}", m.found_generation, or_dash(m.hff, 6)).ok()?;
    writeln!(file, "\n## protected (executable: what the chromosome computes)\n{}", m.infix_protected).ok()?;
    writeln!(file, "\n## plain (symbolic: the protected operators written as ordinary ones)\n{}", m.infix_plain).ok()?;
    writeln!(file, "\n## raw Math\n{}", m.raw_math).ok()?;
    Some(path.to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------------
// The screen
// ---------------------------------------------------------------------------

fn draw(f: &mut Frame, state: &WatchState, ui: &Ui) {
    let area = f.area();
    if area.width < MINIMUM.0 || area.height < MINIMUM.1 {
        compact(f, state, area);
        return;
    }
    // The wide screen is the brief's region table. Narrower than that and the
    // secondary columns and the detail pane go, which is the 80x24 fallback.
    let wide = area.width >= WIDE.0 && area.height >= WIDE.1;
    let rows = if wide {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // header
                Constraint::Length(3),  // global strip
                Constraint::Length(8),  // island cards
                Constraint::Min(8),     // cohort table + detail
                Constraint::Length(4),  // events
                Constraint::Length(1),  // footer
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(3),
                Constraint::Length(5), // ONE island, the selected one
                Constraint::Min(5),
                Constraint::Length(1), // one-line events
                Constraint::Length(1),
            ])
            .split(area)
    };
    header(f, state, rows[0]);
    global_strip(f, state, rows[1]);
    if wide {
        island_cards(f, state, rows[2]);
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(60), Constraint::Length(38)])
            .split(rows[3]);
        // `Enter` expands the detail over the table, which is how a long model
        // summary is read on a screen that is wide but not tall.
        if ui.detail_open {
            detail(f, state, rows[3]);
        } else {
            cohort_table(f, state, split[0], true);
            detail(f, state, split[1]);
        }
        events(f, state, rows[4]);
    } else {
        one_island(f, state, rows[2]);
        if ui.detail_open {
            detail(f, state, rows[3]);
        } else {
            cohort_table(f, state, rows[3], false);
        }
        one_line_events(f, state, rows[4]);
    }
    footer(f, state, rows[5]);
    if ui.model_open {
        model_overlay(f, state, ui, area);
    }
}

/// Below 70x20: the essential metrics and a warning, and nothing that would be
/// unreadable at this size. It is still a useful screen — the global best, the
/// generation and the clock — not an error message.
fn compact(f: &mut Frame, state: &WatchState, area: Rect) {
    let s = state.snapshot.as_ref();
    let generation = s.map_or(0, |s| s.header.generation);
    let mut lines = vec![
        Line::from(Span::styled("hff-watch · terminal too small", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
        Line::from(format!("{}x{} — the full screen needs {}x{}", area.width, area.height, WIDE.0, WIDE.1)),
        Line::from(""),
        Line::from(format!("gen {generation}  best HFF {}", or_dash(s.and_then(|s| s.global.best_hff), 4))),
        Line::from(format!("{}  {}", badge_text(state), or_dash(s.map(|s| s.header.elapsed_ms as f64 / 1000.0), 0))),
    ];
    lines.push(Line::from(Span::styled("q to quit", Style::default().fg(Color::DarkGray))));
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
}

/// The badge that says what the run is doing. FINISHED, STALE and LIVE are three
/// visibly different things, and PAUSED says the SCREEN is paused — never the fit.
fn badge_text(state: &WatchState) -> String {
    let base = match state.liveness() {
        Liveness::Finished => match state.end.as_ref() {
            Some(e) => format!("FINISHED · {}", e.stopped_by),
            None => "FINISHED".to_string(),
        },
        Liveness::Stale => "STALE · no telemetry".to_string(),
        Liveness::Live if state.following => "LIVE".to_string(),
        Liveness::Live => "REPLAY".to_string(),
    };
    if state.paused {
        return format!("{base} · SCREEN PAUSED (the fit is not)");
    }
    base
}

fn badge_colour(state: &WatchState) -> Color {
    match state.liveness() {
        Liveness::Finished => Color::Green,
        Liveness::Stale => Color::Red,
        Liveness::Live => Color::Cyan,
    }
}

fn header(f: &mut Frame, state: &WatchState, area: Rect) {
    let start = state.start.as_ref();
    let s = state.snapshot.as_ref();
    let generation = s.map_or(0, |s| s.header.generation);
    let elapsed = s.map_or(0, |s| s.header.elapsed_ms) as f64 / 1000.0;
    let budget = s.map_or(0, |s| s.budget_ms) as f64 / 1000.0;
    // THE AGE OF THE DATA, which the brief asks for by name: how long ago the
    // frame on screen arrived. A replay's frame is its recording's and has no
    // age, so it says so rather than counting up from when the viewer started.
    let age = match (state.following, state.last_record) {
        (true, Some(t)) => format!("{:.0}s ago", t.elapsed().as_secs_f64()),
        (true, None) => "no telemetry yet".to_string(),
        (false, _) => "recorded".to_string(),
    };
    let one = Line::from(vec![
        Span::styled("◉ HFF-SR / WATCH ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", badge_text(state)), Style::default().fg(badge_colour(state)).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(start.map_or_else(|| "(no run_start)".to_string(), |s| s.dataset.clone())),
        Span::styled(" · seed ", Style::default().fg(Color::DarkGray)),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| s.seed.to_string())),
        Span::styled("  run ", Style::default().fg(Color::DarkGray)),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| s.header.run_id.clone())),
    ]);
    let two = Line::from(vec![
        Span::styled("gen ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{generation}"), Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!("  {elapsed:.0}s / {budget:.0}s")),
        Span::styled("  sampled ", Style::default().fg(Color::DarkGray)),
        Span::raw(age),
        Span::styled("  train/val/third ", Style::default().fg(Color::DarkGray)),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| format!("{}/{}/{}", s.n_train, s.n_val, s.n_extrap))),
        // A viewer that is silently skipping input is lying about what it shows.
        Span::styled(
            if state.bad_lines > 0 || state.rotations > 0 {
                format!("  {} bad lines, {} rotations", state.bad_lines, state.rotations)
            } else {
                String::new()
            },
            Style::default().fg(Color::Yellow),
        ),
    ]);
    f.render_widget(Paragraph::new(vec![one, two]), area);
}

fn global_strip(f: &mut Frame, state: &WatchState, area: Rect) {
    let s = state.snapshot.as_ref();
    let g = s.map(|s| &s.global);
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)])
        .split(area);
    let gain = state.global_gain();
    let best = Paragraph::new(vec![
        Line::from(Span::styled("GLOBAL BEST HFF ↓ (lower is better)", Style::default().fg(Color::DarkGray))),
        Line::from(vec![
            Span::styled(or_dash(g.and_then(|g| g.best_hff), 6), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(gain_text(gain), Style::default().fg(gain_colour(gain))),
        ]),
        // BEST-EVER, separately: a current best is not a record, and the brief
        // asks for the two not to be confused.
        Line::from(Span::styled(
            format!("best ever {}   {} unscored rows", or_dash(g.and_then(|g| g.best_ever_hff), 4), g.map_or(0, |g| g.nan_rows)),
            Style::default().fg(Color::DarkGray),
        )),
    ]);
    f.render_widget(best, columns[0]);
    let quality = Paragraph::new(vec![
        Line::from(Span::styled("ERROR", Style::default().fg(Color::DarkGray))),
        Line::from(format!(
            "train 1-R² {}   val 1-R² {}",
            or_dash(g.and_then(|g| g.r2_train).map(|r| 1.0 - r), 2),
            or_dash(g.and_then(|g| g.r2_val).map(|r| 1.0 - r), 2)
        )),
        Line::from(Span::styled(
            format!(
                "mse {}  log10 p {}  depth {}  head {}",
                or_dash(g.and_then(|g| g.mse_train), 2),
                g.and_then(|g| g.log10_p).map_or_else(|| "—".to_string(), |p| format!("{p:.2}")),
                g.map_or(0, |g| g.t_depth),
                g.map_or(0, |g| g.vhead)
            ),
            Style::default().fg(Color::DarkGray),
        )),
    ]);
    f.render_widget(quality, columns[1]);
    // THE BUDGET BAR is a bar of TIME, which is the only thing on this screen
    // that is genuinely a fraction of a known total. HFF is never drawn as a
    // percentage: it has no ceiling to be a percentage of.
    let (elapsed, budget) = (s.map_or(0, |s| s.header.elapsed_ms), s.map_or(0, |s| s.budget_ms).max(1));
    let fraction = (elapsed as f64 / budget as f64).clamp(0.0, 1.0);
    let bar = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
        .split(columns[2]);
    f.render_widget(Paragraph::new(Span::styled("BUDGET", Style::default().fg(Color::DarkGray))), bar[0]);
    // A THIN TRACK with the number beside it, not a block bar with the label
    // buried in it: at 40 columns a filled `Gauge` reads as a wall of blocks
    // with a percentage somewhere inside, which is not a reading.
    f.render_widget(
        LineGauge::default().filled_style(Style::default().fg(Color::Cyan)).ratio(fraction).label(format!("{:.0}%", fraction * 100.0)),
        bar[1],
    );
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("{:.0}s left", (budget.saturating_sub(elapsed)) as f64 / 1000.0),
            Style::default().fg(Color::DarkGray),
        )),
        bar[2],
    );
}

fn gain_colour(g: Gain) -> Color {
    match g {
        // A FALLING HFF is an improvement, and it is cyan. This is the one place
        // a sign could be read backwards, so it is written out: positive gain =
        // HFF fell = better.
        Gain::Percent(v) if v > 0.0 => Color::Cyan,
        Gain::Percent(v) if v < 0.0 => Color::Red,
        Gain::Percent(_) => Color::Gray,
        Gain::New | Gain::Merged => Color::Yellow,
        Gain::Extinct => Color::Red,
        Gain::Unscored => Color::DarkGray,
    }
}

fn island_cards(f: &mut Frame, state: &WatchState, area: Rect) {
    let islands = state.islands();
    if islands.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("no island data yet", Style::default().fg(Color::DarkGray)))
                .block(Block::default().borders(Borders::ALL).title(" ISLANDS ")),
            area,
        );
        return;
    }
    // More than two islands: a window of them around the selected one, so `Tab`
    // walks a long list and the focused card is always on screen.
    let per_screen = ((area.width / 40).max(1) as usize).min(islands.len());
    let first = state.island.saturating_sub(per_screen - 1).min(islands.len().saturating_sub(per_screen));
    let shown: Vec<usize> = (first..first + per_screen).collect();
    let cells = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(shown.iter().map(|_| Constraint::Ratio(1, per_screen as u32)).collect::<Vec<_>>())
        .split(area);
    for (cell, &i) in cells.iter().zip(&shown) {
        island_card(f, state, i, *cell, i == state.island);
    }
}

fn island_card(f: &mut Frame, state: &WatchState, i: usize, area: Rect, selected: bool) {
    let Some(island) = state.islands().get(i) else { return };
    let style = if selected { Style::default().fg(Color::Cyan) } else { Style::default().fg(Color::DarkGray) };
    let title = format!(" {} · {} rows {} ", island.id.to_uppercase(), island.rows, if i + 1 < state.islands().len() || i > 0 { format!("({}/{})", i + 1, state.islands().len()) } else { String::new() });
    let block = Block::default().borders(Borders::ALL).border_style(style).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)])
        .split(inner);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("best ", Style::default().fg(Color::DarkGray)),
            Span::styled(or_dash(island.best_hff, 4), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("  avg ", Style::default().fg(Color::DarkGray)),
            Span::raw(or_dash(island.avg_hff, 3)),
        ])),
        rows[0],
    );
    // THE SEARCH PULSE: three labelled meters, NO composite. Each is its own
    // measurement with its own name, and a dash carries its reason.
    for (meter, row) in state.pulse(i).iter().zip(rows[1..4].iter()) {
        meter_line(f, meter.0, meter.1, meter.2, *row);
    }
}

/// One pulse meter. A value draws a bar; None draws the REASON it is not there,
/// because "—" on its own tells an operator nothing about whether the metric is
/// broken, off, or simply not applicable in this window.
fn meter_line(f: &mut Frame, label: &str, value: Option<f64>, reason: &str, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(12), Constraint::Min(6)])
        .split(area);
    f.render_widget(Paragraph::new(Span::styled(label, Style::default().fg(Color::DarkGray))), columns[0]);
    match value {
        Some(v) => f.render_widget(
            LineGauge::default().filled_style(Style::default().fg(Color::Cyan)).ratio(v.clamp(0.0, 1.0)).label(format!("{:.0}%", v * 100.0)),
            columns[1],
        ),
        None => f.render_widget(
            Paragraph::new(Span::styled(format!("—  {reason}"), Style::default().fg(Color::Yellow))),
            columns[1],
        ),
    }
}

/// The 80x24 fallback's island region: ONE island, the selected one, on one
/// summary line plus its meters.
fn one_island(f: &mut Frame, state: &WatchState, area: Rect) {
    island_card(f, state, state.island, area, true);
}

fn cohort_table(f: &mut Frame, state: &WatchState, area: Rect, wide: bool) {
    // The title says WHOSE cohorts these are. A per-island split and a global
    // total look identical as a table of numbers, and the brief is emphatic that
    // no panel may pretend to know which island a row occupied.
    let scope = match state.island_focus.then(|| state.islands().get(state.island)).flatten() {
        Some(island) => format!("{} only", island.id.to_uppercase()),
        None => "global totals".to_string(),
    };
    let title = format!(
        " COHORTS · {scope} · sorted by {} · {} rows{} ",
        state.sort.label(),
        state.rows().len(),
        if state.filter.is_empty() { String::new() } else { format!(" · filter /{}", state.filter) }
    );
    let block = Block::default().borders(Borders::ALL).title(title);
    if !state.cohorts_on() {
        // A single label over the whole population is not a cohort table, and a
        // one-row table pretending otherwise is worse than saying so.
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("cohorts are off for this run", Style::default().fg(Color::Yellow))),
                Line::from(Span::styled("(EVOLVE_COHORT_MERGE / Config::cohort_merge is 0)", Style::default().fg(Color::DarkGray))),
            ])
            .block(block),
            area,
        );
        return;
    }
    let rows = state.rows();
    // The gain column states the window it is over. A snapshot is written at the
    // progress beat but no more often than once a second, so the window is
    // whatever the fit's speed made it — labelling it "per 10 generations" would
    // be wrong on most runs.
    let gain_column = state.gain_window().map_or_else(|| "gain".to_string(), |g| format!("gain/{g}g"));
    let header = if wide {
        Row::new(vec!["cohort".to_string(), "born".into(), "rows".into(), "best HFF ↓".into(), "best ever".into(), gain_column, "trajectory".into()])
    } else {
        Row::new(vec!["cohort".to_string(), "rows".into(), "best HFF ↓".into(), gain_column])
    }
    .style(Style::default().fg(Color::DarkGray));
    let body: Vec<Row> = rows
        .iter()
        .map(|r| {
            let selected = state.selected == Some(r.id);
            let style = match (selected, r.extinct) {
                (true, _) => Style::default().bg(Color::Rgb(26, 58, 67)).add_modifier(Modifier::BOLD),
                (false, true) => Style::default().fg(Color::DarkGray),
                (false, false) => Style::default(),
            };
            let gain = Cell::from(gain_text(r.gain)).style(Style::default().fg(gain_colour(r.gain)));
            if wide {
                Row::new(vec![
                    Cell::from(format!("c{}", r.id)),
                    Cell::from(format!("{}", r.birth_generation)),
                    Cell::from(format!("{}", r.rows)),
                    Cell::from(or_dash(r.best_hff, 4)),
                    Cell::from(or_dash(r.best_ever, 4)),
                    gain,
                    Cell::from(bars(&r.spark)).style(Style::default().fg(Color::Cyan)),
                ])
                .style(style)
            } else {
                Row::new(vec![
                    Cell::from(format!("c{}", r.id)),
                    Cell::from(format!("{}", r.rows)),
                    Cell::from(or_dash(r.best_hff, 3)),
                    gain,
                ])
                .style(style)
            }
        })
        .collect();
    let widths: Vec<Constraint> = if wide {
        vec![
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Min(12),
        ]
    } else {
        vec![Constraint::Length(9), Constraint::Length(8), Constraint::Length(12), Constraint::Min(8)]
    };
    f.render_widget(Table::new(body, widths).header(header).block(block), area);
}

/// A sparkline as block characters: LOWER HFF IS A TALLER BAR, and a gap (the
/// cohort had no best then) is a space rather than a carried-forward value.
fn bars(spark: &[u64]) -> String {
    const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    spark.iter().rev().take(24).rev().map(|&v| BLOCKS[(v as usize).min(8)]).collect()
}

fn detail(f: &mut Frame, state: &WatchState, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" COHORT DETAIL ");
    let Some(id) = state.selected else {
        f.render_widget(Paragraph::new(Span::styled("no cohort selected", Style::default().fg(Color::DarkGray))).block(block), area);
        return;
    };
    let Some(r) = state.selected_row() else {
        // The selection is KEPT even when its row has gone — the viewer does not
        // reassign it, it explains it.
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(format!("c{id}"), Style::default().add_modifier(Modifier::BOLD))),
                Line::from(Span::styled("not in the current table", Style::default().fg(Color::Yellow))),
                Line::from(Span::styled("(extinct, or excluded by the filter)", Style::default().fg(Color::DarkGray))),
                Line::from(Span::styled("the selection is kept — j/k to move it", Style::default().fg(Color::DarkGray))),
            ])
            .block(block)
            .wrap(Wrap { trim: true }),
            area,
        );
        return;
    };
    let g = state.snapshot.as_ref().map(|s| &s.global);
    let span = r.spark_span.map_or_else(|| "—".to_string(), |(a, b)| format!("gen {a}–{b}"));
    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!("c{} ", r.id), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(format!("born gen {}", r.birth_generation), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(format!("best now  {}", or_dash(r.best_hff, 6))),
        Line::from(format!("best ever {}", or_dash(r.best_ever, 6))),
        Line::from(vec![Span::raw("gain      "), Span::styled(gain_text(r.gain), Style::default().fg(gain_colour(r.gain)))]),
        Line::from(format!("rows      {}  ({} unscored)", r.rows, r.nan_rows)),
        Line::from(""),
        Line::from(Span::styled(format!("TRAJECTORY · {span}", ), Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled(bars(&r.spark), Style::default().fg(Color::Cyan))),
        // The sparkline's direction, stated on the screen as the brief asks: it
        // is the one thing about this trace that cannot be guessed right.
        Line::from(Span::styled("lower HFF = higher trace", Style::default().fg(Color::DarkGray))),
        Line::from(""),
        Line::from(Span::styled("GLOBAL (this run, not this cohort)", Style::default().fg(Color::DarkGray))),
        Line::from(format!("train 1-R²  {}", or_dash(g.and_then(|g| g.r2_train).map(|r| 1.0 - r), 3))),
        Line::from(format!("val   1-R²  {}", or_dash(g.and_then(|g| g.r2_val).map(|r| 1.0 - r), 3))),
    ];
    // A cohort label is inherited lineage MEMBERSHIP, not a count of independent
    // lines. The brief is emphatic about this and the screen says it where the
    // row count is, because that is where it would otherwise be misread.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "rows are descendants of one pump beat, not independent lines",
        Style::default().fg(Color::DarkGray),
    )));
    if let Some(m) = state.model.as_ref() {
        lines.push(Line::from(Span::styled(
            format!("m · model, found gen {} (run best, not this cohort's)", m.found_generation),
            Style::default().fg(Color::DarkGray),
        )));
    }
    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: true }), area);
}

fn events(f: &mut Frame, state: &WatchState, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" EVENTS ");
    let height = block.inner(area).height as usize;
    let lines: Vec<Line> = state
        .events
        .iter()
        .rev()
        .take(height)
        .rev()
        .map(|(gen, message)| {
            Line::from(vec![
                Span::styled(format!("gen {gen:<6} "), Style::default().fg(Color::DarkGray)),
                Span::raw(message.clone()),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn one_line_events(f: &mut Frame, state: &WatchState, area: Rect) {
    let last = state.events.last().map_or_else(
        || "no events".to_string(),
        |(gen, message)| format!("gen {gen} · {message}"),
    );
    f.render_widget(Paragraph::new(Span::styled(last, Style::default().fg(Color::DarkGray))), area);
}

fn footer(f: &mut Frame, state: &WatchState, area: Rect) {
    if state.filtering {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("/", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(state.filter.clone()),
                Span::styled("▏  Enter to keep · Esc to clear", Style::default().fg(Color::DarkGray)),
            ])),
            area,
        );
        return;
    }
    f.render_widget(
        Paragraph::new(Span::styled(
            "Tab island · j/k cohort · s sort · / filter · g global · Enter detail · m model · Space pause screen · q quit viewer (the fit runs on)",
            Style::default().fg(Color::DarkGray),
        )),
        area,
    );
}

/// `m`: the full expression, in tabs. The SIMPLIFIED tab says what it is rather
/// than showing a form this viewer computed — the fit's own final form costs a
/// saturation and is printed when the fit ends, and a different simplification
/// here would be a fourth string nobody asked for.
fn model_overlay(f: &mut Frame, state: &WatchState, ui: &Ui, area: Rect) {
    let width = area.width.saturating_sub(6).max(20);
    let height = area.height.saturating_sub(4).max(8);
    // saturating: on a terminal narrower than the overlay's own minimum this
    // would otherwise underflow, and a viewer must not panic because a window
    // was dragged small.
    let box_area = Rect { x: area.width.saturating_sub(width) / 2, y: area.height.saturating_sub(height) / 2, width, height };
    f.render_widget(Clear, box_area);
    let tabs: Vec<Span> = MODEL_TABS
        .iter()
        .enumerate()
        .flat_map(|(i, name)| {
            let style = if i == ui.model_tab {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            [Span::styled(*name, style), Span::raw("  ")]
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" MODEL · Tab switches · w wrap · h/l scroll · y write to a file · Esc close ");
    let inner = block.inner(box_area);
    f.render_widget(block, box_area);
    let Some(m) = state.model.as_ref() else {
        f.render_widget(
            Paragraph::new(Span::styled("no model record in this stream yet", Style::default().fg(Color::Yellow))),
            inner,
        );
        return;
    };
    let body = match ui.model_tab {
        0 => m.infix_protected.clone(),
        1 => m.infix_plain.clone(),
        _ => format!(
            "the simplified form is the FIT's own final form: it costs a saturation\nand is printed when the fit ends, so this viewer does not compute a\ndifferent one here.\n\nthe engine's Math, unsimplified:\n\n{}",
            m.raw_math
        ),
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)])
        .split(inner);
    f.render_widget(Paragraph::new(Line::from(tabs)), rows[0]);
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("found at generation {} · HFF {} · depth {}", m.found_generation, or_dash(m.hff, 6), m.t_depth),
            Style::default().fg(Color::DarkGray),
        )),
        rows[1],
    );
    let text = Paragraph::new(body);
    // Wrapping or horizontal scrolling, as the brief asks: a 900-character model
    // is unreadable wrapped into a wall and unreadable cut off, so it is the
    // operator's choice which.
    let text = if ui.model_wrap { text.wrap(Wrap { trim: false }) } else { text.scroll((0, ui.model_scroll)) };
    f.render_widget(text, rows[2]);
    f.render_widget(
        Paragraph::new(Span::styled(
            match &ui.copied {
                Some(path) => format!("written to {path}"),
                None => format!("{} · y writes all three forms to a file", if ui.model_wrap { "wrapped" } else { "scrolling" }),
            },
            Style::default().fg(Color::DarkGray),
        )),
        rows[3],
    );
}
