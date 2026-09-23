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
use fuller::evolve::watch::{elided, gain_text, or_dash, Gain, Liveness, WatchState};
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

/// THE PALETTE, because a terminal is not always dark.
///
/// `DarkGray` on a white background is very nearly white, so every dimmed label
/// — the column headings, the units, the "no pump in this window" notes —
/// vanished on a light terminal, and the selected row's dark blue-grey took the
/// text with it.
///
/// There is no portable way to ASK a terminal what colour it is. `COLORFGBG` is
/// set by some (rxvt, konsole, and terminals that copy them) and is the only
/// widely-honoured hint; the OSC 11 query is not safe to send when we do not own
/// the terminal's input. So: honour `COLORFGBG` when it is there, honour an
/// explicit `HFF_WATCH_THEME=light|dark` above everything, and default to dark,
/// which is what this is usually watched on.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Theme {
    light: bool,
}

impl Theme {
    fn detect() -> Theme {
        if let Ok(v) = std::env::var("HFF_WATCH_THEME") {
            return Theme { light: v.eq_ignore_ascii_case("light") };
        }
        // COLORFGBG is "fg;bg" or "fg;_;bg"; a background of 7 or 15 is white,
        // and anything 0-6 or 8 is dark. Anything else we cannot read, so dark.
        let light = std::env::var("COLORFGBG").ok().is_some_and(|v| {
            v.rsplit(';').next().and_then(|bg| bg.trim().parse::<u8>().ok()).is_some_and(|bg| bg >= 7 && bg != 8)
        });
        Theme { light }
    }

    /// A label, a unit, a heading: present but not the point. The whole reason
    /// this type exists — `DarkGray` is unreadable on white.
    fn dim(self) -> Color {
        if self.light { Color::Rgb(110, 110, 110) } else { Color::DarkGray }
    }

    /// The selected row's background. A wash, not a block.
    fn selection(self) -> Color {
        if self.light { Color::Rgb(206, 228, 236) } else { Color::Rgb(26, 58, 67) }
    }

    /// The INK on a selected row. A cell that carries its own colour — the gain,
    /// the sparkline — keeps that colour over the wash, and on a light terminal
    /// a cyan sparkline or a grey gain on a pale blue background is not
    /// readable. A selected row therefore takes one explicit foreground for
    /// every cell, dark on the light theme and bright on the dark one, so the
    /// selection never hides what it is selecting.
    fn selected_ink(self) -> Color {
        if self.light { Color::Rgb(10, 30, 40) } else { Color::Rgb(226, 242, 247) }
    }

    /// The accent — the run's own numbers. Cyan is legible on both, but it is
    /// thin on white, so the light theme takes it darker.
    fn accent(self) -> Color {
        if self.light { Color::Rgb(0, 95, 115) } else { Color::Cyan }
    }

    /// A warning, or a metric that is not instrumented.
    fn warn(self) -> Color {
        if self.light { Color::Rgb(140, 100, 0) } else { Color::Yellow }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.iter().position(|a| a == "--file" || a == "--follow" || a == "--dump");
    let Some(at) = mode else {
        eprintln!("usage: hff-watch --file <path> | --follow <path> | --dump <path>");
        eprintln!("  --file    replay a recorded telemetry stream");
        eprintln!("  --follow  tail a live one (the fit is never touched)");
        eprintln!("  --dump    print the current frame as text and exit");
        std::process::exit(2);
    };
    let following = args[at] == "--follow";
    let dumping = args[at] == "--dump";
    let Some(path) = args.get(at + 1).cloned() else {
        eprintln!("{} needs a path", args[at]);
        std::process::exit(2);
    };
    let mut state = WatchState::new(following);
    // A REPLAY is read whole, up front: it is a recording, there is nothing to
    // wait for, and reading it in one go means the first frame is the last one
    // rather than an empty screen that fills in.
    let mut tailer = if following && !dumping {
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
    // --dump: the same state machine, printed rather than drawn. A terminal is
    // never opened, so it works down a pipe, in a log, or from another process
    // watching alongside an operator who has the live view open.
    if dumping {
        print!("{}", dump(&state));
        return;
    }
    let mut ui = Ui { theme: Theme::detect(), ..Ui::default() };
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
        // A KEYBOARD THAT IS NOT THERE IS NOT A REASON TO STOP WATCHING. With
        // stdin redirected — a viewer in a pane, under `script`, or on a headless
        // box — crossterm cannot start an input reader and every poll errors.
        // Ending the session on that would mean the screen that needs watching
        // most (an unattended run) is the one that cannot be watched, so the
        // errors are counted and the viewer goes on repainting. It is then a
        // display, and the process is ended from outside.
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
                Err(_) => ui.input_errors += 1,
            },
            Ok(false) => {}
            Err(_) => {
                ui.input_errors += 1;
                // Without a working poll the loop has no timer of its own, and
                // it would spin a core at the speed of the file system.
                std::thread::sleep(TICK);
            }
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
    /// The palette. Detected once at start-up: a terminal does not change
    /// colour mid-run, and re-detecting every frame would read the environment
    /// sixty times a minute for an answer that cannot have moved.
    theme: Theme,
    model_open: bool,
    model_tab: usize,
    /// `Enter`: the detail pane is expanded over the table on a narrow screen.
    detail_open: bool,
    /// Horizontal scroll in the model overlay, for a model too wide to wrap
    /// usefully. `w` toggles wrapping.
    model_scroll: u16,
    model_wrap: bool,
    copied: Option<String>,
    /// Polls that could not read the keyboard — stdin redirected, no tty. The
    /// viewer keeps repainting as a display and SAYS SO, because a screen that
    /// silently ignores every key looks broken rather than read-only.
    input_errors: u64,
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
    let theme = ui.theme;
    let area = f.area();
    if area.width < MINIMUM.0 || area.height < MINIMUM.1 {
        compact(f, theme, state, area);
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
                Constraint::Min(8),     // cohort table + discoveries, and detail
                Constraint::Length(1),  // the winning gene, one line
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
                // ONE island, the selected one: two borders, the best/avg line
                // and ALL THREE pulse meters. A card that silently drops the
                // third meter is a screen that hides a measurement.
                Constraint::Length(6),
                // At 80x24 this is 5 rows for the table and 4 for the
                // discoveries below it: the panel degrades to a short list
                // rather than disappearing, which is the whole point of it.
                Constraint::Min(5),
                Constraint::Length(1), // the winning gene, one line
                Constraint::Length(1), // one-line events
                Constraint::Length(1),
            ])
            .split(area)
    };
    header(f, theme, state, rows[0]);
    global_strip(f, theme, state, rows[1]);
    if wide {
        island_cards(f, theme, state, rows[2]);
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(60), Constraint::Length(38)])
            .split(rows[3]);
        // `Enter` expands the detail over the table, which is how a long model
        // summary is read on a screen that is wide but not tall.
        if ui.detail_open {
            detail(f, theme, state, rows[3]);
        } else {
            // THE DEAD SPACE UNDER THE TABLE is where the discoveries go. The
            // cohort table rarely holds more than ten rows and the region is
            // `Min(8)` of whatever is left, so on a tall terminal the table sat
            // in a box three times its own height. The table takes what it
            // needs; the discoveries take the rest.
            let stack = table_and_discoveries(split[0], state.rows().len());
            cohort_table(f, theme, state, stack[0], true);
            discoveries(f, theme, state, stack[1]);
            detail(f, theme, state, split[1]);
        }
        gene_line(f, theme, state, rows[4]);
        events(f, theme, state, rows[5]);
    } else {
        one_island(f, theme, state, rows[2]);
        if ui.detail_open {
            detail(f, theme, state, rows[3]);
        } else {
            let stack = table_and_discoveries(rows[3], state.rows().len());
            cohort_table(f, theme, state, stack[0], false);
            discoveries(f, theme, state, stack[1]);
        }
        gene_line(f, theme, state, rows[4]);
        one_line_events(f, theme, state, rows[5]);
    }
    footer(f, state, ui, rows[6]);
    if ui.model_open {
        model_overlay(f, state, ui, area);
    }
}

/// Below 70x20: the essential metrics and a warning, and nothing that would be
/// unreadable at this size. It is still a useful screen — the global best, the
/// generation and the clock — not an error message.
fn compact(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    let s = state.snapshot.as_ref();
    let generation = s.map_or(0, |s| s.header.generation);
    let mut lines = vec![
        Line::from(Span::styled("hff-watch · terminal too small", Style::default().fg(theme.warn()).add_modifier(Modifier::BOLD))),
        Line::from(format!("{}x{} — the full screen needs {}x{}", area.width, area.height, WIDE.0, WIDE.1)),
        Line::from(""),
        Line::from(format!("gen {generation}  best HFF {}", or_dash(s.and_then(|s| s.global.best_hff), 4))),
        // Seconds as seconds. `or_dash` is scientific notation, which is right
        // for an HFF angle and absurd for a clock ("3e1 seconds").
        Line::from(format!(
            "{}  {}",
            badge_text(state),
            s.map_or_else(|| "—".to_string(), |s| format!("{:.0}s", s.header.elapsed_ms as f64 / 1000.0))
        )),
    ];
    lines.push(Line::from(Span::styled("q to quit", Style::default().fg(theme.dim()))));
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

fn badge_colour(theme: Theme, state: &WatchState) -> Color {
    match state.liveness() {
        Liveness::Finished => Color::Green,
        Liveness::Stale => Color::Red,
        Liveness::Live => theme.accent(),
    }
}

fn header(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
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
        Span::styled("◉ HFF-SR / WATCH ", Style::default().fg(theme.accent()).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", badge_text(state)), Style::default().fg(badge_colour(theme, state)).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(start.map_or_else(|| "(no run_start)".to_string(), |s| s.dataset.clone())),
        Span::styled(" · seed ", Style::default().fg(theme.dim())),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| s.seed.to_string())),
        Span::styled("  run ", Style::default().fg(theme.dim())),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| s.header.run_id.clone())),
    ]);
    let two = Line::from(vec![
        Span::styled("gen ", Style::default().fg(theme.dim())),
        Span::styled(format!("{generation}"), Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!("  {elapsed:.0}s / {budget:.0}s")),
        Span::styled("  sampled ", Style::default().fg(theme.dim())),
        Span::raw(age),
        Span::styled("  train/val/third ", Style::default().fg(theme.dim())),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| format!("{}/{}/{}", s.n_train, s.n_val, s.n_extrap))),
        // A viewer that is silently skipping input is lying about what it shows.
        Span::styled(
            if state.bad_lines > 0 || state.rotations > 0 {
                format!("  {} bad lines, {} rotations", state.bad_lines, state.rotations)
            } else {
                String::new()
            },
            Style::default().fg(theme.warn()),
        ),
    ]);
    f.render_widget(Paragraph::new(vec![one, two]), area);
}

fn global_strip(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    let s = state.snapshot.as_ref();
    let g = s.map(|s| &s.global);
    // Three columns need about 40 each to say anything. Below that the error
    // column folds into the best one rather than both being cut off mid-word:
    // half a heading is worse than a tighter line.
    let roomy = area.width >= 118;
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(if roomy {
            vec![Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)]
        } else {
            vec![Constraint::Min(30), Constraint::Length(26)]
        })
        .split(area);
    let gain = state.global_gain();
    let (train, val) = (
        or_dash(g.and_then(|g| g.r2_train).map(|r| 1.0 - r), 2),
        or_dash(g.and_then(|g| g.r2_val).map(|r| 1.0 - r), 2),
    );
    let best = Paragraph::new(vec![
        Line::from(Span::styled(
            if roomy { "GLOBAL BEST HFF ↓ (lower is better)" } else { "GLOBAL BEST HFF ↓ (lower wins)" },
            Style::default().fg(theme.dim()),
        )),
        Line::from(vec![
            Span::styled(or_dash(g.and_then(|g| g.best_hff), 6), Style::default().fg(theme.accent()).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(gain_text(gain), Style::default().fg(gain_colour(theme, gain))),
            // Narrow: the errors come here rather than into a column too thin to
            // hold their headings.
            Span::styled(
                if roomy { String::new() } else { format!("   1-R² train {train} val {val}") },
                Style::default().fg(theme.dim()),
            ),
        ]),
        // BEST-EVER, separately: a current best is not a record, and the brief
        // asks for the two not to be confused.
        Line::from(Span::styled(
            format!("best ever {}   {} unscored rows", or_dash(g.and_then(|g| g.best_ever_hff), 4), g.map_or(0, |g| g.nan_rows)),
            Style::default().fg(theme.dim()),
        )),
    ]);
    f.render_widget(best, columns[0]);
    if roomy {
        let quality = Paragraph::new(vec![
            Line::from(Span::styled("ERROR", Style::default().fg(theme.dim()))),
            Line::from(format!("train 1-R² {train}   val 1-R² {val}")),
            Line::from(Span::styled(
                format!(
                    "mse {}  log10 p {}  depth {}  head {}",
                    or_dash(g.and_then(|g| g.mse_train), 2),
                    g.and_then(|g| g.log10_p).map_or_else(|| "—".to_string(), |p| format!("{p:.2}")),
                    g.map_or(0, |g| g.t_depth),
                    g.map_or(0, |g| g.vhead)
                ),
                Style::default().fg(theme.dim()),
            )),
        ]);
        f.render_widget(quality, columns[1]);
    }
    // THE BUDGET BAR is a bar of TIME, which is the only thing on this screen
    // that is genuinely a fraction of a known total. HFF is never drawn as a
    // percentage: it has no ceiling to be a percentage of.
    let (elapsed, budget) = (s.map_or(0, |s| s.header.elapsed_ms), s.map_or(0, |s| s.budget_ms).max(1));
    let fraction = (elapsed as f64 / budget as f64).clamp(0.0, 1.0);
    let bar = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
        // The budget is always the LAST column, whether the strip has three or
        // (narrow, with the errors folded into the first) two.
        .split(columns[columns.len() - 1]);
    f.render_widget(Paragraph::new(Span::styled("BUDGET", Style::default().fg(theme.dim()))), bar[0]);
    // A THIN TRACK with the number beside it, not a block bar with the label
    // buried in it: at 40 columns a filled `Gauge` reads as a wall of blocks
    // with a percentage somewhere inside, which is not a reading.
    f.render_widget(
        LineGauge::default().filled_style(Style::default().fg(theme.accent())).ratio(fraction).label(format!("{:.0}%", fraction * 100.0)),
        bar[1],
    );
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("{:.0}s left", (budget.saturating_sub(elapsed)) as f64 / 1000.0),
            Style::default().fg(theme.dim()),
        )),
        bar[2],
    );
}

fn gain_colour(theme: Theme, g: Gain) -> Color {
    match g {
        // A FALLING HFF is an improvement, and it is cyan. This is the one place
        // a sign could be read backwards, so it is written out: positive gain =
        // HFF fell = better.
        Gain::Percent(v) if v > 0.0 => theme.accent(),
        Gain::Percent(v) if v < 0.0 => Color::Red,
        Gain::Percent(_) => Color::Gray,
        Gain::New | Gain::Merged => theme.warn(),
        Gain::Extinct => Color::Red,
        Gain::Unscored => theme.dim(),
    }
}

fn island_cards(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    let islands = state.islands();
    if islands.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("no island data yet", Style::default().fg(theme.dim())))
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
        island_card(f, theme, state, i, *cell, i == state.island);
    }
}

fn island_card(f: &mut Frame, theme: Theme, state: &WatchState, i: usize, area: Rect, selected: bool) {
    let Some(island) = state.islands().get(i) else { return };
    let style = if selected { Style::default().fg(theme.accent()) } else { Style::default().fg(theme.dim()) };
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
            Span::styled("best ", Style::default().fg(theme.dim())),
            Span::styled(or_dash(island.best_hff, 4), Style::default().fg(theme.accent()).add_modifier(Modifier::BOLD)),
            Span::styled("  avg ", Style::default().fg(theme.dim())),
            Span::raw(or_dash(island.avg_hff, 3)),
        ])),
        rows[0],
    );
    // THE SEARCH PULSE: three labelled meters, NO composite. Each is its own
    // measurement with its own name, and a dash carries its reason.
    for (meter, row) in state.pulse(i).iter().zip(rows[1..4].iter()) {
        meter_line(f, theme, meter.0, meter.1, meter.2, *row);
    }
}

/// One pulse meter. A value draws a bar; None draws the REASON it is not there,
/// because "—" on its own tells an operator nothing about whether the metric is
/// broken, off, or simply not applicable in this window.
fn meter_line(f: &mut Frame, theme: Theme, label: &str, value: Option<f64>, reason: &str, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(12), Constraint::Min(6)])
        .split(area);
    f.render_widget(Paragraph::new(Span::styled(label, Style::default().fg(theme.dim()))), columns[0]);
    match value {
        Some(v) => f.render_widget(
            LineGauge::default().filled_style(Style::default().fg(theme.accent())).ratio(v.clamp(0.0, 1.0)).label(format!("{:.0}%", v * 100.0)),
            columns[1],
        ),
        None => f.render_widget(
            Paragraph::new(Span::styled(format!("—  {reason}"), Style::default().fg(theme.warn()))),
            columns[1],
        ),
    }
}

/// The 80x24 fallback's island region: ONE island, the selected one, on one
/// summary line plus its meters.
fn one_island(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    island_card(f, theme, state, state.island, area, true);
}

fn cohort_table(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect, wide: bool) {
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
                Line::from(Span::styled("cohorts are off for this run", Style::default().fg(theme.warn()))),
                Line::from(Span::styled("(EVOLVE_COHORT_MERGE / Config::cohort_merge is 0)", Style::default().fg(theme.dim()))),
            ])
            .block(block),
            area,
        );
        return;
    }
    // A FOCUSED ISLAND THAT EMITTED NO SPLIT is a silence, not an empty island.
    // The brief's own example record carries `cohorts: []` because the log it
    // was derived from could not say which island a cohort's rows sat on, and
    // drawing a blank table under that heading would be the screen pretending to
    // know.
    if state.island_focus && state.islands().get(state.island).is_some_and(|i| i.cohorts.is_empty()) {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("no per-island cohort split in this stream", Style::default().fg(theme.warn()))),
                Line::from(Span::styled("the producer emitted global totals only · g for those", Style::default().fg(theme.dim()))),
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
    .style(Style::default().fg(theme.dim()));
    let body: Vec<Row> = rows
        .iter()
        .map(|r| {
            let selected = state.selected == Some(r.id);
            let style = match (selected, r.extinct) {
                (true, _) => Style::default().bg(theme.selection()).fg(theme.selected_ink()).add_modifier(Modifier::BOLD),
                (false, true) => Style::default().fg(theme.dim()),
                (false, false) => Style::default(),
            };
            // A selected row's cells take the row's ink; unselected they keep
            // their own meaning-carrying colour.
            let ink = |own: Color| if selected { theme.selected_ink() } else { own };
            let gain = Cell::from(gain_text(r.gain)).style(Style::default().fg(ink(gain_colour(theme, r.gain))));
            if wide {
                Row::new(vec![
                    Cell::from(format!("c{}", r.id)),
                    Cell::from(format!("{}", r.birth_generation)),
                    Cell::from(format!("{}", r.rows)),
                    Cell::from(or_dash(r.best_hff, 4)),
                    Cell::from(or_dash(r.best_ever, 4)),
                    gain,
                    Cell::from(bars(&r.spark)).style(Style::default().fg(ink(theme.accent()))),
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

/// The smallest box the discoveries panel is worth drawing in: two borders and
/// two rows of list. Below that the table keeps the whole region — half a
/// bordered box with no room for a line in it is not a panel.
const DISCOVERIES_MIN: u16 = 4;

/// SPLIT THE TABLE'S COLUMN between the cohort table and the discoveries.
///
/// The table takes what its rows need — a header, two borders and one line per
/// cohort — and the discoveries take the rest. That is the right way round: the
/// table's height is known and bounded (a fit rarely holds more than a dozen
/// cohorts at once), and the discoveries are a scrolling list with no natural
/// end, so giving the fixed thing its size and the growing thing the remainder
/// means neither is ever cut when the other is short.
///
/// WHEN THE TABLE ALONE WOULD FILL THE REGION, the discoveries still get their
/// minimum and the table is scrolled rather than the panel being dropped. A
/// cohort table is already a scrolling list — it lingers its dead for three
/// beats and a long fit mints one cohort per pump — so a row below the fold is
/// a row an operator reaches, while a panel that is not drawn at all is a
/// finding they never learn exists. Only when even that would leave the table
/// nothing to show does the table keep the whole region.
fn table_and_discoveries(area: Rect, cohorts: usize) -> Vec<Rect> {
    // The table's own minimum: two borders, the header, and two rows of it.
    const TABLE_MIN: u16 = 5;
    let wanted = (cohorts as u16).saturating_add(3);
    if area.height < TABLE_MIN.saturating_add(DISCOVERIES_MIN) {
        return vec![area, Rect { height: 0, ..area }];
    }
    // When the table wants more than the region has, the two SHARE it rather
    // than the panel being squeezed to its bare minimum: a list of two lines
    // under a table of nine is a panel nobody reads, and the table below the
    // fold is scrolled to as it always was. A third to the panel, which is
    // three or four findings at the sizes this is actually watched on.
    let ceiling = (area.height - DISCOVERIES_MIN).max(TABLE_MIN);
    let shared = (area.height - area.height / 3).max(TABLE_MIN);
    let table = wanted.min(shared).clamp(TABLE_MIN, ceiling);
    Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(table), Constraint::Min(DISCOVERIES_MIN)])
        .split(area)
        .to_vec()
}

/// THE DISCOVERIES: what snap substituted and what the rounding generator
/// folded, newest first, one line each.
///
/// The two kinds are kept visibly apart — a snap is a substitution INTO the
/// population and a fold is a reduction of the FINAL model — because a list that
/// mixed them would be a list of unrelated numbers. They carry their own colour
/// and their own arrow, and the heading names both.
fn discoveries(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    use fuller::evolve::watch::Find;
    if area.height < DISCOVERIES_MIN {
        return;
    }
    let (snaps, folds) = state.discoveries.iter().fold((0, 0), |(s, d), x| match x.kind {
        Find::Snap => (s + 1, d),
        Find::Fold => (s, d + 1),
    });
    let plural = |n: usize, one: &str, many: &str| if n == 1 { one.to_string() } else { many.to_string() };
    let block = Block::default().borders(Borders::ALL).title(format!(
        " DISCOVERIES · {snaps} {} snapped into genes · {folds} {} folded ",
        plural(snaps, "literal", "literals"),
        plural(folds, "subtree", "subtrees")
    ));
    let inner = block.inner(area);
    if state.discoveries.is_empty() {
        // Nothing found is NOT nothing to say: snap and the fold are both
        // switched on separately, and an empty panel that does not explain
        // itself reads as a broken one.
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("nothing yet", Style::default().fg(theme.dim()))),
                Line::from(Span::styled(
                    "snaps arrive through the run · folds once, after it ends",
                    Style::default().fg(theme.dim()),
                )),
            ])
            .block(block),
            area,
        );
        return;
    }
    let width = inner.width as usize;
    let lines: Vec<Line> = state
        .discoveries
        .iter()
        .rev()
        .take(inner.height as usize)
        .map(|d| {
            // A snap is the accent — it is the thing that went INTO the
            // population. A fold is the warn colour: it removed structure from
            // the reported model, which is a different kind of news. Both are
            // legible on either terminal.
            let (tag, colour) = match d.kind {
                Find::Snap => ("snap", theme.accent()),
                Find::Fold => ("fold", theme.warn()),
            };
            let size = d.nodes.map_or_else(String::new, |n| format!("{n} nodes "));
            let body = format!("{size}{} → {}", d.what, d.became);
            Line::from(vec![
                Span::styled(format!("gen {:<6} ", d.generation), Style::default().fg(theme.dim())),
                Span::styled(format!("{tag} "), Style::default().fg(colour).add_modifier(Modifier::BOLD)),
                // The list is one line per discovery: a fourteen-node blob is
                // the finding, and wrapping it would push the next one off.
                Span::raw(elided(&body, width.saturating_sub(16))),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// THE WINNING GENE, one line, pretty-printed as a function.
///
/// `f(x, y) = ...`, not the engine's s-expression: it is there to be READ, and
/// watching it change as the fit runs is the point of putting it on the screen
/// at all. The literals are rounded for display; `m` still shows the model to
/// the last digit. Too long for the terminal is TRUNCATED and not wrapped — one
/// line is a line, and the full form is a keystroke away.
fn gene_line(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    let label = "model  ";
    let body = match state.gene_line.as_deref() {
        Some(line) => Span::raw(elided(line, (area.width as usize).saturating_sub(label.len()))),
        None if state.model.is_some() => Span::styled("the model record did not parse", Style::default().fg(theme.warn())),
        None => Span::styled("no model record yet", Style::default().fg(theme.dim())),
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(label, Style::default().fg(theme.dim())), body])),
        area,
    );
}

fn detail(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" COHORT DETAIL ");
    let Some(id) = state.selected else {
        f.render_widget(Paragraph::new(Span::styled("no cohort selected", Style::default().fg(theme.dim()))).block(block), area);
        return;
    };
    let Some(r) = state.selected_row() else {
        // The selection is KEPT even when its row has gone — the viewer does not
        // reassign it, it explains it.
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(format!("c{id}"), Style::default().add_modifier(Modifier::BOLD))),
                Line::from(Span::styled("not in the current table", Style::default().fg(theme.warn()))),
                Line::from(Span::styled("(extinct, or excluded by the filter)", Style::default().fg(theme.dim()))),
                Line::from(Span::styled("the selection is kept — j/k to move it", Style::default().fg(theme.dim()))),
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
            Span::styled(format!("c{} ", r.id), Style::default().fg(theme.accent()).add_modifier(Modifier::BOLD)),
            Span::styled(format!("born gen {}", r.birth_generation), Style::default().fg(theme.dim())),
        ]),
        Line::from(format!("best now  {}", or_dash(r.best_hff, 6))),
        Line::from(format!("best ever {}", or_dash(r.best_ever, 6))),
        Line::from(vec![Span::raw("gain      "), Span::styled(gain_text(r.gain), Style::default().fg(gain_colour(theme, r.gain)))]),
        Line::from(format!("rows      {}  ({} unscored)", r.rows, r.nan_rows)),
        Line::from(""),
        Line::from(Span::styled(format!("TRAJECTORY · {span}", ), Style::default().fg(theme.dim()))),
        Line::from(Span::styled(bars(&r.spark), Style::default().fg(theme.accent()))),
        // The sparkline's direction, stated on the screen as the brief asks: it
        // is the one thing about this trace that cannot be guessed right.
        Line::from(Span::styled("lower HFF = higher trace", Style::default().fg(theme.dim()))),
        Line::from(""),
        Line::from(Span::styled("GLOBAL (this run, not this cohort)", Style::default().fg(theme.dim()))),
        Line::from(format!("train 1-R²  {}", or_dash(g.and_then(|g| g.r2_train).map(|r| 1.0 - r), 3))),
        Line::from(format!("val   1-R²  {}", or_dash(g.and_then(|g| g.r2_val).map(|r| 1.0 - r), 3))),
    ];
    // A cohort label is inherited lineage MEMBERSHIP, not a count of independent
    // lines. The brief is emphatic about this and the screen says it where the
    // row count is, because that is where it would otherwise be misread.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "rows are descendants of one pump beat, not independent lines",
        Style::default().fg(theme.dim()),
    )));
    if let Some(m) = state.model.as_ref() {
        lines.push(Line::from(Span::styled(
            format!("m · model, found gen {} (run best, not this cohort's)", m.found_generation),
            Style::default().fg(theme.dim()),
        )));
    }
    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: true }), area);
}

fn events(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
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
                Span::styled(format!("gen {gen:<6} "), Style::default().fg(theme.dim())),
                Span::raw(message.clone()),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn one_line_events(f: &mut Frame, theme: Theme, state: &WatchState, area: Rect) {
    let last = state.events.last().map_or_else(
        || "no events".to_string(),
        |(gen, message)| format!("gen {gen} · {message}"),
    );
    f.render_widget(Paragraph::new(Span::styled(last, Style::default().fg(theme.dim()))), area);
}

fn footer(f: &mut Frame, state: &WatchState, ui: &Ui, area: Rect) {
    let theme = ui.theme;
    if state.filtering {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("/", Style::default().fg(theme.warn()).add_modifier(Modifier::BOLD)),
                Span::raw(state.filter.clone()),
                Span::styled("▏  Enter to keep · Esc to clear", Style::default().fg(theme.dim())),
            ])),
            area,
        );
        return;
    }
    if ui.input_errors > 0 {
        f.render_widget(
            Paragraph::new(Span::styled(
                "no keyboard (stdin is not a terminal) · repainting as a display · end this process to stop watching",
                Style::default().fg(theme.warn()),
            )),
            area,
        );
        return;
    }
    f.render_widget(
        Paragraph::new(Span::styled(
            "Tab island · j/k cohort · s sort · / filter · g global · Enter detail · m model · Space pause screen · q quit viewer (the fit runs on)",
            Style::default().fg(theme.dim()),
        )),
        area,
    );
}

/// `m`: the full expression, in tabs. The SIMPLIFIED tab says what it is rather
/// than showing a form this viewer computed — the fit's own final form costs a
/// saturation and is printed when the fit ends, and a different simplification
/// here would be a fourth string nobody asked for.
fn model_overlay(f: &mut Frame, state: &WatchState, ui: &Ui, area: Rect) {
    let theme = ui.theme;
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
                Style::default().fg(theme.accent()).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.dim())
            };
            [Span::styled(*name, style), Span::raw("  ")]
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.accent()))
        .title(" MODEL · Tab switches · w wrap · h/l scroll · y write to a file · Esc close ");
    let inner = block.inner(box_area);
    f.render_widget(block, box_area);
    let Some(m) = state.model.as_ref() else {
        f.render_widget(
            Paragraph::new(Span::styled("no model record in this stream yet", Style::default().fg(theme.warn()))),
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
            Style::default().fg(theme.dim()),
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
            Style::default().fg(theme.dim()),
        )),
        rows[3],
    );
}


/// The current frame as plain text: the header, the global strip, the islands
/// and the cohort table, in the order the screen shows them.
///
/// It is the SAME `WatchState` the terminal view draws, so a dump and a screen
/// never disagree — and it means a second watcher can follow a run without
/// taking a terminal, which is how an operator and a script watch the same fit.
fn dump(state: &WatchState) -> String {
    use fuller::evolve::watch::{gain_text, or_dash};
    let mut out = String::new();
    let live = match state.liveness() {
        fuller::evolve::watch::Liveness::Live => "LIVE",
        fuller::evolve::watch::Liveness::Stale => "STALE",
        fuller::evolve::watch::Liveness::Finished => "FINISHED",
    };
    match (&state.start, &state.snapshot) {
        (Some(rs), Some(sn)) => {
            out.push_str(&format!(
                "{live}  {}  seed {}  gen {}  {:.0}/{:.0} s\n",
                rs.dataset,
                rs.seed,
                sn.header.generation,
                sn.header.elapsed_ms as f64 / 1000.0,
                sn.budget_ms as f64 / 1000.0
            ));
            out.push_str(&format!(
                "  best hff {}   1-R2 train {}   val {}   log10 p {}\n",
                or_dash(sn.global.best_hff, 6),
                or_dash(sn.global.r2_train.map(|r| 1.0 - r), 3),
                or_dash(sn.global.r2_val.map(|r| 1.0 - r), 3),
                or_dash(sn.global.log10_p, 2)
            ));
            out.push_str(&format!("  global gain {}\n", gain_text(state.global_gain())));
            for isl in state.islands() {
                out.push_str(&format!(
                    "  {:<12} rows {:>7}  best {}\n",
                    isl.id,
                    isl.rows,
                    or_dash(isl.best_hff, 6)
                ));
            }
        }
        _ => out.push_str(&format!("{live}  no snapshot yet\n")),
    }
    let rows = state.rows();
    out.push_str(&format!(
        "\n  {:>6}  {:>6}  {:>8}  {:>12}  {:>12}  {:>10}\n",
        "cohort", "born", "rows", "best hff", "best ever", "gain"
    ));
    for r in rows.iter().take(12) {
        out.push_str(&format!(
            "  {:>6}  {:>6}  {:>8}  {:>12}  {:>12}  {:>10}{}\n",
            r.id,
            r.birth_generation,
            r.rows,
            or_dash(r.best_hff, 6),
            or_dash(r.best_ever, 6),
            gain_text(r.gain),
            if r.extinct { "  EXTINCT" } else { "" }
        ));
    }
    for (gen, what) in state.events.iter().rev().take(4) {
        out.push_str(&format!("  gen {gen:>6}  {what}\n"));
    }
    // THE SAME TWO PANELS THE SCREEN DRAWS, so a dump and a terminal never
    // disagree about what the fit found — and so the panels can be checked
    // without a tty.
    out.push_str("\n  DISCOVERIES\n");
    if state.discoveries.is_empty() {
        out.push_str("  (none yet · snaps arrive through the run, folds once after it ends)\n");
    }
    for d in state.discoveries.iter().rev().take(12) {
        let tag = match d.kind {
            fuller::evolve::watch::Find::Snap => "snap",
            fuller::evolve::watch::Find::Fold => "fold",
        };
        let size = d.nodes.map_or_else(String::new, |n| format!("{n} nodes "));
        out.push_str(&format!("  gen {:>6}  {tag}  {size}{} -> {}\n", d.generation, d.what, d.became));
    }
    out.push_str(&format!(
        "\n  model  {}\n",
        state.gene_line.as_deref().map_or("(no model record yet)", |l| l)
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuller::evolve::telemetry::{parse_stream, Discovery as D, EventKind, Record};
    use fuller::evolve::watch::Find;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    const FIXTURE: &str = include_str!("../../tests/fixtures/telemetry_v1.jsonl");

    /// The fixture, plus a snap and a fold — the fixture predates both, and a
    /// panel drawn from a stream that has none of what it shows proves nothing.
    fn state_with_discoveries() -> WatchState {
        use fuller::evolve::telemetry::{Event, Header, SCHEMA_VERSION};
        let (records, bad) = parse_stream(FIXTURE);
        let mut state = WatchState::new(false);
        state.bad_lines = bad;
        for r in records {
            state.apply_record(r);
        }
        let mut event = |kind, generation, message: &str, d: D| {
            state.apply_record(Record::Event(Event {
                header: Header {
                    schema_version: SCHEMA_VERSION,
                    run_id: "r".into(),
                    seq: u64::from(generation),
                    timestamp_utc: "2026-09-23T00:00:00Z".into(),
                    elapsed_ms: 0,
                    generation,
                },
                kind,
                message: message.into(),
                cohort: None,
                value: d.after,
                before: d.before,
                detail: d.detail,
                nodes: d.nodes,
                row: d.row,
            }));
        };
        event(
            EventKind::Snap,
            240,
            "snap",
            D { before: Some(22.0 / 7.0), after: Some(std::f64::consts::PI), detail: Some("pi".into()), nodes: None, row: Some(7) },
        );
        event(
            EventKind::Fold,
            300,
            "fold",
            D { before: None, after: Some(1.0), detail: Some("tanh(exp(cos(log(x_0))))".into()), nodes: Some(14), row: None },
        );
        state
    }

    /// The whole screen, at one size and one theme, as the characters that
    /// landed on it.
    fn painted(state: &WatchState, theme: Theme, w: u16, h: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("a test terminal");
        let ui = Ui { theme, ..Ui::default() };
        terminal.draw(|f| draw(f, state, &ui)).expect("the frame draws");
        terminal
            .backend()
            .buffer()
            .content()
            .chunks(w as usize)
            .map(|row| row.iter().map(ratatui::buffer::Cell::symbol).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// THE NEW PANELS DRAW, at both layouts and in both themes, and the layout
    /// still holds: nothing is cut off, nothing panics, and the discoveries and
    /// the gene line are both on the screen at the wide size AND at 80x24.
    #[test]
    fn the_discoveries_and_the_gene_line_draw_at_both_sizes_and_in_both_themes() {
        let state = state_with_discoveries();
        for theme in [Theme { light: false }, Theme { light: true }] {
            for (w, h) in [(120, 35), (160, 50), (80, 24)] {
                let screen = painted(&state, theme, w, h);
                assert!(screen.contains("DISCOVERIES"), "{w}x{h} light={}: no discoveries panel\n{screen}", theme.light);
                assert!(screen.contains("snap"), "{w}x{h}: the snap is not on screen\n{screen}");
                assert!(screen.contains("fold"), "{w}x{h}: the fold is not on screen\n{screen}");
                assert!(screen.contains("14 nodes"), "{w}x{h}: the fold's size is not shown\n{screen}");
                assert!(screen.contains("model"), "{w}x{h}: no gene line\n{screen}");
                assert!(screen.contains("f("), "{w}x{h}: the gene line is not a function\n{screen}");
                // Every line fits the terminal: a panel that overruns its width
                // is a panel that has pushed something off the screen.
                for line in screen.lines() {
                    assert_eq!(line.chars().count(), w as usize, "{w}x{h}: a row is not the terminal's width");
                }
            }
        }
    }

    /// AND IT DEGRADES RATHER THAN BREAKING. Below 70x20 the compact warning is
    /// drawn and neither new panel is — it is still a useful screen, not an
    /// error message — and every size in between draws without panicking.
    #[test]
    fn a_small_terminal_falls_back_and_never_panics() {
        let state = state_with_discoveries();
        let small = painted(&state, Theme::default(), 69, 19);
        assert!(small.contains("terminal too small"), "{small}");
        assert!(!small.contains("DISCOVERIES"), "the compact screen drew a panel it has no room for\n{small}");
        // The essential numbers are still there: it is a screen, not an error.
        assert!(small.contains("gen ") && small.contains("best HFF"), "{small}");
        // EVERY size from the minimum up draws. A `Length` added to a layout is
        // exactly where a sum stops fitting, so the range is swept rather than
        // sampled.
        for h in 20..40u16 {
            for w in [70u16, 79, 80, 119, 120, 121] {
                let _ = painted(&state, Theme::default(), w, h);
            }
        }
    }

    /// A STREAM WITH NO DISCOVERIES SAYS SO. Snap and the fold are switched on
    /// separately, and an empty panel that does not explain itself reads as a
    /// broken one rather than as a fit that has not found anything yet.
    #[test]
    fn an_empty_discoveries_panel_explains_itself() {
        let (records, _) = parse_stream(FIXTURE);
        let mut state = WatchState::new(false);
        for r in records {
            state.apply_record(r);
        }
        assert!(state.discoveries.is_empty(), "the fixture predates the discovery events");
        let screen = painted(&state, Theme::default(), 120, 35);
        assert!(screen.contains("nothing yet"), "an empty panel said nothing\n{screen}");
        assert!(screen.contains("after it ends"), "it did not say when folds arrive\n{screen}");
        // And a stream with no model at all says that rather than drawing a
        // half-written function.
        let mut blank = WatchState::new(false);
        blank.apply_record(Record::Snapshot(state.snapshot.clone().expect("a frame")));
        let screen = painted(&blank, Theme::default(), 120, 35);
        assert!(screen.contains("no model record yet"), "{screen}");
    }

    /// THE TABLE TAKES WHAT ITS ROWS NEED, and the discoveries take the rest —
    /// so on a tall terminal every cohort is on screen AND the panel is under
    /// it, which is the dead space the panel was put there to fill.
    #[test]
    fn the_table_gets_its_rows_first_and_the_panel_gets_the_rest() {
        let state = state_with_discoveries();
        let cohorts = state.rows().len();
        assert!(cohorts > 4, "the fixture has enough cohorts to crowd a short region");
        // Tall enough for both: every cohort is drawn and so is the panel.
        let tall = painted(&state, Theme::default(), 140, 48);
        let drawn = state.rows().iter().filter(|r| tall.contains(&format!("c{}", r.id))).count();
        assert_eq!(drawn, cohorts, "a tall terminal lost a cohort row\n{tall}");
        assert!(tall.contains("DISCOVERIES"), "a tall terminal has room for both\n{tall}");

        // AND THE TABLE IS NEVER STARVED. However short the region, it keeps at
        // least a header and rows, or the panel is not drawn at all.
        for h in 24..48u16 {
            let screen = painted(&state, Theme::default(), 140, h);
            if screen.contains("DISCOVERIES") {
                let shown = state.rows().iter().filter(|r| screen.contains(&format!("c{}", r.id))).count();
                assert!(shown >= 2, "h={h}: the panel left the table {shown} rows\n{screen}");
            }
        }
    }

    /// `--dump` shows the SAME two panels the screen draws, so a dump and a
    /// terminal never disagree about what the fit found.
    #[test]
    fn the_dump_carries_the_discoveries_and_the_gene_line() {
        let state = state_with_discoveries();
        let text = dump(&state);
        assert!(text.contains("DISCOVERIES"), "{text}");
        assert!(text.contains("14 nodes tanh(exp(cos(log(x_0))))"), "{text}");
        assert!(text.contains("3.142857143 -> pi") || text.contains("-> pi"), "{text}");
        assert!(text.contains("model  f("), "{text}");
        // Newest first, as the panel shows them: the fold came last.
        let (fold, snap) = (text.find("fold").expect("a fold"), text.rfind("snap").expect("a snap"));
        assert!(fold < snap, "the dump is oldest-first: {text}");
        // Both kinds are named, so the two findings are never read as one list.
        assert_eq!(state.discoveries.iter().filter(|d| d.kind == Find::Fold).count(), 1);
    }
}
