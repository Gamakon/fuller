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
use fuller::evolve::watch::{elided, gain_text, or_dash, Gain, Liveness, PBar, Verdict, WatchState};
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

    /// THE TRAFFIC LIGHTS, and the blue beside them. They are the verdict's
    /// palette — green is a law found, red is one not found, blue is a search
    /// still running — and the p-value's, which is the same reading of the same
    /// bar.
    ///
    /// They go through `Theme` for the reason every colour here does: the
    /// terminal's own `Green` is a pale mid-green that is legible on black and
    /// washes out on white, and `Blue` is worse — a dark navy on black and a
    /// thin wash on white. The light variants are darkened until they hold
    /// against a white background, the dark ones brightened until they hold
    /// against a black one, and the both-themes render tests are what keeps
    /// either from silently regressing.
    ///
    /// GOOD, the green a cleared bar and a found law are drawn in.
    fn good(self) -> Color {
        if self.light { Color::Rgb(0, 110, 40) } else { Color::Rgb(80, 220, 120) }
    }

    /// BAD, the red a missed bar and an unfound law are drawn in.
    fn bad(self) -> Color {
        if self.light { Color::Rgb(170, 20, 20) } else { Color::Rgb(255, 105, 97) }
    }

    /// THE BLUE OF A SEARCH STILL RUNNING. Deliberately not `accent()`: the
    /// accent is the run's ordinary numbers and is cyan, and a verdict that
    /// shared it would be the one word on the screen that did not announce
    /// itself as a verdict.
    fn live(self) -> Color {
        if self.light { Color::Rgb(20, 70, 190) } else { Color::Rgb(110, 160, 255) }
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
    let verdict = state.verdict();
    let mut lines = vec![
        // THE VERDICT SURVIVES THE FALLBACK. A terminal too small for the layout
        // is still a terminal somebody is reading the answer off, and the answer
        // is the first thing on it.
        Line::from(Span::styled(
            format!(" {} ", verdict.label()),
            Style::default().fg(verdict_colour(theme, verdict)).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        )),
        Line::from(Span::styled("hff-watch · terminal too small", Style::default().fg(theme.warn()).add_modifier(Modifier::BOLD))),
        Line::from(format!("{}x{} — the full screen needs {}x{}", area.width, area.height, WIDE.0, WIDE.1)),
        Line::from(format!("gen {generation}  best HFF {}", or_dash(s.and_then(|s| s.global.best_hff), 4))),
        Line::from(p_spans(theme, state, false)),
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
        // FINISHED IS NOT AN OUTCOME. It was green, which read as "it worked" on
        // every run that merely ran out of budget; the verdict banner beside it
        // carries the green now, and this says only that records have stopped.
        Liveness::Finished => theme.accent(),
        Liveness::Stale => Color::Red,
        Liveness::Live => theme.accent(),
    }
}

/// THE VERDICT BANNER'S COLOUR: TRAFFIC LIGHTS for the two endings, blue for
/// the search still running. Green is the only thing on this screen that means
/// the fit cleared its bar, red the only thing that means it did not, and a run
/// still looking is neither — it has not failed, so it is not red, and it has
/// not succeeded, so it must not be green.
fn verdict_colour(theme: Theme, verdict: Verdict) -> Color {
    match verdict {
        Verdict::LawFound => theme.good(),
        Verdict::LawUnfound => theme.bad(),
        Verdict::Searching => theme.live(),
    }
}

/// THE COLOUR OF THE log10 p-VALUE: red while it is ABOVE the stop bar's p half,
/// green once it is at or below it. Lower is better — it is a log10 p-value.
///
/// A stream that carries no bar gets ORDINARY INK and a note, never a colour: a
/// viewer that coloured against a threshold of its own would be judging one run
/// by another run's bar the first time the engine's default moved.
fn p_colour(theme: Theme, bar: PBar) -> Color {
    match bar {
        PBar::Cleared => theme.good(),
        PBar::NotCleared => theme.bad(),
        PBar::NoBar | PBar::NoP => theme.dim(),
    }
}

/// THE log10 p-VALUE AS THE SCREEN DRAWS IT: a dim label, the number in the
/// colour of its verdict against the bar, and the note that says which bar and
/// which half. One builder, used by both layouts and by the compact fallback, so
/// a narrow terminal and a wide one never colour the same number differently.
fn p_spans(theme: Theme, state: &WatchState, roomy: bool) -> Vec<Span<'static>> {
    let bar = state.p_vs_bar();
    let p = state.snapshot.as_ref().and_then(|s| s.global.log10_p);
    let note = p_note(state, roomy);
    vec![
        Span::styled("log10 p ", Style::default().fg(theme.dim())),
        Span::styled(
            p.map_or_else(|| "—".to_string(), |p| format!("{p:.2}")),
            Style::default().fg(p_colour(theme, bar)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            if note.is_empty() { String::new() } else { format!(" {note}") },
            Style::default().fg(theme.dim()),
        ),
    ]
}

/// What the screen says about the p-value beside it: which half of the bar this
/// is, and where the bar came from. It NAMES THE HALF because both halves must
/// pass — a green p on its own is not a law, and the words are the only thing
/// stopping the colour from being read that way.
fn p_note(state: &WatchState, roomy: bool) -> String {
    // WHERE THERE IS A HEADING TO CARRY THE BAR, the number carries none: the
    // wide layout's ERROR heading names it, and repeating it on the same line
    // as the p-value is what overran the column and printed a wrong bar.
    if roomy {
        return String::new();
    }
    match (state.p_vs_bar(), state.stop_log10_p()) {
        (PBar::Cleared, Some(bar)) => format!("≤ bar {bar:.1}"),
        (PBar::NotCleared, Some(bar)) => format!("> bar {bar:.1}"),
        // A MISSING BAR IS SAID AT EVERY WIDTH. It is the one note that explains
        // why the number has no colour, and half of it would say nothing.
        (PBar::NoBar, _) => "no stop bar".to_string(),
        _ => String::new(),
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
    // THE VERDICT, in capitals, first on the line and in reverse video: it is the
    // one thing on this screen an operator wants from across a room, and it is
    // the SEARCH's answer rather than the stream's state. The badge beside it
    // still says LIVE / STALE / FINISHED, because a stale run is SEARCHING and
    // the two readings must not collapse into one word.
    let verdict = state.verdict();
    let one = Line::from(vec![
        Span::styled(
            format!(" {} ", verdict.label()),
            Style::default().fg(verdict_colour(theme, verdict)).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        ),
        Span::raw(" "),
        Span::styled(format!(" {} ", badge_text(state)), Style::default().fg(badge_colour(theme, state)).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::raw(start.map_or_else(|| "(no run_start)".to_string(), |s| s.dataset.clone())),
        Span::styled(" · seed ", Style::default().fg(theme.dim())),
        Span::raw(start.map_or_else(|| "—".to_string(), |s| s.seed.to_string())),
        // The run id is the first thing to go when the line is tight: it is the
        // only item here an operator already knows, having typed the path.
        Span::styled(if area.width >= WIDE.0 { "  run " } else { "" }, Style::default().fg(theme.dim())),
        Span::raw(match (area.width >= WIDE.0, start) {
            (true, Some(s)) => s.header.run_id.clone(),
            (true, None) => "—".to_string(),
            (false, _) => String::new(),
        }),
    ]);
    let two = Line::from(vec![
        // The banner took line one's front, so the viewer names itself here.
        Span::styled("◉ HFF-SR  ", Style::default().fg(theme.accent()).add_modifier(Modifier::BOLD)),
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
    // THE BEST-EVER LINE, which on the narrow layout also carries the p-value:
    // the ERROR column that holds it when there is room is not drawn below 118
    // columns, and a screen that dropped the one coloured number an operator is
    // watching for would be worst at the size it is most often watched at.
    // NARROW MEANS TIGHTER WORDS, not a dropped note: at 80 columns the
    // best-ever line has to hold the p-value and the reason the bar is or is not
    // there, so the unscored count loses its label rather than the note losing
    // its end — a truncated "no sto" says nothing at all.
    let mut best_ever = vec![Span::styled(
        if roomy {
            format!("best ever {}   {} unscored rows", or_dash(g.and_then(|g| g.best_ever_hff), 4), g.map_or(0, |g| g.nan_rows))
        } else {
            format!("ever {}  {} unscored  ", or_dash(g.and_then(|g| g.best_ever_hff), 3), g.map_or(0, |g| g.nan_rows))
        },
        Style::default().fg(theme.dim()),
    )];
    if !roomy {
        best_ever.extend(p_spans(theme, state, false));
    }
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
        Line::from(best_ever),
    ]);
    f.render_widget(best, columns[0]);
    if roomy {
        let quality = Paragraph::new(vec![
            // THE BAR GOES IN THE HEADING, not beside the number. The ERROR
            // column is about 40 columns at 120 wide, and a `Paragraph` clips
            // rather than wraps — the bar printed after the p-value came out as
            // "≤ p bar -19", a DIFFERENT BAR from the one the run set, which is
            // worse than not printing it. The heading has the room, the number
            // keeps its colour, and the two are one line apart.
            Line::from(Span::styled(
                match state.stop_log10_p() {
                    Some(bar) => format!("ERROR · stop bar p ≤ {bar:.1}"),
                    None => "ERROR · no stop bar in stream".to_string(),
                },
                Style::default().fg(theme.dim()),
            )),
            Line::from(format!("train 1-R² {train}   val 1-R² {val}")),
            // THE p-VALUE IS ITS OWN SPAN, because it is the only number on this
            // line that carries a colour: the rest is dim furniture, and the p
            // says whether the fit has cleared half of its stop bar.
            Line::from({
                let mut spans = vec![Span::styled(
                    format!("mse {}  ", or_dash(g.and_then(|g| g.mse_train), 2)),
                    Style::default().fg(theme.dim()),
                )];
                spans.extend(p_spans(theme, state, true));
                spans.push(Span::styled(
                    format!("  depth {}  head {}", g.map_or(0, |g| g.t_depth), g.map_or(0, |g| g.vhead)),
                    Style::default().fg(theme.dim()),
                ));
                spans
            }),
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
    // THE HIGHLIGHT AND THE DETAIL PANE NAME THE SAME ROW. The pane follows the
    // table until the operator picks, so the wash has to follow it too — a
    // highlight on no row while the pane described one would be two panels
    // disagreeing about what is selected.
    let highlighted = state.detail_id();
    let body: Vec<Row> = rows
        .iter()
        .map(|r| {
            let selected = highlighted == Some(r.id);
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
    // THE TRUE TOTALS, not the ring's length. The list below is capped and
    // collapses repeats, so counting IT reported the cap: a finished stream
    // holding 597 snaps said "128 literals snapped", which is the size of the
    // ring. `state.found` counts every event as it arrives.
    let (snaps, folds, reduces) =
        (state.found[Find::Snap as usize], state.found[Find::Fold as usize], state.found[Find::Reduce as usize]);
    let plural = |n: u64, one: &str, many: &str| if n == 1 { one.to_string() } else { many.to_string() };
    // AND THE REDUCTION'S COUNT IS A DASH UNTIL THE FINAL FORM HAS RUN. It only
    // happens once the fit returns, so before that a zero would mean "not yet"
    // while reading as "none" — and a fit killed mid-run never runs it at all.
    // The codebase's own rule: a metric that is not emitted shows `—` and a reason.
    let dropped = if state.tidy_reported { format!("{reduces} dropped") } else { "— dropped (final form pending)".to_string() };
    // THE HEADING MUST FIT THE BORDER IT SITS IN. Three counts and their nouns
    // spelled out is ~95 characters, and ratatui silently truncates a title that
    // overruns — at 80 columns the fold count simply vanished, which is the one
    // number this panel was rebuilt to show. The long form is drawn where there
    // is room for it and the short one where there is not.
    let long = format!(
        " DISCOVERIES · {snaps} {} snapped into genes · {folds} {} folded · {dropped} ",
        plural(snaps, "literal", "literals"),
        plural(folds, "subtree", "subtrees")
    );
    let short = format!(" DISCOVERIES · {snaps} snapped · {folds} folded · {dropped} ");
    let title = if long.chars().count() <= area.width.saturating_sub(2) as usize { long } else { short };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    if state.discoveries.is_empty() {
        // Nothing found is NOT nothing to say: snap and the fold are both
        // switched on separately, and an empty panel that does not explain
        // itself reads as a broken one.
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("nothing yet", Style::default().fg(theme.dim()))),
                Line::from(Span::styled(
                    "snaps and folds arrive through the run · drops once, after it ends",
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
            // A snap is the accent — it went INTO the population. A FOLD IS RED
            // (Andrew: "in a red colour if possible"): it is the operator taking
            // a blob of dead operators out of a gene, and it should read as the
            // loudest thing on the panel. `theme.bad()` is the red that is already
            // proved legible on both terminals by the both-themes render tests; a
            // second red would be a colour nothing else uses. A reduction keeps
            // the warn colour: it is the quieter half, and it must not be mistaken
            // for a fold at a glance.
            let (tag, colour) = match d.kind {
                Find::Snap => ("snap", theme.accent()),
                Find::Fold => ("fold", theme.bad()),
                Find::Reduce => ("drop", theme.warn()),
            };
            let size = d.nodes.map_or_else(String::new, |n| format!("{n} nodes "));
            let body = format!("{size}{} → {}{}", d.what, d.became, d.times());
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
    // The pane always shows a cohort that is ON the table: the operator's pick
    // while it is there, the top row once it is not. An empty table has neither.
    let Some(r) = state.selected_row() else {
        f.render_widget(
            Paragraph::new(Span::styled("no cohorts in the table", Style::default().fg(theme.dim()))).block(block),
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
    // A PICK THAT WAS DROPPED SAYS SO. The pane has fallen back to the top row
    // because the cohort the operator chose has left the table, and a screen
    // that swapped one cohort for another without a word would read as the
    // detail pane showing the wrong thing.
    if let Some(chosen) = state.selected.filter(|id| *id != r.id) {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("c{chosen} has left the table — showing the top row"),
            Style::default().fg(theme.warn()),
        )));
    }
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
/// AN EXPRESSION, INDENTED BY ITS OWN BRACKETS — the model pane's readable form.
///
/// A recovered model is one line of several hundred characters. Wrapped it is a
/// wall; scrolled sideways it is a slot. Neither shows the SHAPE, which is the
/// thing an operator is reading it for: which factor multiplies which sum, how
/// deep the nesting goes, where the constant sits.
///
/// So it is broken the way a nested document is: a bracket that opens starts an
/// indented block, a comma or a top-level operator starts a sibling line, and a
/// bracket that closes ends it. The tokens are untouched — this only adds line
/// breaks and leading spaces, so what is on screen is still exactly the model.
///
/// SHORT SPANS STAY ON ONE LINE. A bracket whose whole contents fit in the width
/// left to it is not split: `(x_0*x_1)` is more readable as itself than as five
/// lines, and a printer that splits everything turns a small model into a column
/// of single characters.
fn indent_expression(src: &str, width: usize) -> String {
    // The matching close for every open bracket, so a span's length is known
    // before deciding whether to break it.
    let bytes: Vec<char> = src.chars().collect();
    let mut close_of = vec![usize::MAX; bytes.len()];
    let mut stack = Vec::new();
    for (i, c) in bytes.iter().enumerate() {
        match c {
            '(' => stack.push(i),
            ')' => {
                if let Some(open) = stack.pop() {
                    close_of[open] = i;
                }
            }
            _ => {}
        }
    }
    let mut out = String::new();
    let mut line = String::new();
    let mut depth = 0usize;
    let mut i = 0usize;
    // A line is only ever flushed here, so the indent and the newline stay
    // together and a stray break cannot lose the leading spaces.
    macro_rules! flush {
        () => {
            if !line.trim().is_empty() {
                out.push_str(&"  ".repeat(depth));
                out.push_str(line.trim_end());
                out.push('\n');
            }
            line.clear();
        };
    }
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            '(' => {
                // The whole span, brackets included. If it fits in what is left
                // of the width it is copied verbatim and never split.
                let end = close_of[i];
                let fits = end != usize::MAX && (end - i + 1) + depth * 2 + line.chars().count() <= width;
                if fits {
                    line.extend(&bytes[i..=end]);
                    i = end + 1;
                    continue;
                }
                line.push('(');
                flush!();
                depth += 1;
            }
            ')' => {
                flush!();
                depth = depth.saturating_sub(1);
                line.push(')');
            }
            // A separator at THIS level ends the sibling. Operators inside a
            // span that fitted were consumed above and never reach here.
            '+' | '-' | '*' | '/' | ',' => {
                line.push(c);
                flush!();
            }
            _ => line.push(c),
        }
        i += 1;
    }
    flush!();
    if out.is_empty() { src.to_string() } else { out }
}

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
    // INDENTED BY DEFAULT. A model is read for its shape, and the shape is what
    // one long line destroys; `w` gives back the raw line for a copy or a diff.
    let pretty = !ui.model_wrap;
    let shown = if pretty { indent_expression(&body, rows[2].width as usize) } else { body };
    let lines = shown.lines().count() as u16;
    let page = rows[2].height;
    // Scrolling is now DOWN the indented text rather than sideways along one
    // line, and it stops at the end instead of running off into blank screen.
    let max_scroll = lines.saturating_sub(page);
    let offset = ui.model_scroll.min(max_scroll);
    let text = Paragraph::new(shown);
    let text = if pretty { text.scroll((offset, 0)) } else { text.wrap(Wrap { trim: false }) };
    f.render_widget(text, rows[2]);
    f.render_widget(
        Paragraph::new(Span::styled(
            match &ui.copied {
                Some(path) => format!("written to {path}"),
                None if pretty => format!(
                    "indented · line {}-{} of {} · h/l scroll · w raw line · y writes a file",
                    offset + 1,
                    (offset + page).min(lines),
                    lines
                ),
                None => "raw line, wrapped · w indents it · y writes a file".to_string(),
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
    // THE VERDICT FIRST, in capitals, with what it rests on. `--dump` is how the
    // answer is read out of a log file, so it opens with the answer.
    let verdict = state.verdict();
    out.push_str(&format!("{}  ({})\n", verdict.label(), verdict.because()));
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
            // The p-value carries its verdict against the bar as WORDS here: a
            // dump has no colour, and the decision the screen draws in green or
            // red must still be readable down a pipe.
            let note = p_note(state, false);
            out.push_str(&format!(
                "  best hff {}   1-R2 train {}   val {}   log10 p {}{}\n",
                or_dash(sn.global.best_hff, 6),
                or_dash(sn.global.r2_train.map(|r| 1.0 - r), 3),
                or_dash(sn.global.r2_val.map(|r| 1.0 - r), 3),
                or_dash(sn.global.log10_p, 2),
                match state.p_vs_bar() {
                    // A CLEARED p NAMES THE OTHER HALF. Both halves must pass
                    // before the fit calls a model a law, and a dump that said
                    // only "CLEARED" would invite exactly the reading the
                    // verdict line above it is there to prevent.
                    PBar::Cleared => format!("  CLEARED {note} (the 1-R² half must pass too)"),
                    PBar::NotCleared => format!("  ABOVE {note}"),
                    PBar::NoBar => format!("  ({note})"),
                    PBar::NoP => String::new(),
                }
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
    // The same heading the screen draws, TRUE TOTALS and all: a dump is how the
    // panel is checked without a tty, so it must not report a different number.
    use fuller::evolve::watch::Find;
    let (snaps, folds, reduces) =
        (state.found[Find::Snap as usize], state.found[Find::Fold as usize], state.found[Find::Reduce as usize]);
    let dropped = if state.tidy_reported { format!("{reduces} dropped") } else { "— dropped (the final form has not run)".to_string() };
    out.push_str(&format!("\n  DISCOVERIES · {snaps} snapped · {folds} folded · {dropped}\n"));
    if state.discoveries.is_empty() {
        out.push_str("  (none yet · snaps and folds arrive through the run, drops once after it ends)\n");
    }
    for d in state.discoveries.iter().rev().take(12) {
        let tag = match d.kind {
            Find::Snap => "snap",
            Find::Fold => "fold",
            Find::Reduce => "drop",
        };
        let size = d.nodes.map_or_else(String::new, |n| format!("{n} nodes "));
        out.push_str(&format!("  gen {:>6}  {tag}  {size}{} -> {}{}\n", d.generation, d.what, d.became, d.times()));
    }
    // THE DETAIL PANE, as the screen shows it. It is here because this is how
    // the pane is checked from a log file, and because the cohort it names is a
    // thing that has been wrong: it used to latch onto the first snapshot's best
    // and go on naming it long after that cohort died.
    out.push_str(&format!(
        "\n  COHORT DETAIL  {}\n",
        match (state.detail_id(), state.selected_row()) {
            (Some(id), Some(r)) => format!(
                "c{id}  born {}  rows {}  best {}{}",
                r.birth_generation,
                r.rows,
                or_dash(r.best_hff, 6),
                match state.selected.filter(|s| *s != id) {
                    Some(chosen) => format!("  (c{chosen} has left the table)"),
                    None => String::new(),
                }
            ),
            _ => "no cohorts in the table".to_string(),
        }
    ));
    out.push_str(&format!(
        "\n  model  {}\n",
        state.gene_line.as_deref().map_or("(no model record yet)", |l| l)
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuller::evolve::telemetry::{parse_stream, Record};
    use fuller::evolve::watch::Find;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// THE INDENTED MODEL KEEPS EVERY TOKEN. A pretty-printer that dropped or
    /// reordered a character would be showing an equation the fit never found,
    /// which is worse than an unreadable one — so the round trip is asserted on
    /// a REAL recovered model, not a toy.
    #[test]
    fn the_indented_model_is_the_same_expression() {
        let model = "(48.999999646850476*((x_0/(x_0 - (x_0**2)))*(x_2/((-49.0)/x_1))))";
        let pretty = indent_expression(model, 40);
        assert!(pretty.contains('\n'), "a model wider than the pane is broken over lines");
        let stripped: String = pretty.chars().filter(|c| !c.is_whitespace()).collect();
        let original: String = model.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(stripped, original, "indenting adds only breaks and spaces");
    }

    /// A SHORT MODEL IS LEFT ALONE. `(x_0*x_1)` split over five lines is less
    /// readable than itself, and most recovered laws are this short.
    #[test]
    fn a_model_that_fits_is_not_split() {
        assert_eq!(indent_expression("(x_0*x_1)", 40).trim(), "(x_0*x_1)");
    }

    /// The deep case: nesting shows as increasing indent, which is the whole
    /// point of the pane.
    #[test]
    fn nesting_shows_as_indent() {
        let pretty = indent_expression("(a+(b*(c+(d*(e+f)))))", 12);
        let depths: Vec<usize> = pretty.lines().map(|l| l.len() - l.trim_start().len()).collect();
        assert!(depths.iter().max() > Some(&0), "deeper terms are indented further");
        let stripped: String = pretty.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(stripped, "(a+(b*(c+(d*(e+f)))))");
    }

    const FIXTURE: &str = include_str!("../../tests/fixtures/telemetry_v1.jsonl");

    /// A REAL RECORDING that has discoveries in it: a 30-second bacres1 fit
    /// (seed 7014) that snapped 166 literals into genes and folded one
    /// near-constant subtree, trimmed. `FIXTURE` predates both kinds, so a panel
    /// drawn from it would be a panel drawn from a stream with none of what it
    /// shows — which proves nothing.
    const DISCOVERIES: &str = include_str!("../../tests/fixtures/telemetry_v1_discoveries.jsonl");

    /// A REAL RECORDING OF THE FOLD OPERATOR: a 400-generation bacres1 fit (seed
    /// 7014) trimmed to the records that matter here — folds that arrived DURING
    /// the search carrying the row they landed in, the leave-one-out's drops after
    /// `run_end`, and the note that says the final form has run.
    ///
    /// `DISCOVERIES` predates all three: its one fold has `row: null` because the
    /// fold could only happen after the fit returned, which is the bug this
    /// fixture exists to show is gone.
    const FOLD_OPERATOR: &str = include_str!("../../tests/fixtures/telemetry_v1_fold_operator.jsonl");

    fn state_with_fold_operator() -> WatchState {
        let (records, bad) = parse_stream(FOLD_OPERATOR);
        assert_eq!(bad, 0, "the recording is clean");
        let mut state = WatchState::new(false);
        for r in records {
            state.apply_record(r);
        }
        state
    }

    fn state_with_discoveries() -> WatchState {
        let (records, bad) = parse_stream(DISCOVERIES);
        assert_eq!(bad, 0, "the recording is clean");
        let mut state = WatchState::new(false);
        state.bad_lines = bad;
        for r in records {
            state.apply_record(r);
        }
        assert!(
            state.discoveries.iter().any(|d| d.kind == Find::Snap) && state.discoveries.iter().any(|d| d.kind == Find::Fold),
            "the recording must carry both kinds"
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

    /// THE COLOUR A RUN OF TEXT WAS DRAWN IN. `painted` gives the characters
    /// only, and the whole point of the p-value's colour is that it is not a
    /// character — so this finds the run on the screen and reads the foreground
    /// off the cell its first character landed in.
    fn colour_of(state: &WatchState, theme: Theme, w: u16, h: u16, needle: &str) -> Option<Color> {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("a test terminal");
        let ui = Ui { theme, ..Ui::default() };
        terminal.draw(|f| draw(f, state, &ui)).expect("the frame draws");
        let buffer = terminal.backend().buffer().clone();
        let rows: Vec<String> = buffer
            .content()
            .chunks(w as usize)
            .map(|row| row.iter().map(ratatui::buffer::Cell::symbol).collect::<String>())
            .collect();
        for (y, row) in rows.iter().enumerate() {
            if let Some(byte_at) = row.find(needle) {
                let x = row[..byte_at].chars().count() as u16;
                return Some(buffer[(x, y as u16)].fg);
            }
        }
        None
    }

    /// A state built from the fixture with its run_start, its p-value and its
    /// ending set to whatever the case under test needs. The recorded streams
    /// predate the stop bar, so the bar has to be put there to test against it —
    /// and that is the point: a stream WITHOUT one is a case of its own.
    fn state_with(bar: Option<f64>, p: Option<f64>, stopped_by: Option<&str>) -> WatchState {
        let (records, _) = parse_stream(FIXTURE);
        let mut state = WatchState::new(false);
        for r in records {
            state.apply_record(r);
        }
        state.start.as_mut().expect("a run_start").stop_log10_p = bar;
        state.snapshot.as_mut().expect("a frame").global.log10_p = p;
        match stopped_by {
            Some(word) => state.end.as_mut().expect("a run_end").stopped_by = word.to_string(),
            None => state.end = None,
        }
        state
    }

    /// THE VERDICT BANNER IS ON THE SCREEN, IN CAPITALS, at every size and in
    /// both themes — and it is coloured by the outcome, not by the fact that the
    /// run ended.
    #[test]
    fn the_verdict_banner_draws_in_capitals_at_every_size_and_theme() {
        for theme in [Theme { light: false }, Theme { light: true }] {
            // TRAFFIC LIGHTS, and blue for the search that is still running.
            let cases = [
                (Some("early_stop"), "LAW FOUND", theme.good()),
                (Some("n_gen"), "LAW UNFOUND", theme.bad()),
                (Some("time"), "LAW UNFOUND", theme.bad()),
                (None, "SEARCHING", theme.live()),
            ];
            for (stopped_by, word, want) in cases {
                let state = state_with(Some(-19.0), Some(-13.0), stopped_by);
                for (w, h) in [(120, 35), (160, 50), (80, 24)] {
                    let screen = painted(&state, theme, w, h);
                    assert!(screen.contains(word), "{w}x{h} light={}: no banner\n{screen}", theme.light);
                    assert_eq!(
                        colour_of(&state, theme, w, h, word),
                        Some(want),
                        "{w}x{h} light={}: {word} is the wrong colour",
                        theme.light
                    );
                    // The layout still holds: every row is exactly the width.
                    for line in screen.lines() {
                        assert_eq!(line.chars().count(), w as usize, "{w}x{h}: a row is not the terminal's width");
                    }
                }
            }
            // THE THREE VERDICTS ARE THREE COLOURS, and none of them is the
            // accent the run's ordinary numbers wear — a banner that shared a
            // colour with the HFF beside it would not announce itself.
            let palette = [theme.good(), theme.bad(), theme.live()];
            for (i, a) in palette.iter().enumerate() {
                for b in &palette[i + 1..] {
                    assert_ne!(a, b, "two verdicts share a colour on light={}", theme.light);
                }
                assert_ne!(*a, theme.accent(), "a verdict wears the accent on light={}", theme.light);
                assert_ne!(*a, theme.dim(), "a verdict wears the dim ink on light={}", theme.light);
            }
        }
    }

    /// STALE AND SEARCHING SIT SIDE BY SIDE. A live run whose stream went quiet
    /// still says SEARCHING — the fit has not ended — and the badge still says
    /// STALE, so an operator can tell a quiet stream from a finished fit.
    #[test]
    fn a_stale_run_shows_both_searching_and_stale() {
        let (records, _) = parse_stream(FIXTURE);
        let mut state = WatchState::new(true);
        for r in records.into_iter().filter(|r| !matches!(r, Record::RunEnd(_))) {
            state.apply_record(r);
        }
        state.last_record = Some(std::time::Instant::now() - std::time::Duration::from_secs(60));
        let screen = painted(&state, Theme::default(), 120, 35);
        assert!(screen.contains("SEARCHING"), "a stale run lost its verdict\n{screen}");
        assert!(screen.contains("STALE"), "the verdict erased the stale badge\n{screen}");
    }

    /// THE p-VALUE IS RED ABOVE THE BAR AND GREEN AT OR BELOW IT, on both
    /// layouts and in both themes — and a stream with NO bar is drawn in
    /// ordinary ink and says so, never coloured against a threshold the viewer
    /// supplied for itself.
    #[test]
    fn the_p_value_is_red_above_the_bar_and_green_once_it_has_cleared_it() {
        for theme in [Theme { light: false }, Theme { light: true }] {
            // The p-value's own column is only drawn at 118 columns and up; the
            // narrow layout carries it on the best-ever line, so both are here.
            for (w, h) in [(120, 35), (160, 50), (80, 24)] {
                // ABOVE the bar: red.
                let above = state_with(Some(-19.0), Some(-13.04), Some("n_gen"));
                assert_eq!(
                    colour_of(&above, theme, w, h, "-13.04"),
                    Some(theme.bad()),
                    "{w}x{h} light={}: a p above the bar is not red\n{}",
                    theme.light,
                    painted(&above, theme, w, h)
                );
                // AT OR BELOW it: green. Exactly on the bar counts, as the
                // engine's own `<=` does.
                for value in ["-21.40", "-19.00"] {
                    let cleared = state_with(Some(-19.0), Some(value.parse().expect("a number")), Some("early_stop"));
                    assert_eq!(
                        colour_of(&cleared, theme, w, h, value),
                        Some(theme.good()),
                        "{w}x{h} light={}: p {value} has cleared the bar and is not green",
                        theme.light
                    );
                }
                // NO BAR IN THE STREAM: ordinary ink, and the screen says why.
                let barless = state_with(None, Some(-13.04), Some("n_gen"));
                assert_eq!(
                    colour_of(&barless, theme, w, h, "-13.04"),
                    Some(theme.dim()),
                    "{w}x{h} light={}: a viewer coloured against a bar it invented",
                    theme.light
                );
                let screen = painted(&barless, theme, w, h);
                assert!(screen.contains("no stop bar"), "{w}x{h}: it did not say the bar is missing\n{screen}");
                // AND THE BAR'S VALUE IS ON SCREEN WHOLE wherever there is one.
                // A `Paragraph` clips rather than wraps, and a bar printed as
                // "-19" or "-1" where the run set -19.0 is a DIFFERENT bar —
                // the failure this assertion exists to catch.
                for st in [&above, &state_with(Some(-19.0), Some(-21.40), Some("early_stop"))] {
                    let screen = painted(st, theme, w, h);
                    assert!(screen.contains("-19.0"), "{w}x{h} light={}: the bar was cut short\n{screen}", theme.light);
                }
                // And the layout holds at every one of these.
                for line in screen.lines() {
                    assert_eq!(line.chars().count(), w as usize, "{w}x{h}: a row is not the terminal's width");
                }
            }
        }
    }

    /// THE BANNER SURVIVES THE NARROW FALLBACK and the fallback still does not
    /// panic. Below 70x20 there is no layout, but there is still an answer.
    #[test]
    fn the_compact_fallback_keeps_the_verdict_and_never_panics() {
        let state = state_with(Some(-19.0), Some(-13.0), Some("early_stop"));
        let small = painted(&state, Theme::default(), 69, 19);
        assert!(small.contains("LAW FOUND"), "the fallback dropped the verdict\n{small}");
        assert!(small.contains("terminal too small"), "{small}");
        // And every size from the minimum up still draws, banner and all.
        for h in 20..40u16 {
            for w in [70u16, 79, 80, 117, 118, 119, 120, 121] {
                let screen = painted(&state, Theme::default(), w, h);
                assert!(screen.contains("LAW FOUND"), "{w}x{h} lost the banner\n{screen}");
                for line in screen.lines() {
                    assert_eq!(line.chars().count(), w as usize, "{w}x{h}: a row is not the terminal's width");
                }
            }
        }
    }

    /// `--dump` CARRIES THE VERDICT AND THE BAR DECISION, because a dump has no
    /// colour and it is how the answer is read out of a log file.
    #[test]
    fn the_dump_opens_with_the_verdict_and_says_where_the_p_value_sits() {
        // Cleared, and the fit stopped early: the dump's first word is the answer.
        let found = state_with(Some(-19.0), Some(-21.4), Some("early_stop"));
        let text = dump(&found);
        assert!(text.starts_with("LAW FOUND"), "{text}");
        assert!(text.contains("early_stop"), "{text}");
        assert!(text.contains("CLEARED"), "the dump did not place the p-value\n{text}");
        assert!(text.contains("bar -19.0"), "the dump did not name the bar\n{text}");
        // The colour's meaning survives as words: BOTH halves are named, so a
        // cleared p is never read on its own as a law.
        assert!(text.contains("1-R²"), "the dump did not name the other half\n{text}");

        // Above the bar, out of budget: the other verdict and the other word.
        let unfound = state_with(Some(-19.0), Some(-13.0), Some("n_gen"));
        let text = dump(&unfound);
        assert!(text.starts_with("LAW UNFOUND"), "{text}");
        assert!(text.contains("ABOVE"), "{text}");

        // Still running: SEARCHING, and no claim either way.
        let searching = state_with(Some(-19.0), Some(-13.0), None);
        let text = dump(&searching);
        assert!(text.starts_with("SEARCHING"), "{text}");
        assert!(!text.contains("LAW"), "a running fit was given a verdict\n{text}");

        // THE DETAIL PANE NAMES A LIVE COHORT, and a dump is where that is
        // checked from a log file.
        let first = found.rows().first().expect("a table").id;
        assert!(dump(&found).contains(&format!("COHORT DETAIL  c{first}")), "{}", dump(&found));

        // A stream with no bar says so rather than judging against one.
        let barless = state_with(None, Some(-13.0), Some("n_gen"));
        let text = dump(&barless);
        assert!(text.contains("no stop bar"), "{text}");
        assert!(!text.contains("CLEARED") && !text.contains("ABOVE"), "it judged without a bar\n{text}");
    }



    /// THE DETAIL PANE SHOWS A COHORT THAT IS IN THE TABLE, not one that died
    /// thousands of generations ago. The pane used to latch onto the best cohort
    /// of the first snapshot it ever saw, so a long fit's viewer read
    /// "c0 · not in the current table" beside a table of live cohorts.
    #[test]
    fn the_detail_pane_shows_a_live_cohort_not_a_latched_dead_one() {
        let state = state_with(Some(-19.0), Some(-13.0), Some("n_gen"));
        let first = state.rows().first().expect("a table").id;
        let screen = painted(&state, Theme::default(), 140, 40);
        assert!(screen.contains("COHORT DETAIL"), "{screen}");
        assert!(screen.contains(&format!("c{first}")), "the pane does not name the table's row\n{screen}");
        assert!(
            !screen.contains("not in the current table"),
            "an untouched pane claimed its cohort had gone\n{screen}"
        );
        // A CHOSEN COHORT THAT HAS GONE RESETS TO THE TOP ROW, and the pane
        // says the pick was dropped rather than swapping one cohort for another
        // without a word. It never draws a cohort that is not on the table.
        let mut chosen = state;
        chosen.selected = Some(999_999);
        let screen = painted(&chosen, Theme::default(), 140, 40);
        assert!(
            !screen.contains("not in the current table"),
            "a dead cohort is still pinned to the pane\n{screen}"
        );
        assert!(screen.contains(&format!("c{first}")), "the pane did not fall back to the top row\n{screen}");
        assert!(screen.contains("has left the table"), "the dropped pick was swapped silently\n{screen}");
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
                assert!(screen.contains("2 nodes"), "{w}x{h}: the fold's size is not shown\n{screen}");
                // The recording's snapped forms are on the screen as FORMS, not
                // as the numbers they compute: `sqrt3` and `phi`, not 3.2114.
                assert!(screen.contains("sqrt3") || screen.contains("g_earth"), "{w}x{h}: no snapped form\n{screen}");
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

    /// ALL THREE KINDS DRAW, at every size and in both themes — and a fold is
    /// visibly a fold, a drop visibly a drop.
    ///
    /// The fold is RED (`theme.bad()`) because it is the operator taking a blob of
    /// dead operators out of a live gene, which is the loudest news the panel
    /// carries; the drop keeps the warn colour so the two are never read as one
    /// finding. Both reds are the ones the verdict banner already uses, so they
    /// are the ones these sizes and themes have always proved legible.
    #[test]
    fn the_fold_and_the_drop_draw_apart_at_every_size_and_in_both_themes() {
        let state = state_with_fold_operator();
        assert!(state.discoveries.iter().any(|d| d.kind == Find::Fold), "the recording has no fold");
        assert!(state.discoveries.iter().any(|d| d.kind == Find::Reduce), "the recording has no reduction");
        for theme in [Theme { light: false }, Theme { light: true }] {
            for (w, h) in [(120, 35), (160, 50), (80, 24)] {
                let screen = painted(&state, theme, w, h);
                assert!(screen.contains("DISCOVERIES"), "{w}x{h} light={}: no panel\n{screen}", theme.light);
                assert!(screen.contains("fold"), "{w}x{h}: the fold is not on screen\n{screen}");
                assert!(screen.contains("drop"), "{w}x{h}: the drop is not on screen\n{screen}");
                // A reduction says the subtree WENT. Its `value` is the mean it
                // was held at to prove it could go, not something the model now
                // carries, so it must never be drawn as an arrow's target.
                assert!(screen.contains("dropped"), "{w}x{h}: a drop did not say it was dropped\n{screen}");
                // AND THE HEADING'S NUMBERS SURVIVE THE WIDTH. A title that
                // overruns its border is truncated silently, and at 80 columns the
                // fold count — the one number this panel was rebuilt to show —
                // simply vanished off the end of the long spelling.
                // Either spelling, but the NUMBER must be there: the heading is
                // drawn long where it fits and short where it does not.
                let folds = state.found[Find::Fold as usize];
                assert!(
                    screen.contains(&format!("{folds} folded")) || screen.contains(&format!("{folds} subtrees folded")),
                    "{w}x{h}: the fold count was truncated away\n{screen}"
                );
                for line in screen.lines() {
                    assert_eq!(line.chars().count(), w as usize, "{w}x{h}: a row is not the terminal's width");
                }
            }
        }
        // AND THE TWO REDS ARE DIFFERENT COLOURS, in both themes: a panel that
        // drew a fold and a drop identically would be showing one finding twice.
        for theme in [Theme { light: false }, Theme { light: true }] {
            assert_ne!(theme.bad(), theme.warn(), "light={}: the fold and the drop share a colour", theme.light);
            assert_ne!(theme.bad(), theme.accent(), "light={}: the fold and the snap share a colour", theme.light);
        }
    }

    /// THE HEADING COUNTS WHAT THE STREAM SENT, not what the ring kept.
    ///
    /// `discoveries` is capped at `DISCOVERIES` and collapses repeats, so counting
    /// IT reported the cap: the finished bacres1 recording holds 597 snap events
    /// and the panel said "128 literals snapped", which is the size of the ring
    /// and not a measurement of anything.
    #[test]
    fn the_heading_reports_the_true_totals_and_not_the_rings_cap() {
        let mut state = state_with_fold_operator();
        let sent = state.found;
        // More snaps than the ring can hold, each distinct so none collapses.
        let base = state.discoveries.last().cloned().expect("a discovery");
        for g in 0..(fuller::evolve::watch::DISCOVERIES as u32 + 40) {
            state.apply_record(Record::Event(fuller::evolve::telemetry::Event {
                header: fuller::evolve::telemetry::Header {
                    schema_version: fuller::evolve::telemetry::SCHEMA_VERSION,
                    run_id: "r".into(),
                    seq: 9000 + u64::from(g),
                    timestamp_utc: fuller::evolve::telemetry::now_utc(),
                    elapsed_ms: 0,
                    generation: 500 + g,
                },
                kind: fuller::evolve::telemetry::EventKind::Snap,
                message: format!("snap {g}"),
                cohort: None,
                value: Some(f64::from(g)),
                before: Some(f64::from(g) + 0.5),
                detail: Some(format!("pi/{g}")),
                nodes: None,
                row: Some(g),
            }));
        }
        let _ = base;
        let added = u64::from(fuller::evolve::watch::DISCOVERIES as u32 + 40);
        assert_eq!(state.found[Find::Snap as usize], sent[Find::Snap as usize] + added, "the total did not follow the stream");
        assert_eq!(state.discoveries.len(), fuller::evolve::watch::DISCOVERIES, "the ring did not stay bounded");
        // The heading says the TOTAL, which is larger than the ring.
        let screen = painted(&state, Theme::default(), 160, 50);
        let total = state.found[Find::Snap as usize];
        assert!(total > fuller::evolve::watch::DISCOVERIES as u64, "the test did not overflow the ring");
        assert!(screen.contains(&format!("{total} literals snapped")), "the heading reported the cap, not the total\n{screen}");
    }

    /// A ZERO THE VIEWER DOES NOT HAVE IS A DASH. The reduction only runs once the
    /// fit returns, so before the engine says it has run, a count of zero would
    /// read as "none found" when it means "not computed yet" — and a fit killed
    /// mid-run never computes it at all, which is why a 60,000-generation stream
    /// showed nothing there for its whole life.
    #[test]
    fn the_drop_count_is_a_dash_until_the_final_form_has_run() {
        // The finished recording that predates the summary note: it genuinely
        // does not know, and must say so.
        let old = state_with_discoveries();
        assert!(!old.tidy_reported, "the old recording carries a summary it should not");
        let screen = painted(&old, Theme::default(), 160, 50);
        assert!(screen.contains("— dropped"), "a count it does not have was printed as a number\n{screen}");
        assert!(screen.contains("final form pending"), "the dash gave no reason\n{screen}");
        // And the recording that DOES carry the note prints the number, even
        // though the number is small.
        let new = state_with_fold_operator();
        assert!(new.tidy_reported, "the note was not seen");
        let screen = painted(&new, Theme::default(), 160, 50);
        assert!(!screen.contains("— dropped"), "a known count was printed as a dash\n{screen}");
        let reduces = new.found[Find::Reduce as usize];
        assert!(screen.contains(&format!("{reduces} dropped")), "the known count is not on screen\n{screen}");
    }

    /// THE FOLD ARRIVES DURING THE SEARCH, which is the whole point of the
    /// operator: its events carry the row they landed in and a generation before
    /// the fit ended, where the old fold could only ever be stamped with the last
    /// generation of the run.
    #[test]
    fn the_folds_in_the_recording_happened_mid_fit() {
        let state = state_with_fold_operator();
        let end = state.end.as_ref().expect("a run_end").generations;
        let folds: Vec<_> = state.discoveries.iter().filter(|d| d.kind == Find::Fold).collect();
        assert!(!folds.is_empty(), "no folds in the recording");
        assert!(folds.iter().any(|d| d.generation < end), "every fold was stamped at the end of the run: {folds:?}");
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
        assert!(text.contains("2 nodes exp((-5.0))"), "{text}");
        assert!(text.contains("-> ((3.0*sqrt3)/(1.0*phi))"), "{text}");
        assert!(text.contains("model  f("), "{text}");
        // Newest first, as the panel shows them: the fold came last.
        let (fold, snap) = (text.find("fold").expect("a fold"), text.rfind("snap").expect("a snap"));
        assert!(fold < snap, "the dump is oldest-first: {text}");
        // Both kinds are named, so the two findings are never read as one list.
        assert_eq!(state.discoveries.iter().filter(|d| d.kind == Find::Fold).count(), 1);
        // AND THE REPEAT IS COLLAPSED. The recording holds four consecutive
        // copies of one substitution — selection had copied the gene across
        // four rows — and they are one line with a count, not four lines.
        assert!(text.contains(" ×4"), "a repeated finding was not collapsed:\n{text}");
    }
}
