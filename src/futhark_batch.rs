//! THE BATCH HOST: a whole generation of the signature family's programs (`crate::futhark_sig`) in ONE compiled
//! program and ONE process.
//!
//! The per-candidate path compiles each candidate on its own (the branch compiler, the WGSL port, `cc` of the
//! generated C and Futhark's runtime, the link) and runs it in a process of its own (the device's set-up, Metal's
//! pipeline compiles, the runs). Here a batch of programs is one Futhark source, each program its own entry point:
//!
//! * a member's QUALITY entry `q<i>` is [`program_cases_named`]'s text (the program as a `#[noinline]` function
//!   `kq<i>` and an entry applying it to every stored case), the per-candidate quality program but for its two names,
//!   so the compiler sees the same program;
//! * a member's TIMED entry `t<i>` is [`program_named`]'s text, the plain program with the declared signature, the
//!   per-candidate timed program but for its name;
//! * the published program, when anything is timed against it, is the entry `yard` (its own text, its `main` so
//!   named), placed last so its own definitions cannot shadow a member's names.
//!
//! The branch compiler builds it as a LIBRARY (`--library`: no `main`, the entry points as C functions), and a C
//! driver written here for the batch calls them: it sets the device up once, reads the cases and the timing input
//! once, and runs a list of commands, writing every result to one file. Only the programs not seen before enter a
//! batch (the caller keeps each program's text-keyed cache).
//!
//! **Timing inside the process.** A `time` command runs a timed entry `runs + 1` times on the timing input, each run
//! as Futhark's command line does it (`-r`: the inputs made, a sync, the call, a sync, the inputs and the output
//! freed), and reports each run's GPU nanoseconds: the sum of the passes the shim timed during that run, read with the
//! shim's pass query (`futhark-wgpu-shim/src/report.rs`). The first run is the warm-up. A run with no timed pass (a
//! program with no GPU work) is reported as such, never as zero.
//!
//! **One program at fault.** Every gene is legal, but the branch compiler sometimes fails on one program (its SetMem
//! limitation, "Ratio has zero denominator", a timeout), and one program's kernels can stop the device's set-up. A
//! batch that does not compile, or whose process stops before its device is ready, is split in halves, recursively,
//! until the program at fault stands alone: that member's every step is [`Outcome::Failed`] with the compiler's or the
//! process's own message, and the rest are scored from their batches. A command that fails while the others run
//! (the entry returns an error, or the process stops inside it) is that step's failure; the process is started again
//! for the commands after it. Nothing is dropped: every step of every member has an outcome.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::futhark_exec::output_with_timeout_bytes;
use crate::futhark_kingdom::FutharkKingdom;
use crate::futhark_sig::{entry_calling, program_cases_named, program_named, read_binary, Array, Signature};
use crate::futhark_wgpu_host::{compile_library_prebuilt, compile_timeout, run_timeout, unix_ns, CompileStages, DriverOf, RunStages, ShimReport};
use crate::typed_gep::Tree;

/// Where a batch runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    /// The branch compiler's webgpu backend, linked against a prebuilt shim (`futhark_wgpu_host::prebuild_shim`).
    Gpu { shim: PathBuf },
    /// Futhark's sequential C backend (the released compiler), on the host: no timing.
    C,
    /// The fork's rust-wgpu backend: the batch is a Rust module, built into a dylib against the runtime prebuilt in
    /// `rt` (`crate::futhark_rust::prebuild_rt`) and run IN THIS PROCESS by the runtime's host, on one device, with
    /// the driver's commands (entries named, not numbered) and the driver's records.
    Rust { rt: PathBuf },
}

/// A member's program.
#[derive(Debug, Clone, Copy)]
pub enum Program<'t> {
    /// A tree of the instance.
    Tree(&'t Tree),
    /// A program's text with the declared signature, its entry point `main` (the published program as a member: the
    /// in-run A/A). It has no quality entry.
    Text(&'t str),
}

/// One step of a member, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// The quality entry on every stored case.
    Quality,
    /// `runs + 1` runs on the timing input (the first a warm-up) of the member's timed entry, or of the published
    /// program's (`yardstick`: the other side of a timed pair).
    Time { yardstick: bool, runs: usize },
    /// The member's timed entry once on the timing input, its output read back.
    Output,
}

/// A member of a batch: a program and what to do with it.
#[derive(Debug, Clone)]
pub struct Member<'t> {
    pub program: Program<'t>,
    pub steps: Vec<Step>,
}

/// One step's result.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The output on every stored case, in case order.
    Quality(Vec<Array>),
    /// Each run's GPU nanoseconds, the warm-up first; `None` for a run in which the GPU timed no pass.
    Time(Vec<Option<f64>>),
    /// The output on the timing input.
    Output(Array),
    /// The step failed: the compiler's, the driver's or the process's own message.
    Failed(String),
}

impl Outcome {
    /// One process-equivalent's time, as `futhark_wgpu_host::ShimReport::fastest_run_ns` takes it from a process run
    /// `runs` times: the fastest of the runs after the warm-up. Infinite when no run timed any pass (no GPU work: no
    /// speed credit, as the per-candidate path gives it); a mix of timed and untimed runs is an error.
    pub fn fastest_ns(&self) -> Result<f64, String> {
        match self {
            Outcome::Time(v) if v.len() >= 2 => {
                let timed = &v[1..];
                if timed.iter().all(Option::is_none) {
                    return Ok(f64::INFINITY);
                }
                if timed.iter().any(Option::is_none) {
                    return Err(format!("runs with and without a timed pass in one timing: {v:?}"));
                }
                Ok(timed.iter().flatten().cloned().fold(f64::INFINITY, f64::min))
            }
            Outcome::Time(v) if v.len() == 1 => Ok(v[0].unwrap_or(f64::INFINITY)),
            Outcome::Failed(e) => Err(e.clone()),
            other => Err(format!("our software has a bug: a timing read from {other:?}")),
        }
    }
}

/// What a batch took, in wall seconds: rendering the source, the compiles that succeeded, the ones that failed (the
/// halving), the processes, and each member's own command time (its quality step, its timing and output steps).
#[derive(Debug, Clone, Default)]
pub struct Cost {
    pub render: f64,
    pub compile: CompileStages,
    pub compiles: usize,
    pub failed_compile_seconds: f64,
    pub failed_compiles: usize,
    pub run: RunStages,
    pub processes: usize,
    /// The seconds each member's quality command took inside the driver, by member.
    pub member_quality_seconds: Vec<f64>,
    /// The seconds each member's timing and output commands took inside the driver, by member.
    pub member_timing_seconds: Vec<f64>,
    /// Of those, the seconds spent creating shaders and pipelines (quality, timing), by member.
    pub member_quality_pipeline_seconds: Vec<f64>,
    pub member_timing_pipeline_seconds: Vec<f64>,
    /// Why each halving happened (a batch that did not compile or whose device set-up failed), with its size.
    pub splits: Vec<String>,
}

/// A batch's results: every member's outcomes, aligned with its steps, and the cost.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub outcomes: Vec<Vec<Outcome>>,
    pub cost: Cost,
}

/// What a host reads: the stored cases' inputs (every case's values in case order, Futhark's binary format), the
/// timing input (the same format), and the published program (its entry point `main`), the other side of a pair.
#[derive(Debug, Clone, Copy)]
pub struct Inputs<'b> {
    pub cases_stdin: &'b [u8],
    pub timing_stdin: Option<&'b [u8]>,
    pub yardstick: Option<&'b str>,
}

/// The batch host of one instance and one run: the signature, the cases, the timing input, where it builds.
pub struct Host<'a> {
    k: &'a FutharkKingdom,
    sig: &'a Signature,
    n_cases: usize,
    backend: Backend,
    dir: PathBuf,
    cases_path: PathBuf,
    timing_path: Option<PathBuf>,
    /// The published program, its entry point `main`.
    yardstick: Option<String>,
    /// Batches built, for their directories' names.
    built: std::cell::Cell<u64>,
    /// The most programs a batch starts with: halved when a batch's bindings pass wgpu's limit, so later batches do
    /// not pay the same failed compile (the cost only: every program is scored the same either way).
    max_batch: std::cell::Cell<usize>,
    /// The rust-wgpu backend's in-process session, made on first use.
    session: std::cell::RefCell<Option<crate::futhark_rust::Session>>,
}

/// The number of a member's entry in a batch.
fn qname(i: usize) -> String {
    format!("q{i}")
}
fn tname(i: usize) -> String {
    format!("t{i}")
}
const YARD: &str = "yard";

/// The published program's text with its entry point `main` renamed: exactly one `entry main ` must be in it.
pub fn rename_main(text: &str, to: &str) -> Result<String, String> {
    let n = text.matches("entry main ").count();
    if n != 1 {
        return Err(format!("the published program has {n} `entry main ` (wants exactly 1)"));
    }
    Ok(text.replacen("entry main ", &format!("entry {to} "), 1))
}

/// A C type of the signature: an `i32` scalar or a rank-`r` array.
fn c_in(rank: usize) -> String {
    if rank == 0 {
        "int32_t".into()
    } else {
        format!("const struct futhark_i32_{rank}d *")
    }
}
fn c_out(rank: usize) -> String {
    if rank == 0 {
        "int32_t *".into()
    } else {
        format!("struct futhark_i32_{rank}d **")
    }
}
fn member_of(rank: usize) -> String {
    if rank == 0 {
        "s".into()
    } else {
        format!("a{rank}")
    }
}

/// The driver's fixed part: the value reader, the records, the main loop. `@...@` are filled in per batch.
const DRIVER: &str = r#"/* THE BATCH DRIVER, written by fuller::futhark_batch for one batch: it calls the batch's entry points, from one
 * process with one Futhark context (one device set-up), on the stored cases and the timing input, and writes every
 * result to one file.
 *
 *   driver <commands> <cases.bin> <timing.bin | -> <results> <seconds>
 *
 * <seconds> bounds the device set-up and each command (alarm: a command past it ends the process, and the host
 * names that command), the per-candidate path's bound of one process.
 *
 * commands, one a line: `q E` (quality entry E on every case), `t E RUNS` (timed entry E, RUNS + 1 runs on the timing
 * input, the first a warm-up, each run's GPU nanoseconds), `o E` (timed entry E once on the timing input, its output).
 * results: records `TAG INDEX STATUS NANOSECONDS PIPELINE_NANOSECONDS BYTES\n` then BYTES of payload; TAG is start,
 * ctx, q, t, o or end; PIPELINE_NANOSECONDS is the shader and pipeline creation inside the record's span (GPU). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
@TIMING_DECLS@
struct futhark_context_config;
struct futhark_context;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
@DECLS@
#define NIN @NIN@
#define NCASES @NCASES@
#define OUT_RANK @OUT_RANK@
static const int IN_RANK[NIN] = {@IN_RANKS@};

typedef struct { int rank; int64_t dims[8]; int64_t n; int32_t *data; } hval;
typedef union { int32_t s;@UNION@ } dval;

static uint64_t unix_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static uint64_t mono_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static FILE *results;
static uint64_t pipe_mark;
static void record(const char *tag, long index, const char *status, uint64_t ns, const void *payload, size_t bytes) {
    uint64_t p = PIPELINE_NS();
    fprintf(results, "%s %ld %s %llu %llu %zu\n", tag, index, status, (unsigned long long)ns, (unsigned long long)(p - pipe_mark), bytes);
    pipe_mark = p;
    if (bytes > 0) fwrite(payload, 1, bytes, results);
    fflush(results);
}
static void fail(const char *tag, long index, const char *msg, int code) {
    record(tag, index, "err", 0, msg, strlen(msg));
    exit(code);
}
static void fail_ctx(const char *tag, long index, struct futhark_context *ctx, const char *what, int code) {
    char *e = futhark_context_get_error(ctx);
    char buf[256];
    snprintf(buf, sizeof buf, "%s: ", what);
    size_t n = strlen(buf) + (e ? strlen(e) : 0) + 1;
    char *msg = malloc(n);
    snprintf(msg, n, "%s%s", buf, e ? e : "(no message)");
    fail(tag, index, msg, code);
}

/* One value of Futhark's binary format (i32 only), as fuller::futhark_sig::write_binary writes it. */
static int read_value(FILE *f, hval *v) {
    int c;
    while ((c = fgetc(f)) == ' ' || c == '\n' || c == '\t' || c == '\r') {}
    unsigned char head[7];
    if (c != 'b') return 1;
    if (fread(head + 1, 1, 6, f) != 6) return 1;
    if (head[1] != 2 || memcmp(head + 3, " i32", 4) != 0) return 1;
    v->rank = head[2];
    if (v->rank > 8) return 1;
    v->n = 1;
    for (int d = 0; d < v->rank; d++) {
        if (fread(&v->dims[d], 8, 1, f) != 1) return 1;
        v->n *= v->dims[d];
    }
    v->data = malloc(sizeof(int32_t) * (size_t)(v->n > 0 ? v->n : 1));
    if (v->n > 0 && fread(v->data, 4, (size_t)v->n, f) != (size_t)v->n) return 1;
    return 0;
}

static void read_values(const char *path, hval *vs, int count) {
    FILE *f = fopen(path, "rb");
    if (!f) fail("start", -1, "cannot open an input file", 2);
    for (int i = 0; i < count; i++) {
        if (read_value(f, &vs[i]) != 0) fail("start", -1, "an input value that is not Futhark's binary i32 format", 2);
        if (vs[i].rank != IN_RANK[i % NIN]) fail("start", -1, "an input of the wrong rank", 2);
    }
    fclose(f);
}

static int upload(struct futhark_context *ctx, const hval *h, dval *d) {
    switch (h->rank) {
    case 0: d->s = h->data[0]; return 0;
@UPLOAD@    default: return 1;
    }
}
static void release(struct futhark_context *ctx, int rank, dval *d) {
    switch (rank) {
    case 0: break;
@FREE@    default: break;
    }
}

/* An output value in Futhark's binary format, appended to a growing buffer. */
typedef struct { unsigned char *p; size_t len, cap; } buf;
static void put(buf *b, const void *x, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 64;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, x, n);
    b->len += n;
}
static int fetch(struct futhark_context *ctx, dval *d, buf *b) {
    unsigned char head[7] = {'b', 2, OUT_RANK, ' ', 'i', '3', '2'};
    put(b, head, 7);
#if OUT_RANK == 0
    put(b, &d->s, 4);
    return 0;
#else
    const int64_t *shape = @SHAPE@(ctx, d->@OUTM@);
    int64_t n = 1;
    for (int i = 0; i < OUT_RANK; i++) { put(b, &shape[i], 8); n *= shape[i]; }
    int32_t *data = malloc(sizeof(int32_t) * (size_t)(n > 0 ? n : 1));
    if (@VALUES@(ctx, d->@OUTM@, data) != 0) return 1;
    if (futhark_context_sync(ctx) != 0) return 1;
    put(b, data, 4 * (size_t)n);
    free(data);
    return 0;
#endif
}

static int call_q(int e, struct futhark_context *ctx, dval *out, dval *in) {
    switch (e) {
@CALL_Q@    default: return -1;
    }
}
static int call_t(int e, struct futhark_context *ctx, dval *out, dval *in) {
    switch (e) {
@CALL_T@    default: return -1;
    }
}
int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s <commands> <cases.bin> <timing.bin|-> <results> <seconds>\n", argv[0]); return 2; }
    unsigned bound = (unsigned)atoi(argv[5]);
    if (bound == 0) { fprintf(stderr, "a bound of 0 seconds\n"); return 2; }
    results = fopen(argv[4], "wb");
    if (!results) { fprintf(stderr, "cannot open %s\n", argv[4]); return 2; }
    record("start", -1, "ok", unix_ns(), NULL, 0);
    static hval cases[NCASES * NIN];
    read_values(argv[2], cases, NCASES * NIN);
    static hval timing[NIN];
    int have_timing = strcmp(argv[3], "-") != 0;
    if (have_timing) read_values(argv[3], timing, NIN);
    alarm(bound);
    struct futhark_context_config *cfg = futhark_context_config_new();
    struct futhark_context *ctx = futhark_context_new(cfg);
    char *e = futhark_context_get_error(ctx);
    if (e) fail("ctx", -1, e, 4);
    alarm(0);
    record("ctx", -1, "ok", unix_ns(), NULL, 0);
    FILE *cmds = fopen(argv[1], "r");
    if (!cmds) fail("ctx", -1, "cannot open the commands", 2);
    char kind[8];
    int entry, runs;
    long index = 0;
    while (fscanf(cmds, "%7s %d", kind, &entry) == 2) {
        alarm(bound);
        pipe_mark = PIPELINE_NS();
        uint64_t t0 = mono_ns();
        if (strcmp(kind, "q") == 0) {
            dval in[NCASES * NIN], out[NCASES];
            for (int i = 0; i < NCASES * NIN; i++)
                if (upload(ctx, &cases[i], &in[i]) != 0) fail_ctx("q", index, ctx, "an input did not upload", 3);
            if (futhark_context_sync(ctx) != 0) fail_ctx("q", index, ctx, "the sync after the inputs", 3);
            int r = call_q(entry, ctx, out, in);
            if (r < 0) fail("q", index, "no such quality entry", 2);
            if (r != 0 || futhark_context_sync(ctx) != 0) fail_ctx("q", index, ctx, "the quality entry failed", 3);
            buf b = {0};
            for (int i = 0; i < NCASES; i++)
                if (fetch(ctx, &out[i], &b) != 0) fail_ctx("q", index, ctx, "an output did not read back", 3);
            for (int i = 0; i < NCASES * NIN; i++) release(ctx, IN_RANK[i % NIN], &in[i]);
            for (int i = 0; i < NCASES; i++) release(ctx, OUT_RANK, &out[i]);
            record("q", index, "ok", mono_ns() - t0, b.p, b.len);
            free(b.p);
        } else if (strcmp(kind, "t") == 0) {
            if (fscanf(cmds, "%d", &runs) != 1 || runs < 0) fail("t", index, "a time command with no runs", 2);
            if (!have_timing) fail("t", index, "a time command with no timing input", 2);
            if (!TIMED_PASSES) fail("t", index, "a time command in a driver with no GPU timing", 2);
            buf b = {0};
            for (int run = 0; run <= runs; run++) {
                dval in[NIN], out[1];
                uint64_t m0 = PASSES();
                for (int i = 0; i < NIN; i++)
                    if (upload(ctx, &timing[i], &in[i]) != 0) fail_ctx("t", index, ctx, "an input did not upload", 3);
                if (futhark_context_sync(ctx) != 0) fail_ctx("t", index, ctx, "the sync after the inputs", 3);
                int r = call_t(entry, ctx, out, in);
                if (r < 0) fail("t", index, "no such timed entry", 2);
                if (r != 0 || futhark_context_sync(ctx) != 0) fail_ctx("t", index, ctx, "the timed entry failed", 3);
                for (int i = 0; i < NIN; i++) release(ctx, IN_RANK[i], &in[i]);
                release(ctx, OUT_RANK, &out[0]);
                if (futhark_context_sync(ctx) != 0) fail_ctx("t", index, ctx, "the sync after a run", 3);
                uint64_t m1 = PASSES();
                char word[64];
                if (m1 == m0) {
                    snprintf(word, sizeof word, "%snone", run ? " " : "");
                } else {
                    double ns = 0;
                    for (uint64_t p = m0; p < m1; p++) {
                        double x = PASS_NS(p);
                        if (x < 0) fail("t", index, "a pass the GPU did not time", 3);
                        ns += x;
                    }
                    snprintf(word, sizeof word, "%s%.1f", run ? " " : "", ns);
                }
                put(&b, word, strlen(word));
            }
            record("t", index, "ok", mono_ns() - t0, b.p, b.len);
            free(b.p);
        } else if (strcmp(kind, "o") == 0) {
            if (!have_timing) fail("o", index, "an output command with no timing input", 2);
            dval in[NIN], out[1];
            for (int i = 0; i < NIN; i++)
                if (upload(ctx, &timing[i], &in[i]) != 0) fail_ctx("o", index, ctx, "an input did not upload", 3);
            int r = call_t(entry, ctx, out, in);
            if (r < 0) fail("o", index, "no such timed entry", 2);
            if (r != 0 || futhark_context_sync(ctx) != 0) fail_ctx("o", index, ctx, "the timed entry failed", 3);
            buf b = {0};
            if (fetch(ctx, &out[0], &b) != 0) fail_ctx("o", index, ctx, "the output did not read back", 3);
            for (int i = 0; i < NIN; i++) release(ctx, IN_RANK[i], &in[i]);
            release(ctx, OUT_RANK, &out[0]);
            record("o", index, "ok", mono_ns() - t0, b.p, b.len);
            free(b.p);
        } else {
            fail("q", index, "an unknown command", 2);
        }
        alarm(0);
        index++;
    }
    record("end", index, "ok", unix_ns(), NULL, 0);
    fclose(results);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
"#;

/// A command of one process: what it runs, and which member's step it is.
#[derive(Debug, Clone)]
struct Cmd {
    line: String,
    member: usize,
    step: usize,
}

/// One record of the results file.
#[derive(Debug)]
struct Record {
    tag: String,
    index: i64,
    ok: bool,
    ns: u64,
    /// The shader and pipeline creation inside the record's span.
    pipeline_ns: u64,
    payload: Vec<u8>,
}

/// One process's records, stages, and how it ended (`Ok` when it wrote its `end` record).
struct Process {
    records: Vec<Record>,
    stages: RunStages,
    ended: Result<(), String>,
}

fn parse_records(raw: &[u8]) -> Result<Vec<Record>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let nl = raw[i..].iter().position(|b| *b == b'\n').ok_or("a results record with no end of line")? + i;
        let head = std::str::from_utf8(&raw[i..nl]).map_err(|_| "a results record that is not text")?;
        let f: Vec<&str> = head.split(' ').collect();
        let [tag, index, status, ns, pipeline_ns, bytes] = f.as_slice() else { return Err(format!("a results record {head:?}")) };
        let bytes: usize = bytes.parse().map_err(|_| format!("a results record {head:?}"))?;
        let payload = raw.get(nl + 1..nl + 1 + bytes).ok_or("a results record cut short")?.to_vec();
        out.push(Record {
            tag: tag.to_string(),
            index: index.parse().map_err(|_| format!("a results record {head:?}"))?,
            ok: *status == "ok",
            ns: ns.parse().map_err(|_| format!("a results record {head:?}"))?,
            pipeline_ns: pipeline_ns.parse().map_err(|_| format!("a results record {head:?}"))?,
            payload,
        });
        i = nl + 1 + bytes;
    }
    Ok(out)
}

impl<'a> Host<'a> {
    /// A host building under `dir`; the cases and the timing input are written there once, here.
    pub fn new(k: &'a FutharkKingdom, sig: &'a Signature, n_cases: usize, backend: Backend, dir: &Path, inputs: Inputs) -> Result<Host<'a>, String> {
        let Inputs { cases_stdin, timing_stdin, yardstick } = inputs;
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        let cases_path = dir.join("cases.bin");
        std::fs::write(&cases_path, cases_stdin).map_err(|e| format!("write {}: {e}", cases_path.display()))?;
        let timing_path = match timing_stdin {
            Some(t) => {
                let p = dir.join("timing.bin");
                std::fs::write(&p, t).map_err(|e| format!("write {}: {e}", p.display()))?;
                Some(p)
            }
            None => None,
        };
        if sig.inputs.iter().chain(std::iter::once(&sig.output)).any(|p| p.elem != "i32" || p.dims.len() > 3) {
            return Err("the batch driver reads and writes i32 values of rank at most 3".into());
        }
        Ok(Host { k, sig, n_cases, backend, dir: dir.to_path_buf(), cases_path, timing_path, yardstick: yardstick.map(str::to_string), built: std::cell::Cell::new(0), max_batch: std::cell::Cell::new(usize::MAX), session: std::cell::RefCell::new(None) })
    }

    fn in_ranks(&self) -> Vec<usize> {
        self.sig.inputs.iter().map(|p| p.dims.len()).collect()
    }

    /// The Futhark source of the members `idx` (their entries, then the published program's when one of them is timed
    /// against it).
    fn source(&self, members: &[Member], idx: &[usize]) -> Result<String, String> {
        let mut s = format!("-- a batch of {} programs (fuller::futhark_batch)\n", idx.len());
        let mut yard = false;
        for &i in idx {
            let m = &members[i];
            let quality = m.steps.contains(&Step::Quality);
            let timed = m.steps.iter().any(|s| matches!(s, Step::Time { yardstick: false, .. } | Step::Output));
            yard |= m.steps.iter().any(|s| matches!(s, Step::Time { yardstick: true, .. }));
            match m.program {
                Program::Tree(t) => {
                    if quality {
                        s.push_str(&program_cases_named(self.k, self.sig, t, self.n_cases, &format!("kq{i}"), &qname(i))?);
                        s.push('\n');
                    }
                    if timed {
                        s.push_str(&program_named(self.k, self.sig, t, &tname(i))?);
                        s.push('\n');
                    }
                }
                Program::Text(text) => {
                    if quality {
                        return Err("our software has a bug: a program given as text has no quality entry in a batch".into());
                    }
                    if timed {
                        // In a module of its own: a second copy of the published program (the in-run A/A beside
                        // `yard`) would define its helpers twice, which Futhark refuses.
                        let def = rename_main(text, "main")?.replacen("entry main ", "def main ", 1);
                        s.push_str(&format!("module m{i} = {{\n{def}\n}}\n{}\n", entry_calling(self.sig, &tname(i), &format!("m{i}.main"))));
                    }
                }
            }
        }
        if yard {
            s.push_str(&rename_main(self.yardstick.as_deref().ok_or("a pair timed with no published program")?, YARD)?);
            s.push('\n');
        }
        Ok(s)
    }

    /// The driver of the members `idx`: the entries' tables (a quality entry's index is its place among the quality
    /// entries, a timed one's among the timed ones, the published program's last).
    /// The commands of the members `idx`, in order (their indices into the driver's tables, as [`Host::driver`]
    /// numbers them).
    fn commands(&self, members: &[Member], idx: &[usize]) -> Result<Vec<Cmd>, String> {
        let gpu = matches!(self.backend, Backend::Gpu { .. } | Backend::Rust { .. });
        let named = matches!(self.backend, Backend::Rust { .. });
        let (mut nq, mut nt) = (0usize, 0usize);
        let mut q_of = std::collections::BTreeMap::new();
        let mut t_of = std::collections::BTreeMap::new();
        for &i in idx {
            let m = &members[i];
            if m.steps.contains(&Step::Quality) {
                q_of.insert(i, nq);
                nq += 1;
            }
            if m.steps.iter().any(|s| matches!(s, Step::Time { yardstick: false, .. } | Step::Output)) {
                t_of.insert(i, nt);
                nt += 1;
            }
        }
        let mut cmds = Vec::new();
        for &i in idx {
            for (j, s) in members[i].steps.iter().enumerate() {
                let line = match (s, named) {
                    (Step::Quality, false) => format!("q {}", q_of[&i]),
                    (Step::Time { yardstick: false, runs }, false) => format!("t {} {runs}", t_of[&i]),
                    (Step::Time { yardstick: true, runs }, false) => format!("t {nt} {runs}"),
                    (Step::Output, false) => format!("o {}", t_of[&i]),
                    (Step::Quality, true) => format!("q {}", qname(i)),
                    (Step::Time { yardstick: false, runs }, true) => format!("t {} {runs}", tname(i)),
                    (Step::Time { yardstick: true, runs }, true) => format!("t {YARD} {runs}"),
                    (Step::Output, true) => format!("o {}", tname(i)),
                };
                if !gpu && matches!(s, Step::Time { .. }) {
                    return Err("our software has a bug: a timed step on the c backend, which has no GPU timing".into());
                }
                cmds.push(Cmd { line, member: i, step: j });
            }
        }
        Ok(cmds)
    }

    /// The driver of the members `idx`, from the compiler's manifest: the entries' tables (a quality entry's index is
    /// its place among the quality entries, a timed one's among the timed ones, the published program's last). A
    /// quality entry returns its cases' outputs as one tuple, which Futhark's C interface gives as an opaque value
    /// with a projection a field; the manifest names them.
    fn driver(&self, members: &[Member], idx: &[usize], manifest: &serde_json::Value) -> Result<String, String> {
        let ranks = self.in_ranks();
        let out_rank = self.sig.output.dims.len();
        let used: std::collections::BTreeSet<usize> = ranks.iter().chain(std::iter::once(&out_rank)).cloned().filter(|r| *r > 0).collect();
        let mut decls = String::new();
        let mut union = String::new();
        let mut upload = String::new();
        let mut free = String::new();
        for r in &used {
            let dims: String = (0..*r).map(|d| format!(", int64_t dim{d}")).collect();
            let args: String = (0..*r).map(|d| format!(", h->dims[{d}]")).collect();
            decls.push_str(&format!(
                "struct futhark_i32_{r}d;\nstruct futhark_i32_{r}d *futhark_new_i32_{r}d(struct futhark_context *ctx, const int32_t *data{dims});\nint futhark_free_i32_{r}d(struct futhark_context *ctx, struct futhark_i32_{r}d *arr);\nint futhark_values_i32_{r}d(struct futhark_context *ctx, struct futhark_i32_{r}d *arr, int32_t *data);\nconst int64_t *futhark_shape_i32_{r}d(struct futhark_context *ctx, struct futhark_i32_{r}d *arr);\n"
            ));
            union.push_str(&format!(" struct futhark_i32_{r}d *a{r};"));
            upload.push_str(&format!("    case {r}: d->a{r} = futhark_new_i32_{r}d(ctx, h->data{args}); return d->a{r} == NULL;\n"));
            free.push_str(&format!("    case {r}: futhark_free_i32_{r}d(ctx, d->a{r}); break;\n"));
        }
        let nin = ranks.len();
        let (mut call_q, mut call_t) = (String::new(), String::new());
        let mut nt = 0usize;
        let q_ins = (0..self.n_cases * nin).map(|i| format!("in[{i}].{}", member_of(ranks[i % nin]))).collect::<Vec<_>>().join(", ");
        let q_proto_ins = (0..self.n_cases * nin).map(|i| c_in(ranks[i % nin])).collect::<Vec<_>>().join(", ");
        let t_proto = format!("(struct futhark_context *ctx, {}, {})", c_out(out_rank), ranks.iter().map(|r| c_in(*r)).collect::<Vec<_>>().join(", "));
        let t_args = format!("ctx, &out[0].{}, {}", member_of(out_rank), (0..nin).map(|i| format!("in[{i}].{}", member_of(ranks[i]))).collect::<Vec<_>>().join(", "));
        // The quality entries' output: one value (one case) or a tuple, an opaque value with a projection a field.
        let mut tuple: Option<(String, String, Vec<String>)> = None;
        let mut nq = 0usize;
        let mut yard = false;
        for &i in idx {
            let m = &members[i];
            if m.steps.contains(&Step::Quality) {
                let e = &manifest["entry_points"][qname(i)];
                let out_ty = e["output"]["type"].as_str().ok_or_else(|| format!("the manifest has no output type for {}", qname(i)))?;
                let ty = &manifest["types"][out_ty];
                if self.n_cases > 1 {
                    if ty["kind"].as_str() != Some("opaque") {
                        return Err(format!("the quality entry {} returns {out_ty}, not the opaque tuple of its {} cases", qname(i), self.n_cases));
                    }
                    let ctype = ty["ctype"].as_str().ok_or("an opaque type with no ctype")?.trim_end_matches('*').trim().to_string();
                    let free = ty["ops"]["free"].as_str().ok_or("an opaque type with no free")?.to_string();
                    let fields: Vec<String> = ty["record"]["fields"].as_array().ok_or("an opaque type with no fields")?.iter().map(|f| f["project"].as_str().map(str::to_string).ok_or("a field with no projection")).collect::<Result<_, _>>()?;
                    if fields.len() != self.n_cases {
                        return Err(format!("the quality entry {} returns {} fields for {} cases", qname(i), fields.len(), self.n_cases));
                    }
                    match &tuple {
                        None => tuple = Some((ctype, free, fields)),
                        Some(t) if t.0 == ctype => {}
                        Some(t) => return Err(format!("the quality entries return two tuple types, {} and {ctype}", t.0)),
                    }
                    let (ctype, free, fields) = tuple.as_ref().ok_or("our software has a bug: no tuple type")?;
                    decls.push_str(&format!("int futhark_entry_{}(struct futhark_context *ctx, {ctype} **out, {q_proto_ins});\n", qname(i)));
                    let projections: String = fields.iter().enumerate().map(|(f, p)| format!(" if (r == 0) r = {p}(ctx, &out[{f}].{}, tup);", member_of(out_rank))).collect();
                    call_q.push_str(&format!("    case {nq}: {{ {ctype} *tup; int r = futhark_entry_{}(ctx, &tup, {q_ins});{projections} if (r == 0) r = {free}(ctx, tup); return r; }}\n", qname(i)));
                } else {
                    decls.push_str(&format!("int futhark_entry_{}{t_proto};\n", qname(i)));
                    call_q.push_str(&format!("    case {nq}: return futhark_entry_{}({t_args});\n", qname(i)));
                }
                nq += 1;
            }
            if m.steps.iter().any(|s| matches!(s, Step::Time { yardstick: false, .. } | Step::Output)) {
                decls.push_str(&format!("int futhark_entry_{}{t_proto};\n", tname(i)));
                call_t.push_str(&format!("    case {nt}: return futhark_entry_{}({t_args});\n", tname(i)));
                nt += 1;
            }
            yard |= m.steps.iter().any(|s| matches!(s, Step::Time { yardstick: true, .. }));
        }
        if let Some((ctype, free, fields)) = &tuple {
            let field_out = c_out(out_rank);
            let mut d = format!("{ctype};\nint {free}(struct futhark_context *ctx, {ctype} *obj);\n");
            for p in fields {
                d.push_str(&format!("int {p}(struct futhark_context *ctx, {field_out} out, const {ctype} *obj);\n"));
            }
            decls = format!("{d}{decls}");
        }
        if yard {
            decls.push_str(&format!("int futhark_entry_{YARD}{t_proto};\n"));
            call_t.push_str(&format!("    case {nt}: return futhark_entry_{YARD}({t_args});\n"));
        }
        let gpu = matches!(self.backend, Backend::Gpu { .. });
        let timing_decls = if gpu {
            "uint64_t futhark_wgpu_shim_timed_passes(void);\ndouble futhark_wgpu_shim_pass_ns(uint64_t index);\nuint64_t futhark_wgpu_shim_pipeline_ns(void);\n#define TIMED_PASSES 1\n#define PASSES() futhark_wgpu_shim_timed_passes()\n#define PASS_NS(p) futhark_wgpu_shim_pass_ns(p)\n#define PIPELINE_NS() futhark_wgpu_shim_pipeline_ns()\n"
        } else {
            "#define TIMED_PASSES 0\n#define PASSES() 0\n#define PASS_NS(p) ((void)(p), -1.0)\n#define PIPELINE_NS() 0ull\n"
        };
        let shape = if out_rank > 0 { format!("futhark_shape_i32_{out_rank}d") } else { "(void)".into() };
        let values = if out_rank > 0 { format!("futhark_values_i32_{out_rank}d") } else { "(void)".into() };
        let c = DRIVER
            .replace("@DECLS@", &decls)
            .replace("@NIN@", &nin.to_string())
            .replace("@NCASES@", &self.n_cases.to_string())
            .replace("@OUT_RANK@", &out_rank.to_string())
            .replace("@IN_RANKS@", &ranks.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))
            .replace("@UNION@", &union)
            .replace("@UPLOAD@", &upload)
            .replace("@FREE@", &free)
            .replace("@SHAPE@", &shape)
            .replace("@VALUES@", &values)
            .replace("@OUTM@", &member_of(out_rank))
            .replace("@CALL_Q@", &call_q)
            .replace("@CALL_T@", &call_t)
            .replace("@TIMING_DECLS@", timing_decls);
        Ok(c)
    }

    /// Compile the members `idx` into one executable.
    fn compile(&self, members: &[Member], idx: &[usize], cost: &mut Cost) -> Result<(PathBuf, Vec<Cmd>, CompileStages), String> {
        let n = self.built.get();
        self.built.set(n + 1);
        let dir = self.dir.join(format!("b{n}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| format!("remove the stale {}: {e}", dir.display()))?;
        }
        let t = std::time::Instant::now();
        let source = self.source(members, idx)?;
        cost.render += t.elapsed().as_secs_f64();
        let cmds = self.commands(members, idx)?;
        let driver = |manifest: &serde_json::Value| self.driver(members, idx, manifest);
        // Each program its own compile bound, as on the per-candidate path.
        let timeout = compile_timeout() * u32::try_from(idx.len().max(1)).map_err(|_| "a batch too large")?;
        let (exe, stages) = match &self.backend {
            Backend::Gpu { shim } => compile_library_prebuilt(&source, &driver, &dir, shim, timeout)?,
            Backend::C => compile_c_library(&source, &driver, &dir, timeout)?,
            // The crate's name is unique in the process: the host keeps every module it loads.
            Backend::Rust { rt } => crate::futhark_rust::compile_module(&source, &dir, rt, &format!("p{}_b{n}", std::process::id()), timeout)?,
        };
        Ok((exe, cmds, stages))
    }

    /// One process of `exe` on `cmds` (all of them, from the first): the records it wrote, its stages, and how it
    /// ended (`Ok` when it wrote its `end` record; otherwise its exit and its error output).
    fn process(&self, exe: &Path, cmds: &[Cmd], timestamps: bool) -> Result<Process, String> {
        if let Backend::Rust { rt } = &self.backend {
            return self.in_process(rt, exe, cmds);
        }
        let dir = exe.parent().ok_or("an executable with no directory")?;
        let cmd_path = dir.join("commands.txt");
        let res_path = dir.join("results.bin");
        let report_path = dir.join("shim_report.json");
        for p in [&res_path, &report_path] {
            if p.exists() {
                std::fs::remove_file(p).map_err(|e| format!("remove the stale {}: {e}", p.display()))?;
            }
        }
        std::fs::write(&cmd_path, cmds.iter().map(|c| format!("{}\n", c.line)).collect::<String>()).map_err(|e| format!("write {}: {e}", cmd_path.display()))?;
        // The timing input is read only by a process that runs on it (the histogram's is 128 MB).
        let on_timing = cmds.iter().any(|c| c.line.starts_with("t ") || c.line.starts_with("o "));
        let timing = if on_timing { self.timing_path.as_deref().ok_or("a timed or output step with no timing input")? } else { Path::new("-") };
        // The driver bounds its device set-up and each command by the per-candidate path's bound of one process; this
        // bound on the whole process is a backstop.
        let bound = run_timeout();
        let mut cmd = Command::new(exe);
        cmd.arg(&cmd_path).arg(&self.cases_path).arg(timing).arg(&res_path).arg(bound.as_secs().max(1).to_string());
        if matches!(self.backend, Backend::Gpu { .. }) {
            cmd.env("FUTHARK_WGPU_SHIM_REPORT", &report_path);
            if timestamps {
                cmd.env("FUTHARK_WGPU_SHIM_TIMESTAMPS", "1");
            }
        }
        let limit = bound * u32::try_from(cmds.len() + 2).map_err(|_| "a batch too large")?;
        let spawn = unix_ns();
        let t = std::time::Instant::now();
        let out = output_with_timeout_bytes(&mut cmd, limit, None);
        let total = t.elapsed().as_secs_f64();
        let raw = std::fs::read(&res_path).unwrap_or_default();
        let records = parse_records(&raw)?;
        let clocks = std::fs::read_to_string(&report_path).ok().and_then(|r| ShimReport::parse(&r).ok()).and_then(|r| r.clocks);
        let stages = RunStages::of(spawn, total, clocks.as_ref());
        let ended = match out {
            Ok(o) if o.status.success() && records.last().is_some_and(|r| r.tag == "end") => Ok(()),
            Ok(o) => {
                // SIGALRM (14) is the driver's own bound on one command.
                let alarm = if std::os::unix::process::ExitStatusExt::signal(&o.status) == Some(14) { format!(" (the command ran past its bound of {} s)", bound.as_secs()) } else { String::new() };
                Err(format!("the batch process stopped ({}{alarm}):\n{}", o.status, String::from_utf8_lossy(&o.stderr)))
            }
            Err(e) => Err(e),
        };
        Ok(Process { records, stages, ended })
    }

    /// THE RUST-WGPU BACKEND'S PROCESS, in this process: the runtime's host runs the commands on the module and writes
    /// the driver's records. It ends as a process that wrote its `end` record (the host runs every command; one that
    /// failed has its failure record and the next runs on a new context). The stages: `pipelines` is the shader
    /// module and pipeline creation the records report, `runs` the rest.
    fn in_process(&self, rt: &Path, dylib: &Path, cmds: &[Cmd]) -> Result<Process, String> {
        let mut g = self.session.borrow_mut();
        if g.is_none() {
            *g = Some(crate::futhark_rust::Session::new(rt, run_timeout())?);
        }
        let session = g.as_ref().ok_or("our software has a bug: no session")?;
        let on_timing = cmds.iter().any(|c| c.line.starts_with("t ") || c.line.starts_with("o "));
        let cases = std::fs::read(&self.cases_path).map_err(|e| format!("{}: {e}", self.cases_path.display()))?;
        let timing = match (&self.timing_path, on_timing) {
            (Some(p), true) => Some(std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?),
            (None, true) => return Err("a timed or output step with no timing input".into()),
            (_, false) => None,
        };
        let commands: String = cmds.iter().map(|c| format!("{}\n", c.line)).collect();
        let t = std::time::Instant::now();
        let raw = session.run(dylib, &commands, &cases, self.n_cases, self.sig.inputs.len(), timing.as_deref())?;
        let total = t.elapsed().as_secs_f64();
        let records = parse_records(&raw)?;
        let pipelines: f64 = records.iter().map(|r| r.pipeline_ns as f64 / 1e9).sum();
        let stages = RunStages { pipelines, runs: (total - pipelines).max(0.0), total, ..RunStages::default() };
        let ended = if records.last().is_some_and(|r| r.tag == "end") {
            Ok(())
        } else {
            Err(format!("the in-process host stopped: {}", records.iter().rev().find(|r| !r.ok).map(|r| String::from_utf8_lossy(&r.payload).to_string()).unwrap_or_default()))
        };
        Ok(Process { records, stages, ended })
    }

    /// Run every step of `members`, each a batch of the members not seen before; see the module's header for the
    /// halving. Every step gets an outcome.
    pub fn run(&self, members: &[Member]) -> Result<BatchResult, String> {
        let mut outcomes: Vec<Vec<Option<Outcome>>> = members.iter().map(|m| vec![None; m.steps.len()]).collect();
        let n = members.len();
        let mut cost = Cost { member_quality_seconds: vec![0.0; n], member_timing_seconds: vec![0.0; n], member_quality_pipeline_seconds: vec![0.0; n], member_timing_pipeline_seconds: vec![0.0; n], ..Cost::default() };
        let all: Vec<usize> = (0..members.len()).filter(|i| !members[*i].steps.is_empty()).collect();
        for chunk in all.chunks(self.max_batch.get().max(1)) {
            self.group(members, chunk, &mut outcomes, &mut cost)?;
        }
        let outcomes = outcomes
            .into_iter()
            .map(|v| v.into_iter().map(|o| o.ok_or_else(|| "our software has a bug: a batch step with no outcome".to_string())).collect::<Result<Vec<_>, _>>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BatchResult { outcomes, cost })
    }

    fn fail_member(&self, members: &[Member], i: usize, msg: &str, outcomes: &mut [Vec<Option<Outcome>>]) {
        for o in outcomes[i].iter_mut().take(members[i].steps.len()) {
            *o = Some(Outcome::Failed(msg.to_string()));
        }
    }

    fn group(&self, members: &[Member], idx: &[usize], outcomes: &mut [Vec<Option<Outcome>>], cost: &mut Cost) -> Result<(), String> {
        let t = std::time::Instant::now();
        let (exe, cmds, stages) = match self.compile(members, idx, cost) {
            Ok(c) => c,
            Err(e) => {
                cost.failed_compile_seconds += t.elapsed().as_secs_f64();
                cost.failed_compiles += 1;
                return self.split(members, idx, &e, outcomes, cost);
            }
        };
        cost.compile.add(&stages);
        cost.compiles += 1;
        if matches!(self.backend, Backend::Gpu { .. } | Backend::Rust { .. }) {
            // Futhark's webgpu code numbers the bindings across the whole program, and wgpu refuses a binding index
            // above its limit at the device's set-up (measured 2026-10-28: a batch of 24 histogram programs reached
            // 1011). Halving before the process runs saves the failed set-up.
            let dir = exe.parent().ok_or("an executable with no directory")?;
            let top = if matches!(self.backend, Backend::Rust { .. }) {
                crate::futhark_rust::max_binding_index(&std::fs::read_to_string(dir.join("prog.rs")).map_err(|e| format!("the generated module: {e}"))?)
            } else {
                max_binding_index(&std::fs::read_to_string(dir.join("prog.c")).map_err(|e| format!("the generated C: {e}"))?)
            };
            if top > WGPU_MAX_BINDING_INDEX {
                self.max_batch.set(self.max_batch.get().min((idx.len() / 2).max(1)));
                return self.split(members, idx, &format!("the batch's kernels use binding index {top}, over wgpu's limit of {WGPU_MAX_BINDING_INDEX} (Futhark's webgpu code numbers bindings across the whole program)"), outcomes, cost);
            }
        }
        let timestamps = cmds.iter().any(|c| c.line.starts_with("t "));
        let mut from = 0;
        let mut first = true;
        while from < cmds.len() {
            let Process { records, stages, ended } = self.process(&exe, &cmds[from..], timestamps)?;
            cost.run.add(&stages);
            cost.processes += 1;
            if !records.iter().any(|r| r.tag == "ctx" && r.ok) {
                // The process stopped before its device was ready: one program's kernels can do that.
                let msg = records.iter().find(|r| !r.ok).map(|r| String::from_utf8_lossy(&r.payload).to_string()).unwrap_or_default();
                let why = format!("the batch's device set-up failed: {msg}\n{}", ended.err().unwrap_or_default());
                if !first {
                    return Err(format!("our software has a bug: a batch's device set-up failed after it had run commands: {why}"));
                }
                let rest: Vec<usize> = idx.to_vec();
                return self.split(members, &rest, &why, outcomes, cost);
            }
            first = false;
            let mut done = 0;
            let mut recorded_failure = false;
            for r in records.iter().filter(|r| ["q", "t", "o"].contains(&r.tag.as_str())) {
                let c = cmds.get(from + r.index as usize).ok_or("a results record of no command")?;
                if r.tag == "q" {
                    cost.member_quality_seconds[c.member] += r.ns as f64 / 1e9;
                    cost.member_quality_pipeline_seconds[c.member] += r.pipeline_ns as f64 / 1e9;
                } else {
                    cost.member_timing_seconds[c.member] += r.ns as f64 / 1e9;
                    cost.member_timing_pipeline_seconds[c.member] += r.pipeline_ns as f64 / 1e9;
                }
                let o = if !r.ok {
                    Outcome::Failed(String::from_utf8_lossy(&r.payload).to_string())
                } else {
                    match r.tag.as_str() {
                        "q" => {
                            let v = read_binary(&r.payload)?;
                            if v.len() != self.n_cases {
                                return Err(format!("our software has a bug: a quality record of {} values for {} cases", v.len(), self.n_cases));
                            }
                            Outcome::Quality(v)
                        }
                        "t" => Outcome::Time(
                            String::from_utf8_lossy(&r.payload)
                                .split(' ')
                                .map(|w| if w == "none" { Ok(None) } else { w.parse::<f64>().map(Some).map_err(|_| format!("a timing word {w:?}")) })
                                .collect::<Result<_, _>>()?,
                        ),
                        _ => Outcome::Output(read_binary(&r.payload)?.into_iter().next().ok_or("an output record with no value")?),
                    }
                };
                recorded_failure = matches!(o, Outcome::Failed(_));
                outcomes[c.member][c.step] = Some(o);
                done = r.index as usize + 1;
            }
            match ended {
                Ok(()) => {
                    if from + done != cmds.len() {
                        return Err(format!("our software has a bug: the batch process ended after {} of {} commands", from + done, cmds.len()));
                    }
                    break;
                }
                Err(why) => {
                    // A command that failed wrote its record, and the next process starts after it; one the process
                    // stopped inside wrote none: the failure is that command's, with the process's own message.
                    let at = from + done;
                    if recorded_failure {
                        from = at;
                    } else if at < cmds.len() {
                        outcomes[cmds[at].member][cmds[at].step] = Some(Outcome::Failed(why));
                        from = at + 1;
                    } else {
                        return Err(format!("our software has a bug: the batch process stopped after its last command: {why}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// The members `idx` did not build or start: halve them, or name the one at fault.
    fn split(&self, members: &[Member], idx: &[usize], why: &str, outcomes: &mut [Vec<Option<Outcome>>], cost: &mut Cost) -> Result<(), String> {
        if idx.len() == 1 {
            self.fail_member(members, idx[0], why, outcomes);
            return Ok(());
        }
        cost.splits.push(format!("a batch of {} programs was halved: {why}", idx.len()));
        let (a, b) = idx.split_at(idx.len() / 2);
        self.group(members, a, outcomes, cost)?;
        self.group(members, b, outcomes, cost)
    }
}

/// wgpu's largest binding index in a bind group layout (`Limits::max_bindings_per_bind_group`, 1000 by default).
const WGPU_MAX_BINDING_INDEX: u64 = 1000;

/// The largest binding index the generated C's kernels declare (`.scalars_binding =N` and `binding_indices[] = {..}`).
fn max_binding_index(c: &str) -> u64 {
    let mut top = 0;
    let digits = |t: &str| t.split(|ch: char| !ch.is_ascii_digit()).filter(|w| !w.is_empty()).filter_map(|w| w.parse::<u64>().ok()).collect::<Vec<_>>();
    for (i, _) in c.match_indices(".scalars_binding =") {
        let rest = &c[i + ".scalars_binding =".len()..];
        let n: String = rest.chars().take_while(char::is_ascii_digit).collect();
        top = top.max(n.parse().unwrap_or(0));
    }
    for (i, _) in c.match_indices("binding_indices[] = {") {
        let rest = &c[i + "binding_indices[] = {".len()..];
        let body = &rest[..rest.find('}').unwrap_or(0)];
        top = top.max(digits(body).into_iter().max().unwrap_or(0));
    }
    top
}

/// The c backend's library build: `futhark c --library` (the released compiler), then `cc` of its C and the driver.
fn compile_c_library(source: &str, driver: DriverOf, dir: &Path, timeout: Duration) -> Result<(PathBuf, CompileStages), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, source).map_err(|e| format!("write {}: {e}", fut.display()))?;
    let mut stages = CompileStages::default();
    let t = std::time::Instant::now();
    let out = crate::futhark_exec::output_with_timeout(Command::new("futhark").arg("c").arg("--library").arg(&fut).arg("-o").arg(dir.join("prog")), timeout, None)?;
    stages.futhark = t.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(format!("futhark c --library failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    let m = dir.join("prog.json");
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&m).map_err(|e| format!("{}: {e}", m.display()))?).map_err(|e| format!("{}: {e}", m.display()))?;
    let d = dir.join("driver.c");
    std::fs::write(&d, driver(&manifest)?).map_err(|e| format!("write {}: {e}", d.display()))?;
    let exe = dir.join("prog_c");
    let t = std::time::Instant::now();
    let cc = crate::futhark_exec::output_with_timeout(Command::new("cc").arg("-O2").arg(dir.join("prog.c")).arg(&d).arg("-o").arg(&exe).arg("-lm"), timeout, None)?;
    stages.cc = t.elapsed().as_secs_f64();
    if !cc.status.success() {
        return Err(format!("cc of the c backend's library failed:\n{}", String::from_utf8_lossy(&cc.stderr)));
    }
    Ok((exe, stages))
}

#[cfg(test)]
mod tests;
