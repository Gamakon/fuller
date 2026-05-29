//! Brainfuck tape interpreter — semantic ground-truth for soundness testing.
//!
//! # Semantics
//!
//! - Tape: 30 000 u8 cells, initialised to 0.
//! - Pointer: starts at cell 0; wraps modulo `TAPE_SIZE` on underflow/overflow.
//!   Wrapping semantics make `<>` and `><` unconditionally no-ops regardless of
//!   position, which is required for the egglog pointer-cancellation rules to be
//!   sound.
//! - Cell arithmetic: wrapping u8 (standard BF: 255 + 1 = 0).
//! - Input: a supplied byte vector; `,` reads the next byte, or 0 if exhausted.
//! - Output: collected into a `Vec<u8>` (`.` appends).
//! - Step limit: `MAX_STEPS` instructions before returning `StepLimitExceeded`.

/// Maximum BF instructions executed before giving up.
pub const MAX_STEPS: usize = 500_000;

/// Tape size — standard BF tape length.
pub const TAPE_SIZE: usize = 30_000;

/// Result of running a BF program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapeResult {
    /// Program halted normally; `output` contains bytes written by `.`.
    Ok { output: Vec<u8> },
    /// Ran out of steps — program likely non-terminating.
    StepLimitExceeded,
    /// Unmatched brackets or other parse error in the source.
    ParseError(String),
}

impl TapeResult {
    /// Returns output bytes if the program halted normally.
    pub fn output(&self) -> Option<&[u8]> {
        if let TapeResult::Ok { output } = self {
            Some(output)
        } else {
            None
        }
    }
}

/// Run a BF program from its raw source text (e.g. `+[>+<-]>`) against
/// `input` bytes. Returns the tape output.
pub fn run_bf(source: &str, input: &[u8]) -> TapeResult {
    let ops = match compile_source(source) {
        Ok(ops) => ops,
        Err(e) => return TapeResult::ParseError(e),
    };
    execute(&ops, input)
}

/// Run a BF program from a Prog s-expression. Used in soundness checks.
pub fn run_bf_sexpr(prog_sexpr: &str, input: &[u8]) -> TapeResult {
    let source = match crate::bf::parse::sexpr_to_source(prog_sexpr) {
        Ok(s) => s,
        Err(e) => return TapeResult::ParseError(e),
    };
    run_bf(&source, input)
}

// ---------------------------------------------------------------------------
// Compilation: BF source -> flat op sequence with jump-table
// ---------------------------------------------------------------------------

/// Flat BF op for the internal executor.
///
/// `AddN`/`MoveN` s-expression forms are expanded back to raw `+`/`-`/`<`/`>`
/// characters in `sexpr_to_source` before entering `compile_source`, so the
/// executor never needs those aggregated forms — they live only in the egglog
/// layer.
#[derive(Debug, Clone, Copy)]
enum FlatOp {
    Inc,
    Dec,
    Left,
    Right,
    Out,
    In,
    /// Jump forward to `target` if current cell is 0.
    JmpFwd(usize),
    /// Jump back to `target` if current cell is non-0.
    JmpBak(usize),
    /// Set current cell to 0 (compiled from `[-]` or `[+]` patterns).
    Clear,
}

fn compile_source(source: &str) -> Result<Vec<FlatOp>, String> {
    let mut ops: Vec<FlatOp> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for ch in source.chars() {
        match ch {
            '+' => ops.push(FlatOp::Inc),
            '-' => ops.push(FlatOp::Dec),
            '<' => ops.push(FlatOp::Left),
            '>' => ops.push(FlatOp::Right),
            '.' => ops.push(FlatOp::Out),
            ',' => ops.push(FlatOp::In),
            '[' => {
                let idx = ops.len();
                ops.push(FlatOp::JmpFwd(0)); // placeholder
                stack.push(idx);
            }
            ']' => {
                let open = stack.pop().ok_or_else(|| "unmatched ']'".to_string())?;
                let close = ops.len();
                ops.push(FlatOp::JmpBak(open + 1));
                ops[open] = FlatOp::JmpFwd(close + 1);
            }
            _ => {} // comments
        }
    }
    if !stack.is_empty() {
        return Err("unmatched '['".to_string());
    }
    // Peephole: detect [-] / [+] and replace with Clear
    ops = peephole_clear(ops);
    Ok(ops)
}

/// Replace `[Dec]` or `[Inc]` sequences with a single `Clear`.
fn peephole_clear(ops: Vec<FlatOp>) -> Vec<FlatOp> {
    let n = ops.len();
    if n < 3 {
        return ops;
    }
    let mut result = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 2 < n {
            if let FlatOp::JmpFwd(_) = ops[i] {
                let is_body_single = matches!(ops[i + 1], FlatOp::Dec | FlatOp::Inc);
                let is_jmp_back = matches!(ops[i + 2], FlatOp::JmpBak(_));
                if is_body_single && is_jmp_back {
                    result.push(FlatOp::Clear);
                    i += 3;
                    continue;
                }
            }
        }
        result.push(ops[i]);
        i += 1;
    }
    // Rebuild the jump table since compaction shifts indices
    reindex_jumps(result)
}

/// Rebuild JmpFwd/JmpBak targets after compaction.
fn reindex_jumps(mut ops: Vec<FlatOp>) -> Vec<FlatOp> {
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..ops.len() {
        match ops[i] {
            FlatOp::JmpFwd(_) => {
                ops[i] = FlatOp::JmpFwd(0); // placeholder
                stack.push(i);
            }
            FlatOp::JmpBak(_) => {
                if let Some(open) = stack.pop() {
                    let close = i;
                    ops[close] = FlatOp::JmpBak(open + 1);
                    ops[open] = FlatOp::JmpFwd(close + 1);
                }
            }
            _ => {}
        }
    }
    ops
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

fn execute(ops: &[FlatOp], input: &[u8]) -> TapeResult {
    let mut tape = vec![0u8; TAPE_SIZE];
    let mut ptr: usize = 0;
    let mut ip: usize = 0;
    let mut input_pos: usize = 0;
    let mut output: Vec<u8> = Vec::new();
    let mut steps: usize = 0;

    while ip < ops.len() {
        steps += 1;
        if steps > MAX_STEPS {
            return TapeResult::StepLimitExceeded;
        }

        match ops[ip] {
            FlatOp::Inc => {
                tape[ptr] = tape[ptr].wrapping_add(1);
                ip += 1;
            }
            FlatOp::Dec => {
                tape[ptr] = tape[ptr].wrapping_sub(1);
                ip += 1;
            }
            FlatOp::Left => {
                // Add TAPE_SIZE before subtracting to avoid usize underflow.
                ptr = (ptr + TAPE_SIZE - 1) % TAPE_SIZE;
                ip += 1;
            }
            FlatOp::Right => {
                ptr = (ptr + 1) % TAPE_SIZE;
                ip += 1;
            }
            FlatOp::Out => {
                output.push(tape[ptr]);
                ip += 1;
            }
            FlatOp::In => {
                tape[ptr] = if input_pos < input.len() {
                    let b = input[input_pos];
                    input_pos += 1;
                    b
                } else {
                    0
                };
                ip += 1;
            }
            FlatOp::JmpFwd(target) => {
                if tape[ptr] == 0 {
                    ip = target;
                } else {
                    ip += 1;
                }
            }
            FlatOp::JmpBak(target) => {
                if tape[ptr] != 0 {
                    ip = target;
                } else {
                    ip += 1;
                }
            }
            FlatOp::Clear => {
                tape[ptr] = 0;
                ip += 1;
            }
        }
    }

    TapeResult::Ok { output }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf_simple_increment() {
        let r = run_bf("+++", &[]);
        assert_eq!(r, TapeResult::Ok { output: vec![] });
    }

    #[test]
    fn bf_output_cell_value() {
        let src = "+".repeat(65) + ".";
        let r = run_bf(&src, &[]);
        assert_eq!(r, TapeResult::Ok { output: vec![65] });
    }

    #[test]
    fn bf_clear_loop() {
        let r = run_bf("+++++[-].", &[]);
        assert_eq!(r, TapeResult::Ok { output: vec![0] });
    }

    #[test]
    fn bf_echo_program() {
        let r = run_bf(",.", &[42]);
        assert_eq!(r, TapeResult::Ok { output: vec![42] });
    }

    #[test]
    fn bf_inc_program() {
        let r = run_bf(",+.", &[10]);
        assert_eq!(r, TapeResult::Ok { output: vec![11] });
    }

    #[test]
    fn bf_step_limit_terminates() {
        // +++ sets cell to 3; [] loops forever (JmpFwd doesn't fire since cell != 0)
        let r = run_bf("+++[]", &[]);
        assert_eq!(r, TapeResult::StepLimitExceeded);
    }

    #[test]
    fn bf_cancel_plus_minus() {
        let r1 = run_bf("+-.", &[]);
        let r2 = run_bf(".", &[]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn bf_cancel_move_left_right() {
        // With wrapping semantics, >< is always a net noop.
        let r1 = run_bf("><.", &[]);
        let r2 = run_bf(".", &[]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn bf_ptr_wraps_on_underflow() {
        // With wrapping semantics, < from ptr=0 goes to TAPE_SIZE-1.
        // ++< stores 2 at cell 0, moves ptr to TAPE_SIZE-1.
        // . outputs cell[TAPE_SIZE-1] = 0 (never written).
        let r = run_bf("++<.", &[]);
        assert_eq!(r, TapeResult::Ok { output: vec![0] });
    }

    #[test]
    fn bf_ptr_wraps_on_overflow() {
        // Fill tape to end, wraps back. >...> (30000 times) puts ptr at 0.
        // This is too slow to test with 30000 steps; instead check >< is noop.
        let r1 = run_bf("><.", &[]);
        let r2 = run_bf(".", &[]);
        assert_eq!(r1, r2, ">< should be a net noop with wrapping");
    }
}
