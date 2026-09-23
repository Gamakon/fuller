//! THE GIT COMMIT OF THE BINARY, baked in at compile time.
//!
//! A run card names the code as well as the settings, and "the code" must mean
//! the source this binary was BUILT from. Reading `git rev-parse HEAD` at run
//! time names whatever HEAD the checkout happens to be on when the fit starts —
//! which, with several worktrees committing through a day, is routinely not the
//! commit that produced the running binary. Compile time is the only moment the
//! two are the same thing.
//!
//! Neither variable may ever fail the build: a source tree unpacked from a
//! tarball has no git at all, and that is a fine way to build fuller. Both fall
//! back to "unknown".

use std::process::Command;

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // Uncommitted changes mean the commit alone does not identify the source, so
    // the card says so rather than naming a commit that is not what ran.
    let dirty = match git(&["status", "--porcelain"]) {
        Some(out) => !out.trim().is_empty(),
        // No git: "not dirty" would be a claim we cannot make, but the commit is
        // already "unknown" and that is what a reader will look at.
        None => false,
    };
    println!("cargo:rustc-env=FULLER_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FULLER_GIT_DIRTY={dirty}");

    // Rebuild when HEAD moves. In a WORKTREE `.git` is a FILE pointing at
    // `<repo>/.git/worktrees/<name>`, so the path to watch is whatever git
    // itself reports, never a hardcoded `.git/HEAD`.
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={}/HEAD", dir.trim());
    }
}

/// A git command's stdout, or None when git is absent or the command failed.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}
