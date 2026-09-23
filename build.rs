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
    // Uncommitted changes to TRACKED files mean the commit alone does not
    // identify the source, so the card says so rather than naming a commit that
    // is not what ran.
    //
    // `--untracked-files=no` is the whole reading. A plain `git status
    // --porcelain` counts UNTRACKED files, and the checkout where fits actually
    // run has sixty-odd untracked `logs/` entries at any moment — so every real
    // card would say `dirty: true` and the flag would carry no information from
    // the one place it is wanted.
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(out) => !out.trim().is_empty(),
        // No git: "not dirty" would be a claim we cannot make, but the commit is
        // already "unknown" and that is what a reader will look at.
        None => false,
    };
    println!("cargo:rustc-env=FULLER_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FULLER_GIT_DIRTY={dirty}");

    // WHAT MUST MAKE THIS RERUN, and the trap in it. In a WORKTREE `.git` is a
    // FILE pointing at `<repo>/.git/worktrees/<name>`, so every path here is
    // whatever git itself reports and never a hardcoded `.git/HEAD`.
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
    }
    // HEAD ALONE IS NOT ENOUGH, and this was measured rather than reasoned: on a
    // branch HEAD is a SYMREF (`ref: refs/heads/runcard`) whose bytes do not
    // change when you commit or merge — only the ref file does. Watching HEAD
    // alone, a commit followed by a release build baked the PREVIOUS hash into
    // the binary, and the card then named a commit that was not what ran, which
    // is the exact failure the card exists to end.
    //
    // `--git-common-dir` and not `--git-dir`: a worktree's refs live in the main
    // repository's directory, not the worktree's own. A DETACHED HEAD has no
    // symref, `symbolic-ref -q` fails, and HEAD itself then holds the hash, so
    // the line above already covers it.
    if let (Some(common), Some(r)) = (git(&["rev-parse", "--git-common-dir"]), git(&["symbolic-ref", "-q", "HEAD"])) {
        println!("cargo:rerun-if-changed={common}/{r}");
        // A packed ref has no file of its own; this is where it moved to.
        println!("cargo:rerun-if-changed={common}/packed-refs");
    }
    // AND THE SOURCE, so `dirty` is read AFTER the edit that made it dirty
    // rather than staying whatever it was at the last commit.
    for p in ["src", "examples", "Cargo.toml", "build.rs"] {
        println!("cargo:rerun-if-changed={p}");
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
