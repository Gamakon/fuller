//! THE KILL TRIGGER — the one way to stop an evolution run, in fuller and phylu alike
//! (`docs/STANDARD_runners.md`).
//!
//! The run writes its run file when it starts (`OUT/killme.trigger` for a runner with an output directory, or the
//! path `EVOLVE_RUN_FILE` names for phylu's engine) and ends cleanly, after the current generation, as soon as the
//! file is gone. A signal is never used: a kill with a command buffer on the device wedged the GPU (2026-10-10).

/// The run file's name inside a run's output directory.
pub const RUN_FILE: &str = "killme.trigger";

/// The environment variable naming the run file for a run with no output directory of its own (phylu's engine).
/// Unset: no file, the run ends only by its budgets.
pub const RUN_FILE_ENV: &str = "EVOLVE_RUN_FILE";

/// Whether the run may continue: the run file `EVOLVE_RUN_FILE` names exists, or none was named.
pub fn run_file_present() -> bool {
    run_file_present_at(std::env::var(RUN_FILE_ENV).ok().as_deref())
}

/// The rule itself: no name, or an empty one, means no run file and the run continues; a name means the run
/// continues only while the file exists.
pub fn run_file_present_at(path: Option<&str>) -> bool {
    match path {
        Some(p) if !p.is_empty() => std::path::Path::new(p).exists(),
        _ => true,
    }
}

/// The run file's text: the process id and the time, so a reader of the directory knows whose it is.
fn run_file_text() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("pid {}\nstarted_unix {now}\ndelete this file to end the run\n", std::process::id())
}

/// Write the run file `EVOLVE_RUN_FILE` names, when it names one.
pub fn write_run_file() -> Result<(), String> {
    if let Ok(p) = std::env::var(RUN_FILE_ENV) {
        if !p.is_empty() {
            std::fs::write(&p, run_file_text()).map_err(|e| format!("run file {p}: {e}"))?;
        }
    }
    Ok(())
}

/// A run file at a known path: written at the start, checked between generations, removed at the end.
#[derive(Debug)]
pub struct RunFile {
    path: std::path::PathBuf,
}

impl RunFile {
    /// Write the run file at `path`.
    pub fn create(path: &std::path::Path) -> Result<RunFile, String> {
        std::fs::write(path, run_file_text()).map_err(|e| format!("run file {}: {e}", path.display()))?;
        Ok(RunFile { path: path.to_path_buf() })
    }

    /// The file's path.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Whether the run may continue: the file is still there.
    pub fn present(&self) -> bool {
        self.path.exists()
    }

    /// Remove the file at the run's end. Already gone (the operator stopped the run) is the ordinary case.
    pub fn remove(&self) {
        if let Err(e) = std::fs::remove_file(&self.path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("run file {}: {e}", self.path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No run file named: the run continues. Named: it continues while the file exists and ends once it is deleted.
    #[test]
    fn the_run_continues_while_its_file_exists() {
        assert!(run_file_present_at(None));
        assert!(run_file_present_at(Some("")));
        let path = std::env::temp_dir().join(format!("fuller_run_file_{}", std::process::id()));
        let f = RunFile::create(&path).expect("the run file is written");
        assert!(f.present() && run_file_present_at(path.to_str()));
        std::fs::remove_file(&path).expect("the operator deletes it");
        assert!(!f.present() && !run_file_present_at(path.to_str()));
        f.remove();
    }
}
