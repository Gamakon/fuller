//! FITNESS SCHEMES (Andrew, 2026-11-01): "Fitness is a configurable definition, set out per experiment."
//!
//! A fitness function is a NAMED [`Scheme`], chosen per experiment in the run's configuration (`--fitness NAME`, or
//! `EVOLVE_FITNESS=NAME` for phylu's env-configured engine) and recorded in the run's header, its report or run card,
//! and the stream's `run_start` ([`FitnessStart`]). [`SCHEMES`] holds every scheme a runner offers today. A new fitness
//! is a new scheme beside the others; it never changes what an existing name computes.
//!
//! The OLD fitness functions of earlier experiments are schemes too (an ungated stage 2, the study's seven quality
//! inputs, the Futhark kingdom's two bands, the regex generation-0 scaling), each marked NOT TO BE RUN AS IS with the
//! fault that retired it: redefine the fitness function before using one. Each runner's default is its newest scheme.
//! `docs/FITNESS_SCHEMES.md` lists every scheme and the commit of each past experiment.
//!
//! A scheme belongs to one [`Family`] of runners (the runners that compute the same inputs); a runner asks for its
//! scheme with [`scheme_for`], which refuses a name of another family, and takes [`default_for`] when none is given.

use serde::{Deserialize, Serialize};

use crate::futhark_sig::{QualityPart, QUALITY_PARTS};
use crate::runner::{two_level_hff, FitnessStage, TwoLevel};

/// The runners that share one set of fitness inputs; a scheme is defined for exactly one family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// fuller `examples/futhark_evolve.rs`: the signature family's stored cases (`futhark_sig`).
    FutharkSig,
    /// A staged runner with quality inputs of its own (phylu `examples/wgsl_true_case_fit.rs`, fuller
    /// `examples/futhark_pixel_evolve.rs`).
    Staged,
    /// The Futhark kingdom (fuller `futhark_search::run`, phylu `evolve::shape_typed`).
    Shapes,
    /// phylu's regex kingdom (`examples/regex_fit.rs`, the engine's regex HFF).
    Regex,
}

impl Family {
    /// The family's name, as the run card and the errors say it.
    pub fn name(self) -> &'static str {
        match self {
            Family::FutharkSig => "futhark_sig",
            Family::Staged => "staged",
            Family::Shapes => "shapes",
            Family::Regex => "regex",
        }
    }
}

/// What stage 2 of a staged fitness does with an inexact program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageTwo {
    /// The two-level HFF for every program (fuller 922ec1a6): a fast inexact program can rank above an exact one.
    Ungated,
    /// GATED BY CORRECTNESS (fuller f465a5cf): an exact program (every quality input 0) scores the two-level angle,
    /// in `[0, π/2)`; an inexact one scores `π/2 + θ_quality`.
    Gated,
}

/// THE FUTHARK KINGDOM'S UNEXPRESSED FITNESS before the gate (fuller f572a7cc): 1.5.
pub const SHAPES_NOT_EXPRESSED_BEFORE_GATE: f32 = 1.5;

/// The unexpressed fitness of the gated shape fitness: above every gated value (at most π).
pub const SHAPES_NOT_EXPRESSED_GATED: f32 = 4.0;

/// A scheme's definition: what the runner computes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Definition {
    /// [`Family::FutharkSig`]: the quality inputs a case are the parts `quality` (`futhark_sig::quality_inputs_of`);
    /// the fitness is [`staged`] under `stage_two`.
    FutharkSig { quality: &'static [QualityPart], stage_two: StageTwo },
    /// [`Family::FutharkSig`], THE K-MEANS SCHEME (`docs/KMEANS_FITNESS.md`, `futhark_sig::kmeans_converged`): the
    /// program is run to UltraDim's stopping rule on every fitness case, candidate and seed alike, and the five
    /// inputs a case are each relative to the seed's run (`kmeans_inputs`): cosine inertia and mean log10 angular
    /// percentile of the final centroids, wall clock to the stopping point, iterations, inertia gained an iteration.
    /// One stage: TrueNorth over every case's five inputs; no gate, speed is in the same run.
    FutharkConverged,
    /// [`Family::Staged`]: [`staged`] under `stage_two` over the runner's own quality and speed inputs.
    Staged { stage_two: StageTwo },
    /// [`Family::Shapes`], THE TWO BANDS (fuller f3972766): a correct program scores `0.49 · t/(t + t_ref)` (`t` its
    /// traffic), an incorrect one `0.5 + 0.5 (1 − agreement)`, both in `f32`.
    ShapesTwoBand { not_expressed: f32 },
    /// [`Family::Shapes`]: the two-level HFF of the log-scaled disagreements and the three cost ratios under
    /// `stage_two` (`futhark_search::two_level_under`).
    ShapesHff { stage_two: StageTwo, not_expressed: f32 },
    /// [`Family::Regex`]: `column_max_generation0` divides each objective by its maximum over the generation-0
    /// population (the device freezes it at the first evaluation); otherwise HFF reads the objectives as they are (on
    /// `[0, 1]` by construction). `length` adds the rule-length objective (`nfa_len`) to the columns.
    Regex { column_max_generation0: bool, length: bool },
}

/// One named fitness scheme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scheme {
    /// The name a configuration gives (`--fitness NAME`).
    pub name: &'static str,
    pub family: Family,
    /// What it computes, in one sentence.
    pub description: &'static str,
    /// The commit that made it the fitness it is.
    pub commit: &'static str,
    /// Its inputs, in words.
    pub inputs: &'static str,
    /// Its stages, in words.
    pub stages: &'static str,
    /// `Some(fault)` for an OLD scheme: the known fault that retired it and the commit that found it. Such a scheme
    /// is NOT TO BE RUN AS IS; a runner given one says so in its header ([`Scheme::card_line`]).
    pub fault: Option<&'static str>,
    pub definition: Definition,
}

/// The warning every old scheme carries.
pub const OLD_SCHEME_WARNING: &str = "NOT TO BE RUN AS IS: this fitness has a known fault; redefine the fitness function before using it";

/// EVERY FITNESS SCHEME A RUNNER OFFERS. Never change what an entry computes: add a scheme.
pub const SCHEMES: &[Scheme] = &[
    Scheme {
        name: "futhark-slope4-gated",
        family: Family::FutharkSig,
        description: "quality: the slope check's formula (futhark_sig::QUALITY_PARTS); stage 2 gated by correctness",
        commit: "fuller f465a5cf",
        inputs: "per fitness case: wrong share; (1-r)/2 for Spearman's and Pearson's r with the correct output (absent for a constant answer); shares off by over 50 % and 1 %; speed: r/(1+r) a timing slot",
        stages: "1: TrueNorth over quality until the best is exact; 2: two-level HFF for an exact program, pi/2 + theta_quality for an inexact one",
        fault: None,
        definition: Definition::FutharkSig { quality: QUALITY_PARTS, stage_two: StageTwo::Gated },
    },
    // NOT TO BE RUN AS IS: this fitness has a known fault (stage 2 ungated: a fast inexact program ranks above an
    // exact one; found and fixed in fuller f465a5cf). Redefine the fitness function before using it.
    Scheme {
        name: "futhark-slope4",
        family: Family::FutharkSig,
        description: "futhark-slope4-gated with stage 2 ungated (the 2026-10-29 reruns)",
        commit: "fuller b8163635",
        inputs: "as futhark-slope4-gated",
        stages: "1: TrueNorth over quality; 2: two-level HFF, ungated",
        fault: Some("stage 2 ungated: a fast inexact program ranks above an exact one (fuller f465a5cf)"),
        definition: Definition::FutharkSig { quality: QUALITY_PARTS, stage_two: StageTwo::Ungated },
    },
    // NOT TO BE RUN AS IS: this fitness has a known fault (the study's seven inputs reward the all-zeros output and
    // fail the slope check on histogram, found by the slope check of 2026-10-29, fuller 6f76baac/b8163635; and stage 2
    // is ungated, fuller f465a5cf). Redefine the fitness function before using it.
    Scheme {
        name: "futhark-study7",
        family: Family::FutharkSig,
        description: "the 2026-10-27 study's seven linear quality inputs a case; stage 2 ungated",
        commit: "fuller 38216485",
        inputs: "per fitness case: wrong share; shares off by over 1, 10, 50 % of the correct magnitude; mean and largest error over the correct output's range (e/(1+e)); share of neighbouring pairs in the wrong order; speed: r/(1+r) a timing slot",
        stages: "1: TrueNorth over quality; 2: two-level HFF, ungated",
        fault: Some("the seven inputs reward the all-zeros output and fail the slope check on histogram (fuller 6f76baac, b8163635); stage 2 ungated (fuller f465a5cf)"),
        definition: Definition::FutharkSig { quality: &[QualityPart::Study], stage_two: StageTwo::Ungated },
    },
    Scheme {
        name: "futhark-kmeans-converged",
        family: Family::FutharkSig,
        description: "k-means run to UltraDim's stopping rule (inertia change < 1e-4 or 100 iterations); per case, relative to the seed's run: cosine inertia and mean log10 angular percentile of the final centroids, seconds to the stopping point, iterations, inertia gained an iteration; one stage, no gate (docs/KMEANS_FITNESS.md)",
        commit: "fuller bd929652",
        inputs: "per fitness case, each r/(1+r) of a ratio to the seed's run (1/2 is equal to the seed): cosine inertia; 10^(mean log10 p - seed's); seconds to the stopping point; iterations; seed's inertia gained an iteration over the candidate's; a run that does not finish is 1 on all five",
        stages: "one: TrueNorth over every case's five inputs",
        fault: None,
        definition: Definition::FutharkConverged,
    },
    Scheme {
        name: "staged-gated",
        family: Family::Staged,
        description: "quality only until the best is exact, then the two-level HFF of quality and speed, gated by correctness",
        commit: "fuller f465a5cf",
        inputs: "the runner's own quality inputs; speed: its timing slots",
        stages: "1: TrueNorth over quality; 2: two-level HFF for an exact program, pi/2 + theta_quality for an inexact one",
        fault: None,
        definition: Definition::Staged { stage_two: StageTwo::Gated },
    },
    // NOT TO BE RUN AS IS: this fitness has a known fault (stage 2 ungated: a fast inexact program ranks above an
    // exact one; found and fixed in fuller f465a5cf). Redefine the fitness function before using it.
    Scheme {
        name: "staged",
        family: Family::Staged,
        description: "quality only until the best is exact, then the two-level HFF of quality and speed, ungated",
        commit: "fuller 922ec1a6",
        inputs: "the runner's own quality inputs; speed: its timing slots",
        stages: "1: TrueNorth over quality; 2: two-level HFF, ungated",
        fault: Some("stage 2 ungated: a fast inexact program ranks above an exact one (fuller f465a5cf)"),
        definition: Definition::Staged { stage_two: StageTwo::Ungated },
    },
    Scheme {
        name: "shapes-hff-gated",
        family: Family::Shapes,
        description: "the two-level HFF of the log-scaled disagreements and the three cost ratios, gated by correctness; unexpressed 4.0",
        commit: "fuller f572a7cc",
        inputs: "quality: log_scaled disagreement a suite input set; speed: traffic, allocations, depth, each x/(x + reference)",
        stages: "one: two-level HFF for a correct program, pi/2 + theta_quality for an incorrect one",
        fault: None,
        definition: Definition::ShapesHff { stage_two: StageTwo::Gated, not_expressed: SHAPES_NOT_EXPRESSED_GATED },
    },
    // NOT TO BE RUN AS IS: this fitness has a known fault (ungated: a wrong but cheap program ranks above an exact
    // but dear one; found and fixed in fuller f572a7cc). Redefine the fitness function before using it.
    Scheme {
        name: "shapes-hff",
        family: Family::Shapes,
        description: "the two-level HFF of the log-scaled disagreements and the three cost ratios, ungated; unexpressed 1.5",
        commit: "fuller b1bccf66",
        inputs: "as shapes-hff-gated",
        stages: "one (two-level HFF, ungated)",
        fault: Some("ungated: a wrong but cheap program ranks above an exact but dear one (fuller f572a7cc)"),
        definition: Definition::ShapesHff { stage_two: StageTwo::Ungated, not_expressed: SHAPES_NOT_EXPRESSED_BEFORE_GATE },
    },
    // NOT TO BE RUN AS IS: this fitness has a known fault (it ranks a correct program on traffic alone, allocations
    // and depth unread, and is not the runners' two-level HFF; replaced in fuller b1bccf66). Redefine the fitness
    // function before using it.
    Scheme {
        name: "shapes-two-band",
        family: Family::Shapes,
        description: "correct: 0.49 * traffic/(traffic + reference traffic); incorrect: 0.5 + 0.5 * (1 - agreement); unexpressed 1.5",
        commit: "fuller f3972766",
        inputs: "correctness over the input suite, the graded agreement, the static traffic",
        stages: "one",
        fault: Some("a correct program is ranked on traffic alone (allocations and depth unread), not by the runners' two-level HFF (fuller b1bccf66)"),
        definition: Definition::ShapesTwoBand { not_expressed: SHAPES_NOT_EXPRESSED_BEFORE_GATE },
    },
    Scheme {
        name: "regex-raw",
        family: Family::Regex,
        description: "the regex objectives read as they are (on [0, 1] by construction), no rule-length objective",
        commit: "phylu 759f0936",
        inputs: "the regex columns (env-selected), unnormalised",
        stages: "one",
        fault: None,
        definition: Definition::Regex { column_max_generation0: false, length: false },
    },
    Scheme {
        name: "regex-raw-length",
        family: Family::Regex,
        description: "regex-raw with the rule length an objective (what EVOLVE_REGEX_PARSIMONY=1 selects)",
        commit: "phylu 759f0936",
        inputs: "the regex columns (env-selected) plus nfa_len, unnormalised",
        stages: "one",
        fault: None,
        definition: Definition::Regex { column_max_generation0: false, length: true },
    },
    // NOT TO BE RUN AS IS: this fitness has a known fault (each objective over its generation-0 maximum makes the
    // angle depend on the layout's first population, so two runs' angles are not on one scale; found in phylu
    // 759f0936). Redefine the fitness function before using it.
    Scheme {
        name: "regex-colmax",
        family: Family::Regex,
        description: "the regex objectives divided by their generation-0 column maximum, no rule-length objective",
        commit: "phylu 8b05aaa5",
        inputs: "the regex columns (env-selected), each over its generation-0 maximum",
        stages: "one",
        fault: Some("the generation-0 column maximum makes the angle depend on the layout's first population; runs are not on one scale (phylu 759f0936)"),
        definition: Definition::Regex { column_max_generation0: true, length: false },
    },
];

/// The default scheme of each family: the one its newest experiments use.
pub fn default_for(family: Family) -> &'static Scheme {
    let name = match family {
        Family::FutharkSig => "futhark-slope4-gated",
        Family::Staged => "staged-gated",
        Family::Shapes => "shapes-hff-gated",
        Family::Regex => "regex-raw",
    };
    scheme(name).expect("every family's default is in SCHEMES")
}

/// The scheme of this name, of any family.
pub fn scheme(name: &str) -> Result<&'static Scheme, String> {
    SCHEMES.iter().find(|s| s.name == name).ok_or_else(|| format!("no fitness scheme {name:?}; the schemes are {} (an old result is reproduced at its own commit: docs/FITNESS_SCHEMES.md)", SCHEMES.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")))
}

/// The names of a family's schemes.
pub fn names_for(family: Family) -> Vec<&'static str> {
    SCHEMES.iter().filter(|s| s.family == family).map(|s| s.name).collect()
}

/// The scheme a Futhark problem is run with: `name`, or the problem's own default ([`crate::futhark_sig::Problem::fitness`])
/// when `None`. A scheme of another family, or of another kind than the problem's default (an answer-scored scheme on
/// k-means, which has no answers; the k-means scheme on a problem with answers), is an error that says so.
pub fn scheme_for_problem(p: &crate::futhark_sig::Problem, name: Option<&str>) -> Result<&'static Scheme, String> {
    let own = scheme(p.fitness)?;
    let Some(name) = name else { return Ok(own) };
    let s = scheme_for(Family::FutharkSig, Some(name))?;
    let converged = |d: Definition| matches!(d, Definition::FutharkConverged);
    if converged(s.definition) != converged(own.definition) {
        return Err(format!("the fitness scheme {name:?} is not of {}'s kind: {} is scored {}, and its schemes are {}", p.name, p.name, if converged(own.definition) { "on its clustering objective run to UltraDim's stopping rule (docs/KMEANS_FITNESS.md), with no stored answers" } else { "against its stored answers" }, names_for(Family::FutharkSig).into_iter().filter(|n| scheme(n).is_ok_and(|x| converged(x.definition) == converged(own.definition))).collect::<Vec<_>>().join(", ")));
    }
    Ok(s)
}

/// The scheme a runner of `family` was configured with: `name`, or the family's default when `None`. A name of
/// another family is an error that lists the family's own.
pub fn scheme_for(family: Family, name: Option<&str>) -> Result<&'static Scheme, String> {
    let Some(name) = name else { return Ok(default_for(family)) };
    let s = scheme(name)?;
    if s.family != family {
        return Err(format!("the fitness scheme {name:?} is for the {} runners, not {}; this runner's schemes are {}", s.family.name(), family.name(), names_for(family).join(", ")));
    }
    Ok(s)
}

impl Scheme {
    /// The quality parts of an answer-scored `FutharkSig` scheme.
    pub fn quality_parts(&self) -> Result<&'static [QualityPart], String> {
        match self.definition {
            Definition::FutharkSig { quality, .. } => Ok(quality),
            Definition::FutharkConverged => Err(format!("the fitness scheme {} scores no program against stored answers: it runs the program to UltraDim's stopping rule and has no quality parts (docs/KMEANS_FITNESS.md)", self.name)),
            _ => Err(format!("our software has a bug: the fitness scheme {} is not a futhark_sig scheme", self.name)),
        }
    }

    /// Whether this is the k-means scheme (`Definition::FutharkConverged`): one stage, no answers, no gate.
    pub fn is_converged(&self) -> bool {
        matches!(self.definition, Definition::FutharkConverged)
    }

    /// The stage-2 rule of a staged scheme (`FutharkSig`, `Staged`, `ShapesHff`).
    pub fn stage_two(&self) -> Result<StageTwo, String> {
        match self.definition {
            Definition::FutharkSig { stage_two, .. } | Definition::Staged { stage_two } | Definition::ShapesHff { stage_two, .. } => Ok(stage_two),
            _ => Err(format!("our software has a bug: the fitness scheme {} has no stage 2", self.name)),
        }
    }

    /// The fitness of a staged scheme (`FutharkSig`, `Staged`): [`staged`] under its stage-2 rule. The k-means scheme
    /// has one stage and reads its inputs through [`converged`]: it is an error here.
    pub fn staged(&self, stage: FitnessStage, quality: &[f64], speed: &[f64]) -> Result<TwoLevel, String> {
        staged(self.stage_two()?, stage, quality, speed)
    }

    /// What the stream's `run_start` and the run card record of the scheme.
    pub fn start(&self) -> FitnessStart {
        FitnessStart { name: self.name.to_string(), family: self.family.name().to_string(), commit: self.commit.to_string(), description: self.description.to_string() }
    }

    /// One line for a run's header and report: `fitness\t<name> (<commit>): <description>`, and for an old scheme
    /// the warning and its fault.
    pub fn card_line(&self) -> String {
        match self.fault {
            None => format!("fitness\t{} ({}): {}", self.name, self.commit, self.description),
            Some(f) => format!("fitness\t{} ({}): {} -- {OLD_SCHEME_WARNING}: {f}", self.name, self.commit, self.description),
        }
    }
}

/// The fitness scheme as the stream's `run_start` carries it (`telemetry::RunStart::fitness`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessStart {
    pub name: String,
    pub family: String,
    pub commit: String,
    pub description: String,
}

/// THE K-MEANS SCHEME'S FITNESS (`Definition::FutharkConverged`, `docs/KMEANS_FITNESS.md`): TrueNorth over every
/// fitness case's five inputs (`futhark_sig::kmeans_converged::kmeans_inputs`, each in [0, 1], 0 best, 1/2 equal to
/// the seed), in case order; one stage, no speed group (the wall clock is one of the five). No inputs is an error:
/// a problem with no fitness case cannot be scored.
pub fn converged(inputs: &[f64]) -> Result<TwoLevel, String> {
    if inputs.is_empty() {
        return Err("our software has a bug: the k-means scheme was given no inputs (no fitness case)".into());
    }
    if let Some(x) = inputs.iter().find(|x| !(0.0..=1.0).contains(*x)) {
        return Err(format!("our software has a bug: a k-means input {x} is outside [0, 1]"));
    }
    let q = crate::score::truenorth_angle(inputs);
    Ok(TwoLevel { fitness: q, quality: q, speed: 0.0 })
}

/// A STAGED FITNESS: stage 1 is TrueNorth over the quality inputs (the speed inputs must be empty, and the speed
/// angle is 0); stage 2 is [`stage_two`] under `rule`.
pub fn staged(rule: StageTwo, stage: FitnessStage, quality: &[f64], speed: &[f64]) -> Result<TwoLevel, String> {
    match stage {
        FitnessStage::Quality => {
            if !speed.is_empty() {
                return Err("our software has a bug: speed inputs in stage 1 (quality only)".into());
            }
            let q = if quality.is_empty() { 0.0 } else { crate::score::truenorth_angle(quality) };
            Ok(TwoLevel { fitness: q, quality: q, speed: 0.0 })
        }
        FitnessStage::QualityAndSpeed => Ok(stage_two(rule, quality, speed)),
    }
}

/// The two-level HFF of `quality` and `speed` under the stage-2 `rule`: ungated, or gated by correctness (an exact
/// program, every quality input 0, scores the two-level angle in `[0, π/2)`; an inexact one `π/2 + θ_quality`, so no
/// inexact program, however fast, ranks above an exact one).
pub fn stage_two(rule: StageTwo, quality: &[f64], speed: &[f64]) -> TwoLevel {
    let t = two_level_hff(quality, speed);
    match rule {
        StageTwo::Ungated => t,
        StageTwo::Gated if quality.iter().all(|x| *x == 0.0) => t,
        StageTwo::Gated => TwoLevel { fitness: std::f64::consts::FRAC_PI_2 + t.quality, ..t },
    }
}

#[cfg(test)]
mod tests;
