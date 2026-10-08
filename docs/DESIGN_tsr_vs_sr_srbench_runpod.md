# EXPERIMENTAL DESIGN: typed vs untyped symbolic regression on SRBench, run on Runpod

## Context

**Objective.** One controlled experiment that a paper can carry: does the
typed configuration (TSR, transcendental depth ≤ 2) recover more SRBench
ground-truth laws than the untyped configuration of the same engine, under
identical conditions, and at what budget? Every number comes from SRBench's
own split, seeds, checker and aggregation, so the headline is comparable to
the published SRBench table and to AIFeynman's 53 % at zero noise.

**Approach.** Fits on one rented RTX 4090; everything slow or durable on the
laptop. The protocol is the pre-registered, Codex-reviewed one in
`phylu/docs/PLAN_srbench_competition.md` v2 (SRBench's 75/25 split per law ×
seed, sterile environment, no truth and no test row anywhere near the fit,
SRBench's unmodified `assess_symbolic_model` offline), with one addition: the
untyped configuration runs beside the typed one in the same driver. Each fit
runs to the 90 s cap with the -19 gate as early stop; the evaluation counter
lets the same run be censored at SRBench's 1,000,000 evaluations afterwards,
so one spend gives two numbers: solved within 1 M evaluations (SRBench's rule)
and solved within 90 s (the engine's working budget).

**Unique feature.** Nothing is bought until the whole pipeline, including every
figure, has run end to end on the one-seed data already on disk (AB1). The
rented hour only repeats a rehearsed procedure with more seeds.

**Status.** AB1 (dev seed 7013, both configurations, 133 laws, 90 s, 4090) is
done and on the laptop at `phylu/logs/RUNPOD_AB1/`; its SRBench check is
running here. Pod `ab1-4090` is stopped. The driver for the reviewed protocol
(`scripts/srbench_tf_run.py`) exists and was dry-run; it lacks the untyped
configuration and a fit-only phase. Nothing of the reported run is launched.

**Why this design.** AB1 says the gate counts are close (TSR 79, untyped 81 of
133 on one seed), so the paper's result may be "no difference at this budget",
and that must be defensible: pre-registered seeds, SRBench's protocol, a paired
test over laws. The reviewed protocol exists; AB1 deviated from it on the
split, the ranking rows and the seed, so AB1 stays a development result.

## 1. Hypotheses and what the data decides

- **H1 (recovery).** TSR's SRBench solve rate ≥ untyped's at 1 M evaluations and
  at 90 s. Decided by the per-law solve fraction over ten seeds, paired across
  the 130 scored problems: Wilcoxon signed-rank on the per-law difference, and
  a bootstrap over seeds for the interval on each headline mean.
- **H2 (efficiency).** Among laws both configurations solve, TSR reaches the
  gate in fewer evaluations. Decided by paired evaluations-to-gate per law
  (median over seeds), Wilcoxon.
- **H3 (where the law is found).** The canonicaliser (fuller's kitchen sink)
  turns gate passers into SRBench solutions at the same rate for both
  configurations. Decided by the found-via stage counts.
- A null on H1 is a result: the typed projection buys closure and a smaller
  search space, not recovery, at this budget. It is reported either way.

## 2. Settings, fixed, and why

| setting | value | why |
|---|---|---|
| laws | the 133 SRBench ground-truth problems (130 scored; SRBench excludes feynman_test_10, I_26_2, I_30_5) | SRBench's set, PMLB layout with metadata.yaml |
| seeds | PRIMARY: SRBench's first ten (23654 15795 860 5390 16850 29910 4426 21962 14423 28020), pre-registered in the v2 plan; AB1's 7013 is the development result, reported separately, not one of the ten | what published tables use; 23654 and 15795 were watched during judge-side fixes, stated in the paper. v2's SECONDARY ten (the contamination control) is OUT of this spend (another $22); run it only if H1's verdict turns on those two seeds |
| split | sklearn `train_test_split(train_size=0.75, random_state=seed)` per law × seed; the fit and the canonicaliser read only the 75 % | SRBench's own partition; the 25 % is only read by the offline scorer |
| configurations | TSR: `EVOLVE_TYPED_DEPTH=2`; untyped: unset. Nothing else differs | the one-variable A/B |
| engine card | CASCADE2: 3 genes, head 34, intake 1000 + champion 1000, pump every 50, cohort merge 10 000, typed snap every 10 (subtree), fold/snap/beam every 20, `EVOLVE_WRAPPERS=identity`, `EVOLVE_DIVERSITY=1` | the production card; identity wrapper so a solution is the engine's form, not a log/sqrt wrap |
| stop | early stop at the law bar: log10 p ≤ -19 and 1-R² ≤ 1e-10 (train and val); cap 90 s wall clock; no evaluation cap in the run | Andrew's ruling, 2026-10-08: 1e-10 finds an actual law, anything looser is approximate; if a run is not stopped by a law it runs to the cap. Matches the v2 card (at 1e-6 the gate fired on 28 near-misses in 531 trials). AB1 ran at 1e-6, one more reason it is a development result. Build item 1a changes `law_runner.py`'s fixed bar to 1e-10 |
| 1 M censor | post hoc from `evaluations.total` and the gate generation (30 260 evaluations per generation, so 1 M ≈ 33 generations); the canonicaliser's own scoring is post-processing and not counted | SRBench's rule, recovered from the same run; AB1: 108 of 160 gate passes fell within 1 M |
| canonicaliser | kitchen sink on train.tsv, HFF-ranked, one form emitted, laptop (Metal) | the finisher the v2 plan built; deterministic on a device, and the device is recorded; Metal/Vulkan float parity is a stated threat |
| verdict | SRBench `assess_symbolic_model` on the emitted form, laptop, offline | the only score that counts; sympy never on rented time |
| aggregation | per problem the fraction of ten trials solved, unweighted mean over 130, in % | SRBench's results notebook |
| device | one RTX 4090 (Vulkan, driver 580), concurrency 1 | AB1's device; the 4090 is kernel-bound so concurrency only shares the queue and would change generations per fit |

## 3. What one run produces

Per law × seed × configuration: `card.json` (every setting, both commits,
device), `run.log`, `hof.tsv`, `stream.jsonl` (snapshots with evaluations by
source, distinct genes and chromosomes, best HFF; model records with the
generation found), `train.tsv`; from the laptop phase: the canonicalised form,
the SRBench JSON and `.updated` verdict. Per run: `results.tsv`, `summary.txt`.

## 4. Figures and tables, produced by scripts tested on AB1 first

| id | what | fields |
|---|---|---|
| T1 | headline: % solved, TSR vs untyped, at 1 M evaluations and at 90 s, bootstrap 95 % CI over seeds; Feynman and Strogatz subsets beside AIFeynman's 55.78 % / 27.14 % | verdicts, evaluations.total, gate generation |
| F1 | solve fraction vs budget, two curves per axis (evaluations with the 1 M line; seconds with the 90 s line) | gate generation, evaluations per generation, seconds |
| F2 | per-law paired solve fraction, TSR (x) vs untyped (y), 130 points, Feynman/Strogatz marked | T1's per-law table |
| F3 | evaluations-to-gate, paired, laws solved by both | stream model records |
| F4 | found via: engine form / canonicaliser stage (forms, ratio, snap, fold, physics), per configuration | ks provenance line |
| F5 | diversity: distinct chromosomes and max gene copies vs generation, median over laws, per configuration | stream snapshots |
| T2 (appendix) | law × configuration × seed: solved / gate / none, generations, evaluations, seconds | results.tsv |

The scripts live at `phylu/scripts/paper/` (`fig1_budget.py` … `tab1_headline.py`),
each reading a results root; the go/no-go is every one rendering from AB1.

## 5. The run on Runpod, step by step

1. **Build (laptop, before any rental; about one working day).** (a)
   `srbench_tf_run.py`: `--typed-depth 2|none`, `--phase fit|finish` (fit
   writes card, run.log, hof, stream; finish runs the kitchen sink later),
   the stop at 1e-10 (and `law_runner.py`'s fixed bar moved to 1e-10), train.tsv written ONCE per law × seed and read by both
   configurations, then gzipped after the second fit; each fit's stream
   gzipped on exit (the stream is per-edit events, typed_snap and fold, not
   gated by progress_every; gzip is the cut). (b) Pod wrapper
   `scripts/runpod/sweep_tf.sh`: laws in order, both configurations, ten
   seeds, `--phase fit`, output `/workspace/TF10/`. (c) Laptop
   `scripts/runpod/finish_tf.sh`: `--phase finish` over a results root, P=4,
   then `srbench_tf_score.py`. (d) The figure scripts of §4. (e) Rebuild the
   image at the new phylu commit (add `rsync`), push to the caveserver registry
   and, for replication, to Docker Hub as `gamakon/phylu-srbench:<commit>`.
2. **Mechanical rehearsal on AB1 (laptop).** `finish` + score + every figure
   over `logs/RUNPOD_AB1`: exercises the scripts, not the protocol (AB1's fits
   saw whole files, not the 75 % split). The protocol rehearsal is step 3.
3. **Dry run on the pod, 15 min, about $0.20.** Resume `ab1-4090` (or a new pod
   from the image, `NVIDIA_DRIVER_CAPABILITIES=all`, 20 GB disk); delete
   `/workspace/AB1*` (copied, verified); `vulkaninfo --summary` lists the GPU;
   two laws × two dev seeds × both configurations through steps 1b–1c end to
   end, figures included.
4. **The reported run.** Announce, then `nohup bash sweep_tf.sh`; tail
   `/workspace/TF10/sweep.log`; the web watcher on 8787. 2660 fits at AB1's
   average 40 s: about 30 h, $22 at $0.74/h; worst case (no early stops) 67 h,
   $50. Copy out twice: at the halfway mark and at the end.
5. **Copy out.** `ssh … "cd /workspace && tar -cf - TF10" | tar -xf -` into
   `phylu/logs/TF10/` (no rsync in the current image); verify the file count
   and 2660 card.json against the pod; then **stop the pod** at once and
   terminate it after the laptop phase has run.
6. **Laptop phase.** finish + score (2660 trials: kitchen sink ≈ 1 min each at
   P=4 ≈ 11 h; sympy verdicts on gate passers, hours), figures, T1/T2.
7. **Record.** Commit `logs/TF10/{results.tsv,summary.txt,figures/}` and the
   run card; the paper's numbers are read from these files only.

**Money.** $22 expected, $50 worst case, plus the dry run. With about $17 in
the account, top up $15 for the expected case or $35 to cover the worst case
without a mid-run stop.

**Disk on the pod.** AB1 wrote 2.1 GB for 266 fits: 1.0 GB of per-law tsv
copies, 382 MB of streams, 223 MB of hofs. TF10 raw would be about 7.5 GB of
train.tsv (up to 12 MB each × 1330) plus 6 GB of streams and hofs, over the
20 GB disk with datasets. With train.tsv shared by the two configurations and
gzipped after both fits, and streams gzipped on exit, TF10 is under 4 GB.
AB1 is removed from the pod first (copied and verified).

## 6. Replication package

- Image `gamakon/phylu-srbench:<commit>` on Docker Hub: the two binaries
  (`evolve_fit`, `kitchen_sink`, x86_64, Vulkan), the scripts, SRBench's
  checker at `409beb3`, the 133 PMLB datasets with metadata (PMLB licence
  permits redistribution). Sources: phylu, fuller, hff at the commits in
  `REFS`; release of the sources is a separate decision.
- Pod recipe: any NVIDIA pod, `NVIDIA_DRIVER_CAPABILITIES=all`, 20 GB disk, the
  image's default command (sshd), then `bash scripts/runpod/sweep_tf.sh`.
- The seed list, the law order (`experiments/order_prioritized.txt`), the card
  and the aggregation script are in the image; the figures regenerate from
  `results.tsv`.

## 7. The paper, sketched

**Title.** Does a type system help symbolic regression find laws? A controlled
SRBench study of typed and untyped Gene Expression Programming on the GPU.

**Claim.** On SRBench's 130 ground-truth problems, under SRBench's protocol,
typed GEP (transcendental depth ≤ 2) solves X % within 1 M evaluations and
Y % within 90 s; untyped GEP, same engine and card, solves X′ % and Y′ %. The
difference is / is not significant (Wilcoxon over laws, p); TSR reaches the
gate in Z × fewer evaluations where both solve; the canonicaliser converts
W % of gate passes into certified solutions for both. Against AIFeynman's 53 %
at zero noise, quoted by subset.

**Sections.** 1 Introduction: the closure argument for types (from the GECCO
paper), the question. 2 Engine and configurations: one paragraph each on the
geneframe, the typed projection, the law bar, the canonicaliser; the single
variable. 3 Protocol: SRBench's split, seeds, stop, censor, verdict,
aggregation; the sterile-environment guarantees; the pre-registration. 4
Results: T1, F1, F2, F3, F4, F5. 5 Threats: two watched seeds; the 90 s cap on a
4090 is not SRBench's compute; the kernel-bound device; the gate as early
stop costs trials on near-misses (28 false stops in 531 at 1e-6, measured). 6
Replication: §6 above. Appendix: T2.

**What the run proves.** Whether the typed kingdom changes recovery or only
efficiency, with SRBench's own numbers, and a reproducible recipe anyone with
$25 and a GPU pod can rerun.

## 8. Go / no-go

Go when: every §4 figure renders from AB1; the pod dry run writes the files
§3 lists and the laptop phase scores them; the account holds the expected cost
plus margin. Stop at any point the pod is idle longer than a copy takes.
