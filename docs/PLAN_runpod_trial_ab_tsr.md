# PLAN: first Runpod trial — the TSR vs SR A/B on one H100

## Context

**Objective.** Answer the question the paper cannot: does typed symbolic
regression (TSR, depth ladder F/T1/T2) recover more SRBench laws than untyped
SR under identical conditions? Same seed, same settings, same device, same
stop. Secondarily: prove the engine runs on a rented Linux GPU, and that a
law job writes everything we need and we get the files back.

**Approach.** One H100 pod. A container holding phylu, fuller, the SRBench
data and the checker. One law job per law carrying both arms (TSR, untyped),
one dev seed, -19 gate as the stop, plain OLS (identity wrapper only), the 90 s
clock as a spending cap. Up to 8 jobs concurrently on the one GPU, at the knee
the first hour measures. All outputs to a network volume, collected to the
laptop at the end.

**Unique feature.** The controlled comparison is cheap because the whole run is
under $3 of GPU and under an hour; the expensive part is the plumbing, built
once and reused for the official run and the noise study.

**Status.** Plan. Nothing built on the Runpod side. Runner, counters, OLS
switch and web watcher exist in phylu; Runpod account signed in, no credit, no
API key.

**Why this order.** Everything downstream (official seeds, noise levels)
is the same job with more arguments, so the first trial should exercise every
part once: image, device, concurrency, outputs, collection.

## What is already built (phylu)

| part | commit | note |
|---|---|---|
| `scripts/law_runner.py` — one law, N seeds, gate stop, device/commits on rows | 26ff1cb, 821bec5d | `--stop clock --evaluations none` for this trial |
| evaluation counters by source, on stream and watcher | e61bc23d | always on |
| diversity telemetry (`EVOLVE_DIVERSITY=1`) | c49f9de, 48bc3b8 | on for this trial |
| `EVOLVE_WRAPPERS=identity` | 78c549f4 | plain OLS |
| A/B driver (`logs/AB_TYPED/sweep.sh`) | — | laptop form; the container form is step 2 |
| `scripts/phylu_web_watch.py --port` | d5c60f37 | the page we watch |
| checkpoint / resume | — | not needed at 90 s fits |

## Gaps, in build order

1. **Runner: both arms in one job.** `law_runner.py` runs one arm. Add
   `--arms tsr,untyped` (TSR = `EVOLVE_TYPED_DEPTH=2`, untyped = unset), one
   folder per arm under the law, one `result.json` with both. *(phylu, small.)*
2. **Checker in the container.** `scripts/ks_one.py` and the oracle server
   have laptop paths; make dataset and binary paths arguments/env; the oracle
   (Python + sympy + SRBench's `assess_symbolic_model`) must install from
   requirements inside the image. *(phylu, medium — the one real porting job.)*
3. **Build on the pod, not an image, for this trial.** Cross-building Rust
   for amd64 under emulation on the Mac would eat the day. Use a stock Ubuntu
   or CUDA base pod, install Rust and `libvulkan1` + `vulkan-tools` + Python
   with sympy and the SRBench checker deps, clone both repos at pinned commits
   onto the network volume at `/workspace`, `cargo build --release --features
   gpu` there for `evolve_fit` and `kitchen_sink`. The H100 pod mounts the
   same volume and reuses the binaries (same x86_64; wgpu is GPU-agnostic).
   The data must be the PMLB layout the checker expects, each law's folder
   with its `.tsv.gz` **and its `metadata.yaml`** (the ground-truth formula
   SRBench's `assess_symbolic_model` reads), copied from the hff clone. A
   registry image is for the later 30-worker serverless run, not this trial.
4. **Sweep driver for the pod.** A script that reads the law list, keeps N
   jobs running (`xargs -P N` is enough), writes to `/workspace/AB1/<law>/`,
   starts the web watcher on port 8787. *(small.)*
5. **Collector.** After the sweep: `tar` the volume folder, pull with
   `runpodctl receive` or the volume's S3 interface, verify 133 `result.json`
   present, produce the law × arm table. *(small, runs on the laptop.)*

## Runpod steps (Andrew's actions marked ★)

1. ★ Load credit: $20 covers the trial with margin. ★ Create an API key and
   `export RUNPOD_API_KEY=…` in the shell this session runs from (OAuth alone
   does not unlock runpodctl). Then `runpodctl ssh add-key` once.
2. Create a network volume, 20 GB, in a data centre that has H100 SXM stock
   (today: AP-IN-1, AP-JP-1, CA-MTL-1, EU-FR-1, EU-NL-1, EUR-IS-3, EUR-NO-2,
   US-GA-2, US-MO-1, US-NE-1; all LOW, re-read at launch). The volume is
   locked to its DC, so every pod in this plan runs there.
3. **Test hour, same DC, volume mounted, 2 h terminate guard.** Today no DC
   has both the RTX 4090 and the H100, so the test GPU is either the H100
   itself ($3.49 for the hour, no device switch, the simplest) or a co-located
   cheaper card: A40 at $0.49 in CA-MTL-1, L40S at $1.09 in EU-NL-1. ★ Andrew
   picks. Checks: `vulkaninfo --summary` lists the GPU; the build of item 3;
   `cargo test --release --features gpu` in phylu and fuller (record which
   float-parity tests differ on Vulkan); one law, one seed, ms/generation
   against the laptop's 15.5. Read the pod's vCPU count: 8 fits plus sympy
   checkers is a CPU question, not a GPU one.
4. **The A/B, H100 SXM pod ($3.49/h), volume mounted, 2 h terminate guard.**
   First ten minutes: the knee, measured on the H100 itself (shear flow at 1,
   2, 4, 8 concurrent, about $0.60); a knee measured on another card does not
   transfer. Then the sweep driver at that concurrency (plan: 8), the web page
   through the pod's HTTP proxy URL, run to the end. Expected under one hour at
   the cap, less with early stops. Under 8-way contention a hard law gets
   fewer generations before the 90 s cap than on the laptop, so the two arms
   compare fairly with each other, but the found-count is not comparable to
   CASCADE2's.
5. Collect, verify, remove the pod. Keep the volume until the files are on the
   laptop and checked; then delete it.

## Settings, fixed for the trial

CASCADE2 configuration: 3 genes, 1000+1000 (intake + champion), immigration
interval 50, merge age 10,000, typed snap every 10; dev seed 7013; gate -19
with 1-R² ≤ 1e-6 as the early stop; 90 s clock as the cap only; wrappers
identity; `EVOLVE_DIVERSITY=1`; evaluation counters on. Arms differ in one
variable: `EVOLVE_TYPED_DEPTH=2` or unset. Law order: `experiments/
order_prioritized.txt` (most likely first), then leave it alone.

## Outputs we get back

Per law, per arm: `run.log`, `card.json` (device, both commits, every switch),
`hof.tsv`, `stream.jsonl` (telemetry with diversity and evaluations by
source); per law: `result.json` (gate passed / stopped by / seconds /
generations / evaluations / SRBench verdict per arm) and `summary.txt`. Plus
the test hour's `vulkaninfo`, test logs and the knee table. Everything under
one `AB1/` folder on the laptop.

## Cost and time

| step | GPU | time | cost |
|---|---|---|---|
| test hour | 1 × H100 SXM (or a co-located A40 at $0.49) | 1 h | $3.49 |
| A/B, 266 fits at 90 s cap, 8 concurrent | 1 × H100 SXM | ≤ 50 min | ≤ $2.90 |
| volume 20 GB, a day | — | — | < $0.20 |
| **total** | | **≈ 2 h of attention** | **≈ $7 (≈ $4 with the A40 test)** |

Build effort before any rental: items 1–5 above, roughly one working day
across phylu and fuller.

## What would stop us

- **Every pod must be created with the env `NVIDIA_DRIVER_CAPABILITIES=all`.**
  The Vulkan ICD ships with the host driver and is mounted into the container
  only under the `graphics` capability; no stock Runpod template sets it, and
  without it `vulkaninfo` finds no device. This is the first thing the test
  hour checks.
- Vulkan still not visible with that set → the graphics capability is exposed by
  Runpod's container runtime, not by anything we install. Check Runpod's docs
  and templates for Vulkan support; if the runtime does not pass it, Runpod
  pods are out for us and the fallback is a provider that rents a full VM.
- Knee below 4 → scale out on cheaper pods (option B) instead; cost rises to
  about $5.
- Float-parity tests fail on Vulkan → record, run anyway (results are
  bit-identical to themselves on that device), and state the device.
- Checker port takes longer than a day → run the sweep without `--check`,
  collect models, run the SRBench verdict on the laptop afterwards.

## What we will know at the end

Law × arm: found / gate / none, generations, seconds, evaluations, final
diversity. The controlled TSR-vs-SR number. Whether the engine runs on Linux
Vulkan, the H100 knee, and whether the job-and-collect loop works end to end.
