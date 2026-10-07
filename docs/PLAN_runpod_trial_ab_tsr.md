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
3. **Image.** `Dockerfile`: Ubuntu 24.04, Rust stable, `libvulkan1` +
   `vulkan-tools`, Python 3 + sympy + SRBench checker deps, both repos at pinned
   commits, `cargo build --release --features gpu` for `evolve_fit` and
   `kitchen_sink`, SRBench ground-truth datasets (133 `.tsv.gz`, ~tens of MB)
   baked in. Entry point: `law_runner.py` reading `PHYLU_*` env. Push to a
   registry (Docker Hub or GHCR). Build on the laptop with
   `--platform linux/amd64`. *(fuller side can write it; phylu owns the paths.)*
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
2. Create a network volume, 20 GB, in a data centre with H100 stock (read live
   at launch; today H100 SXM shows HIGH across many).
3. **Test hour, RTX 4090 pod ($0.74/h), same DC, volume mounted, 2 h
   terminate guard:** `vulkaninfo --summary` lists the GPU; `cargo test
   --release --features gpu` in phylu and fuller (record which float-parity
   tests differ on Vulkan); one law, one seed, ms/generation vs laptop 15.5;
   then shear flow at 1, 2, 4, 8 concurrent fits → the knee. Remove the pod.
4. **The A/B, H100 SXM pod ($3.49/h), volume mounted, 2 h terminate guard:**
   run the sweep driver at the knee's concurrency (plan: 8), watch the web page
   through the pod's HTTP proxy URL, let it run to the end. Expected under one
   hour at the cap, less with early stops.
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
| test hour | 1 × RTX 4090 | 1 h | $0.74 |
| A/B, 266 fits at 90 s cap, 8 concurrent | 1 × H100 SXM | ≤ 50 min | ≤ $2.90 |
| volume 20 GB, a day | — | — | < $0.20 |
| **total** | | **≈ 2 h of attention** | **≈ $4** |

Build effort before any rental: items 1–5 above, roughly one working day
across phylu and fuller.

## What would stop us

- Vulkan not visible in the container → fix the image (ICD, driver
  capabilities) before renting anything bigger.
- Knee below 4 → use 4090 pods (option B) instead; cost rises to ~$5.
- Float-parity tests fail on Vulkan → record, run anyway (results are
  bit-identical to themselves on that device), and state the device.
- Checker port takes longer than a day → run the sweep without `--check`,
  collect models, run the SRBench verdict on the laptop afterwards.

## What we will know at the end

Law × arm: found / gate / none, generations, seconds, evaluations, final
diversity. The controlled TSR-vs-SR number. Whether the engine runs on Linux
Vulkan, the H100 knee, and whether the job-and-collect loop works end to end.
