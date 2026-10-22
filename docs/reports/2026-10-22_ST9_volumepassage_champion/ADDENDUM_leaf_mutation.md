# ST9 addendum: LEAF mutation at real speed (2026-10-22)

RESULTS_SUMMARY

## Why

The morning's ST9 note (`README.md`, this directory) measured three problems.

- **The champion is not faster.** Its HFF gain came from the timed pairs, which were measured under 16 tiled submits at 256 x 256. Its one plausible speed change (workgroup 8 x 8 became 2 x 16) helped only there: 0.978x tiled, 1.0016x as one submit.
- **A literal became a parameter.** `sin(rf2 * 1.0 + t)` became `sin(rf2 * params.speed + t)`. It passes because no fitness case varies the parameters, and `speed` is 1.0 at the host's defaults.
- **Mutation was destructive.** A LOW child's tree overlapped its first parent's only about 0.5. A Karva edit that changes a node's argument signature shifts every later read of every type frame, so one edit rewrites much of the expressed tree. The median pair overlap in the population was 0.21.

## What changed

### LEAF mutation (phylu 231da1f4, `src/evolve/typed_vary.rs`, `typed_vary.wgsl` kernel `typed_leaf_main`, `src/evolve/true_wgsl_fit.rs`)

- **Positions.** Only the parent's expressed nodes are edited, read by fuller's typed walk.
- **Weight.** Each position has weight `(1/s)^alpha`, where `s` is its Horton-Strahler order (a leaf is order 1). `--leaf-weight depth` gives `(depth+1)^alpha` instead. With alpha = 2, a leaf is 4 times as likely as an order-2 node. Edits are drawn without replacement (`--leaf-edits`, default 1).
- **Replacement.** A symbol of the same output type with the identical ordered argument-type signature. An expressed `?_T` takes a new `C_T` index. A symbol alone in its class is absent from the draw.
- **Legality.** The replacement occupies the same cell, has the same output type and reads the same argument types in the same order. No later read of any type frame moves, so the child's tree is the parent's tree with exactly the edited nodes relabelled. Every symbol drawn is in the cell's typed domain, the same domain LOW draws from, with a different distribution. So every child is a legal kernel by construction, with no filter and no repair. The host test checks this on 10,000 genes (exact edit: the child differs in exactly the edited nodes).
- **Schedule.** LEAF is a fourth strength of the ST8 schedule (`--mutation-leaf P` is its share of children). A LEAF child takes no other mutation; crossover still pairs it. The diversity dial lowers the exponent to `alpha * (1 - dial)` when diversity falls, which spreads edits towards the root.

### Real-speed timing at 64 x 64

At 64 x 64 the whole pass fits under the per-submit work cap, so every kernel runs as one submit, as the shipped shader runs. The log: `encoded original ... 1 submits a pass` at every case; shipped original 0.174 ms, encoded original 0.182 ms GPU median (t = 5). The timed pairs in HFF are therefore measured in the setting where a speed-up would be real.

### Population dump (`write_dump`, format also in `docs/PLAN_true_gep_wgsl.md` ST8b)

Every `--dump-every N` generations, `OUT/dump/gNNNNNN.bin`, little-endian u32 words:

- **Header:** magic `ST8BDMP1` (2 words), version 1, generation, rows, framed types `n_t`, buckets `B` (1024), fitness inputs `I` (29), record words, cell words `C`; then per framed type its type code, its width `W_T` (`h_T + t_T` cut to `--dump-type-width`, 256) and `h_T + t_T`.
- **Record per individual:** generation, row, HFF (f32), picture exact (1, 0 or unknown), GPU ms (f32), born generation, hall of fame flag, cells cut, expressed nodes; then three vectors:
  1. **expressed cells:** per type, the symbols of its expressed nodes in frame order (`C` words, padded);
  2. **subtree-hash bag:** every expressed node's subtree hash counted in bucket `hash % 1024`;
  3. **the 29 HFF inputs** (f32).

UltraDim reads these to map the population: HDBSCAN and k-means for clusters (species), UMAP for a picture of the population over generations, and novelty detection for individuals far from every cluster. The dump is the input for measuring diversity in genotype (vectors 1 and 2) and in behaviour (vector 3) without choosing a distance in advance.

## How it was run

`/private/tmp/claude-501/st9p3/run.sh` took the GPU lock and ran, from phylu, under an 18-minute guard:

    wgsl_true_image_fit_leaf --shader volumepassage --size 64 --pop 10 --generations 500 --seed 3 \
        --mutation-leaf 1.0 --leaf-alpha 2 --pump-every 50 --checkpoint-every 100 --chart-every 100 \
        --dump-every 1 --out /private/tmp/claude-501/st9p3/run

Started 08:37:20Z on fuller e36faaee, phylu dedab1ad plus the ST8b working tree (committed as phylu 231da1f4 at 08:37:35Z). Binary `/private/tmp/claude-501/st9p3/wgsl_true_image_fit_leaf`, log `/private/tmp/claude-501/st9p3/run.log`, output `/private/tmp/claude-501/st9p3/run/` (`series.tsv`, `ledger.tsv`, `chart.png`, `hall_of_fame/`, `dump/`). Every child was a LEAF child (share 1.0), one edit each. Fitness is HFF as in the morning run (29 inputs: 2 fixed cases, 5 timed pairs, 2 held-out cases), at 64 x 64. Elites, tournament, crossover and the pump were as in the morning run except population 10.

## Results

RESULTS_BODY

## Limits

- **Population 10, one seed (3), 500 generations.** The morning run had population 32 at 256 x 256 for 3,468 generations, so the two runs are not a paired comparison.
- **No A/B.** LEAF was not run against LOW at the same settings (size 64, population 10, seed 3). Any difference from the morning run mixes the operator with the size, the timing mode and the population.
- **Crossover was still on.** A child's change from its first parent includes crossover; the parent-overlap figure covers both.
- **Parameters still never vary across fitness cases**, so a literal replaced by an equal parameter still passes.
- **One GPU** (Apple M3 Max, Metal). The 64 x 64 pass takes about 0.18 ms, so the timed pairs are short and launch overhead is a larger share of each.

## Next steps

1. **A/B of operators, several seeds.** LEAF against LOW against the classic GEP operators already in the typed core: one-point, two-point and gene recombination, IS and RIS transposition, each alone and in Ferreira's mix. Same kingdom, size, population and generations; at least five seeds each; best and mean HFF against generation, parent overlap, picture-exact share of children, and distinct trees.
2. **Vary the parameters across fitness cases.** Add held-out cases with other values of `speed`, the colour fields and a negative time, so a literal replaced by a parameter no longer draws the same picture.
3. **NVIDIA/Vulkan timing target.** Repeat the timing on an NVIDIA GPU under Vulkan (the Runpod 4090 build), so a speed-up is measured on a second compiler and device.
4. **Map the dumps.** Run UltraDim over `run/dump/` (clusters, UMAP over generations, novelty) and report how many species the population holds and when they appear.
5. **Larger size in one submit.** Time at 256 and 512 with all tiles in one command buffer, so larger pictures are timed without the 3x wait cost.
