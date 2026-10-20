# Multi-typed GEP rebuild: summary report, 2026-10-20

**Conclusion.** The multi-typed GEP rebuild moved SR, regex and the WGSL kingdom onto one typed core with nothing substituted: SR passed the gate in 481 of 1,000 seed-runs on both the old and the new engine over 100 laws; on the UK-postcode column the typed regex rule had 100.0 % held-out recall against 97.5 %, while evaluating 3.5 times fewer individuals in the same 180 s (28 to 30 % of the old generation rate at equal budget); and the WGSL kingdom runs end to end on 11 interfaces, but seed admission is still being fixed, so run 6d is pending and no WGSL score is claimed.

## Context

Objective: one general multi-typed GEP core (a head and a tail per output type, per-type Dc, shared constant pools) that every kingdom uses, with no repair, rejection, substitution or fallback. A misfit is reported as "our software has a bug: ...". Design: `nucleotable/docs/MultiTypedGEP.md` (rev 4, with the implementation notes from phylu), adversarial review in `docs/reviews/2026-10-19_multityped_gep_codex.md`. Kingdom reports from phylu: [SR on the typed core](../../../phylu/docs/reports/2026-10-20_sr_typed_gep.md) (40 laws; superseded by the 100-law totals below) and [regex on the typed core](../../../phylu/docs/reports/2026-10-20_regex_typed_gep.md).

## 1. What was built

### Design (fuller)

| item | commit |
|---|---|
| rev 4 design | `eba4f52f` |
| bound table | `ca816f1e` |
| layout | `3004a9b1` |
| chromosomes and ADF | `ff7fccc8` |
| defect tests | `0b005f03` |
| WGSL terminals | `034bf41f` |
| node limit | `e04e9762` |
| implementation notes (md, html) | `7c22ae10`, `bde03d4c` |

Also `1ef6490d`, `5e815c9b`, `8dad48ec`.

### Host side (fuller)

`GeneCodec` (`express` / `write_back`) is the only way a host rewrites a gene. Fallbacks removed: geneframe `62981dd2`, loader fallback tables `4c339113`. WGSL typed-gene encode: `07356d9e`, `0398fabe`, `1d844b4d`, `b7d7577e` (481 of 481 saved genes round trip; 1,759 of 1,798 corpus kernels read and round trip). Aggregates (matrices of f16 as duals): `1e0c4667`, `2c06047d`, `dfbcb53c`, `33931ab4`.

### Typed GPU core (phylu)

| part | commits |
|---|---|
| init, vary, decode, walk, select | `65a1cbc7`, `f279b306`, `34bae3a2`, `809949ba`, `bb11e3e2`, `8e994549`, `036dd9e4`, `fb6c91fd`, `038ec20e`, `f752a9af`, `2b7cc0e9` |
| SR on the typed core | `c77c9328` |
| SR golden (keyed by share and root_function) | `168b2aba`, `6fc6c717`, `20f65a23` |
| write-back and cleanse | `5ba2ff53`, `b844940e`, `039f20a1` |
| snap, fold, beam through the codec | `96a7a91d`, `1241acf0`, `3fd78355` |
| resume fixes (checkpoint format 3 carries per-row scores) | `f78e5884`, `9de90042` |
| layout-free kernels (shader text independent of layout; scratch in chunked storage buffers) | `7e798bf2`, `ed8521e9` |
| host contract tests (at most 29 bindings per stage; uniform sizes match the WGSL structs) | `942fc267` |
| head draw (`FunctionShare`, `root_function`) | `c0bde1bd`, `1257b834`, `2a4c2b1e` |

### Kingdoms moved

| kingdom | commits |
|---|---|
| SR | `c77c9328` |
| regex | `81d0f041`, `ec5488f3`, `d9c8e31e`, `0d04b552`, `0e69b7d4`, `e7e6d07a`, `093aae96`, `19859915`, `4a30091c`, `3e0c5712`, `a076dcaa` |
| WGSL | `94575f3d`, `497c4533`, `e0a34fd7`, `7de95bf0`, `defef2a8`, `43eb72ab`, `f6bb9a67`, `67d4fe54` |

### Removed paths (phylu)

`a2ecc962`, `f5fc359c`, `6cbb876d`, and `2cdd40f6` (the old WGSL path).

## 2. Measurements

### 2.1 SR, old against new (100 laws, 10 seeds each, gate passed)

| measurement | expected | old | new | difference |
|---|---|---|---|---|
| run 1, laws 1 to 40 (400 seed-runs) | no loss | 382 | 380 | -2 |
| run 2, laws 41 to 100 (600 seed-runs) | no loss | 99 | 101 | +2 |
| total, of 1,000 | no loss | 481 | 481 | 0 |
| laws 10 of 10 in both arms | n/a | 34 | 34 | 0 |
| killed runs | 0 | 0 | 0 | 0 |

Arms: old is phylu `7d5e4e96` with fuller `946e5a59`; new is phylu `093aae96` with fuller `1e0c4667`. Window 04:54Z to about 06:25Z; run 2 was stopped at law 60 of 93 for the WGSL deadline (`552c6c96`). Laws that differ (old to new, of 10):

| run | law | old | new | run | law | old | new |
|---|---|---|---|---|---|---|---|
| 1 | III_13_18 | 9 | 10 | 2 | I_12_4 | 9 | 10 |
| 1 | III_15_27 | 10 | 7 | 2 | I_43_43 | 1 | 0 |
| 1 | II_34_29b | 8 | 9 | 2 | III_15_14 | 8 | 10 |
| 1 | II_4_23 | 10 | 7 | 2 | II_10_9 | 4 | 2 |
| 1 | II_8_7 | 9 | 10 | 2 | II_13_17 | 7 | 6 |
| 1 | I_6_2a | 7 | 6 | 2 | II_6_11 | 6 | 7 |
| 1 | shearflow1 | 5 | 7 | 2 | I_32_5 | 8 | 7 |
| 2 | III_19_51 | 2 | 4 | 2 | glider2 | 3 | 7 |
| 2 | barmag1 | 2 | 0 | 2 | I_26_2 | 0 | 2 |
| 2 | barmag2 | 2 | 1 | 2 | I_13_12 | 1 | 0 |
| 2 | I_10_7 | 0 | 1 | 2 | I_34_1 | 2 | 1 |
| 2 | III_15_12 | 0 | 1 | 2 | I_30_5 | 3 | 0 |
| 2 | III_17_37 | 1 | 2 | 2 | I_27_6 | 1 | 0 |
| 2 | glider1 | 1 | 2 | | | | |

### 2.2 Regex, old against new (UK postcodes, 180 s, seeds 7013 to 7016, means over four seeds)

Source: phylu `logs/REGEX_TYPED_AB/table.txt`; held-out set 198 unseen UK postcodes, 60 US ZIP, 30 ZIP+4, 60 Canadian, 50 corrupted UK.

| measurement | old | oldnosimp | newA28 | newB28 | newA28eq | newB28eq | newA12 | newB12 |
|---|---|---|---|---|---|---|---|---|
| held-out recall, % of 198 | 97.5 | 97.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| foreign formats leaked, of 600 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| corrupted UK leaked, of 200 | 104 | 104 | 10 | 12 | 10 | 6 | 0 | 10 |
| train FN / FP | 14 / 18 | 14 / 18 | 0 / 1 | 0.5 / 4 | 0 / 1 | 0.2 / 2 | 0 / 0 | 0 / 1 |
| rule length, characters | 17 | 17 | 54 | 70.8 | 54 | 68 | 86.2 | 65.2 |
| generations | 24,641 | 19,471 | 1,007 (22.9 s) | 612 (13.7 s) | 6,964 | 7,438 | 398 (6.0 s) | 137 (2.3 s) |
| individuals, millions | 64.1 | 50.6 | 2.6 | 1.6 | 18.1 | 19.3 | 1.0 | 0.4 |

The rules are longer (54 characters against 17). The throughput expectation in the plan was missed.

### 2.3 Regex speed (60 s, N = 28, milliseconds of the four timers summed per generation)

| build | generations | sum, ms |
|---|---|---|
| v1 `0e69b7d4` | 2,758 | 22.4 |
| v2 `942fc267` (after match kernel `19859915`) | 5,174 | 11.8 |
| A, `552c6c96` default draw | 3,962 | 15.5 |
| B, half | 4,161 | 14.7 |
| C, half, waits restored | 4,283 | 14.3 |
| D, waits restored | 3,722 | 16.5 |

The old build takes 7.3 ms a generation; the typed arm is at 28 to 30 % of the old generation rate at equal budget. Next targets: selection (the elite kernel is `a076dcaa`), cleanse and crossover, and matching each distinct program once.

### 2.4 WGSL run 6c, as it stands

Logs: `logs/V24/run6c_halfroot.log`, `logs/V24/run6c_proproot.log`; started 06:29:30Z, problem `v24_ingest5b`, 20 generations, about 210 to 220 s a run.

| measurement | halfroot | proproot |
|---|---|---|
| seeds read | 29 under 11 interfaces; 3 not read by the safe reader (seed21, seed30, seed31) | same |
| seeds admitted per interface | 5, 4, 4, 2, 3, 1, 1, 6, 1, 1, 1 | same |
| initial population | no single-node genes; median 15 to 20 nodes | no single-node genes; 5 to 93 nodes |
| best fitness, interface 0 | 0.33877 | 0.33969 |
| other interfaces | 0.58569 | interface 8 has no feasible row (2.57080) |
| result | runs end to end | runs end to end |

| run | state | result |
|---|---|---|
| 6c | done | table above; seed admission is a known bug |
| **6d** | **PENDING** | **not run; follows the seed-admission fix** |

## 3. Caveats

- Stray GPU runs: my own GPU tests ran around 03:00Z to 03:10Z during the regex A/B, together with mtgep-regex's device tests in the same window. Throughput in that window is lower than it would be alone.
- Machine-load drift: the load reached 100 at one point; some speed-table runs read 15 to 40 % higher host times than others.
- The old regex arm (phylu `dfcbb1e2`) trained on all 661 rows; the typed arms held out 132. Train FN and FP above are over all rows for both.
- SR is single-typed, so the 481 against 481 result tests non-regression only.
- The arms draw different random numbers; per-law differences in 2.1 are within what seeds alone would give, but I have not run a paired test.
- The SR runs used the fair coin head draw on both arms, and the new SR arm predates the layout-free kernels (`7e798bf2`).
- The 40-law SR report is superseded by 2.1.

## 4. Open

- Seed admission in the WGSL run (3 seeds not read; 5 interfaces with one admitted seed; interface 8 of proproot has no feasible row). Then run 6d.
- Removal of the fallback node in fuller `src/wgsl`.
- Aggregates beyond matrices of f16.
- Speed targets for regex: selection, cleanse and crossover, matching each distinct program once.
