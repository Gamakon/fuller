# Audit of docs/reports/2026-10-20_multityped_gep_rebuild.md

Audited 2026-10-20. Every figure in the report was compared with its source log or commit. SR totals recomputed from `phylu/logs/SR_OLD_NEW/run{1,2}/table.tsv` with `python3 -I`; regex rows compared with `phylu/logs/REGEX_TYPED_AB/table.txt`; the speed table recomputed from the `REGEX_TIMING` lines of the profile logs; WGSL from `fuller/logs/V24/run6c_*.log`; each hash resolved with `git cat-file` in fuller or phylu and its message read against the claim. Corrections are in fuller `4774cfc0`.

## Errors (7, all corrected in 4774cfc0; no conclusion changed)

| # | figure | report value | source value | status |
|---|---|---|---|---|
| 1 | 2.3 v2 sum of the four timers (`942fc267`) | 11.8 | 3.163 + 1.372 + 6.048 + 1.308 = 11.891 (`profile_n28_v2.log`), so 11.9. The phylu regex report carries the same 11.8. | WRONG, fixed |
| 2 | 2.4 halfroot initial population median nodes | median 15 to 20 | per-interface medians 18, 20, 15, 5, 17, 12, 16, 16, 21, 10, 10: 5 to 21 (`run6c_halfroot.log`, "typed done" lines) | WRONG, fixed |
| 3 | 2.4 start time | 06:29:30Z for both runs | halfroot 06:29:30Z, proproot 06:33:22Z (first line of each log) | WRONG, fixed |
| 4 | 3 caveat, new SR arm vs layout-free kernels | predates `7e798bf2` | `7e798bf2` is an ancestor of `093aae96` (the arm's commit, `sr_new_bin/COMMIT`); the arm predates `ed8521e9` (layout-free init, vary, decode) and `6cbb876d` | WRONG, fixed |
| 5 | 2.2 heading | "180 s" for all columns | only old, oldnosimp, newA28eq, newB28eq ran 180 s (`s` = 180.0 in `table.txt`); newA28 22.9 s, newB28 13.7 s, newA12 6.0 s, newB12 2.3 s stopped at a perfect device score | MISLEADING, fixed |
| 6 | 1 table, `5ba2ff53` under "write-back and cleanse" | write-back and cleanse | message: "the evaluator's node scratch is sized from the layout"; not write-back or cleanse | WRONG ROW, moved to the core row |
| 7 | 2.4 proproot "5 to 93 nodes" | 5 to 93 nodes | medians 46, 93, 60, 5, 85, 22, 36, 61, 89, 23, 15: 5 to 93 are medians, not node counts (range is 3 to 3540) | WORDING, fixed to "median 5 to 93" |

## Unsourced (not changed)

| figure | report value | finding |
|---|---|---|
| Caveat: "the load reached 100 at one point" | 100 | No log or phylu report records a load figure; only the 15 to 40 % host-time drift is sourced (recomputed from the profiles: evaluate and vary host time in C and D are 13 to 43 % above v2). Needs its source or removal by the author. |
| 2.3 B, C, D build commits | run on `552c6c96` with env switches | `regex_fit_prof*.COMMIT` records v1, v2 and A only; B, C, D are the same binary with `EVOLVE_FUNCTION_SHARE` and `EVOLVE_SYNC_STAGES` per the phylu report; the logs do not record the env. |
| Caveat: "I have not run a paired test" | n/a | No paired test in `logs/SR_OLD_NEW`; consistent. |

## 2.1 SR, old against new

| figure | report value | source value | status |
|---|---|---|---|
| run 1 seed-runs | 400 | 40 laws x 10 = 400 | OK |
| run 1 gate old / new / diff | 382 / 380 / -2 | 382 / 380 / -2 (tsv sum, table.txt TOTAL) | OK |
| run 2 seed-runs | 600 | 60 laws completed x 10 = 600 (sweep.log stopped at [61/93] old arm) | OK |
| run 2 gate old / new / diff | 99 / 101 / +2 | 99 / 101 / +2 | OK |
| total old / new | 481 / 481 / 0 | 382 + 99 = 481; 380 + 101 = 481 | OK |
| laws 10 of 10 in both arms | 34 | 32 (run 1) + 2 (run 2) = 34 | OK |
| killed runs | 0 / 0 | 0 / 0 in both tsv files | OK |
| 100 laws | 100 | 40 + 60 | OK |
| window | 04:54Z to about 06:25Z | START 04:54:56Z (run 1); last line 06:25:07Z (run 2) | OK |
| old arm | phylu `7d5e4e96`, fuller `946e5a59` | `sr_old_bin/COMMIT`: fuller 946e5a59 phylu 7d5e4e96 | OK |
| new arm | phylu `093aae96`, fuller `1e0c4667` | `sr_new_bin/COMMIT`: phylu 093aae96 fuller 1e0c4667 | OK |
| run 2 stopped at law 60 of 93, commit `552c6c96` | as stated | commit message: "stopped at 60 of 93 for the WGSL deadline" | OK |
| 7 differing laws, run 1 (old to new) | III_13_18 9-10, III_15_27 10-7, II_34_29b 8-9, II_4_23 10-7, II_8_7 9-10, I_6_2a 7-6, shearflow1 5-7 | identical in tsv, no others differ | OK |
| 20 differing laws, run 2 | III_19_51 2-4, barmag1 2-0, barmag2 2-1, I_10_7 0-1, III_15_12 0-1, III_17_37 1-2, glider1 1-2, I_12_4 9-10, I_43_43 1-0, III_15_14 8-10, II_10_9 4-2, II_13_17 7-6, II_6_11 6-7, I_32_5 8-7, glider2 3-7, I_26_2 0-2, I_13_12 1-0, I_34_1 2-1, I_30_5 3-0, I_27_6 1-0 | identical in tsv (20 laws differ, none missing) | OK |
| "fair coin head draw on both arms" | as stated | `c0bde1bd`, `1257b834`, `2a4c2b1e` are not ancestors of `093aae96`; the draw was the old fair coin | OK |

## 2.2 Regex, old against new (`table.txt`, `score_all.out`)

| figure | report value | source value | status |
|---|---|---|---|
| held-out set | 198 UK, 60 US ZIP, 30 ZIP+4, 60 Canadian, 50 corrupted UK | header of `table.txt` | OK |
| held-out recall: old, oldnosimp / all typed | 97.5 / 100.0 | 97.5 / 100.0 (means) | OK |
| foreign leaked of 600 | 0 in all 8 columns | 0 of 600 in all 8 means | OK |
| corrupted UK leaked of 200 | 104, 104, 10, 12, 10, 6, 0, 10 | 104, 104, 10 (A28), 12 (B28), 10 (A28eq), 6 (B28eq), 0 (A12), 10 (B12) | OK |
| train FN / FP | 14/18, 14/18, 0/1, 0.5/4, 0/1, 0.2/2, 0/0, 0/1 | same, from the mean lines | OK |
| rule length | 17, 17, 54, 70.8, 54, 68, 86.2, 65.2 | same | OK |
| generations | 24,641; 19,471; 1,007; 612; 6,964; 7,438; 398; 137 | same | OK |
| times in parentheses | 22.9, 13.7, 6.0, 2.3 s | same | OK |
| individuals, millions | 64.1, 50.6, 2.6, 1.6, 18.1, 19.3, 1.0, 0.4 | 64.07, 50.63, 2.62, 1.59, 18.11, 19.34, 1.04, 0.36 | OK |
| "3.5 times fewer individuals in the same 180 s" | 3.5 | 64.07 / 18.11 = 3.54 (A28eq); 64.07 / 19.34 = 3.31 (B28eq): 3.3 to 3.5 | OK (upper end) |
| "28 to 30 % of the old generation rate" | 28 to 30 | 6,964 / 24,641 = 28.3 %; 7,438 / 24,641 = 30.2 % | OK |
| rules "54 characters against 17" | 54, 17 | 54.0, 17.0 | OK |
| old arm trained on all 661 rows; typed arms held out 132 | 661, 132 | 461 + 200 = 661; holdout 92 + 40 = 132 (`profile_N12.log`, phylu report) | OK |
| old arm commit `dfcbb1e2` | as stated | `OLD_BUILD.txt`, commit message "print a CharClass from its computed bits" | OK |

## 2.3 Regex speed (`REGEX_TIMING` lines, sum = vary + evaluate + read-back + pump)

| figure | report value | source value | status |
|---|---|---|---|
| v1 `0e69b7d4`: generations, sum | 2,758; 22.4 | 2,758; 22.368 (`profile_N28.log`; COMMIT 0e69b7d4) | OK |
| v2 `942fc267`: generations, sum | 5,174; 11.8 | 5,174; 11.891 | WRONG, now 11.9 |
| A `552c6c96`: generations, sum | 3,962; 15.5 | 3,962; 15.470 (`profile_n28_v3.log`) | OK |
| B half: generations, sum | 4,161; 14.7 | 4,161; 14.733 (`profile_n28_v3_half.log`) | OK |
| C half, waits restored | 4,283; 14.3 | 4,283; 14.339 (`profile_n28_v3_half_sync.log`) | OK |
| D waits restored | 3,722; 16.5 | 3,722; 16.511 (`profile_n28_v3_sync.log`) | OK |
| old build 7.3 ms a generation | 7.3 | 180,000 / 24,641 = 7.305 | OK |
| "28 to 30 %" (repeat of 2.2) | 28 to 30 | as above | OK |
| N = 28, 60 s | stated | every `REGEX_TIMING` line reads "in 60.0 s"; N = 28 from file names | OK |

## 2.4 WGSL run 6c

| figure | report value | source value | status |
|---|---|---|---|
| seeds read, interfaces | 29 under 11 | 29 read under 11 interfaces (both logs) | OK |
| seeds not read | seed21, seed30, seed31 (3) | same three "typed seed unread" lines | OK |
| admitted per interface | 5, 4, 4, 2, 3, 1, 1, 6, 1, 1, 1 | same; sums to 29 | OK |
| interfaces with one admitted seed | 5 | interfaces 5, 6, 8, 9, 10 | OK |
| problem, generations | v24_ingest5b, 20 | header line, `gens 20` | OK |
| run duration | about 210 to 220 s | halfroot 209.2 s, proproot 219.9 s at generation 20 | OK |
| no single-node genes | both | `single-node 0` on all 22 "typed done" lines | OK |
| best fitness interface 0 | 0.33877 / 0.33969 | 0.33877 / 0.33969 | OK |
| other interfaces, halfroot | 0.58569 | all 10 at 0.58569 at generation 20 (interface 5 had no feasible row at generation 0 and 1, feasible by 20) | OK |
| proproot interface 8 | no feasible row (2.57080) | `best 2.57080 best feasible none`, feasible rows 0/8 | OK |
| result | runs end to end | both logs end with "typed done ... generations 20" | OK |
| `typed_ledger.tsv` | not cited in values | present in both run folders, 2,669 lines each | n/a |

## 1. Commits and counts

All 78 distinct hashes in the report resolve (fuller hashes in fuller, phylu hashes in phylu). Messages read against the claim:

| group | hashes | status |
|---|---|---|
| Design (fuller) | `eba4f52f` rev 4 design, `ca816f1e` bound table, `3004a9b1` layout, `ff7fccc8` chromosomes and ADF, `0b005f03` defect tests (100,000 planted), `034bf41f` WGSL terminals, `e04e9762` node limit, `7c22ae10` and `bde03d4c` implementation notes (both change the md and the html), `1ef6490d` rev-4 tests, `5e815c9b` DuckDB views tests, `8dad48ec` SHADER_F16 | OK |
| Host (fuller) | `62981dd2` geneframe fallbacks removed, `4c339113` loader fallback tables removed, `07356d9e`, `0398fabe`, `1d844b4d` (481 of 481), `b7d7577e` (481 of 481; 1,759 of 1,798; harvest 7,365 of 11,697), aggregates `1e0c4667`, `2c06047d`, `dfbcb53c`, `33931ab4` | OK |
| WGSL encode counts | 481 of 481; 1,759 of 1,798 | `docs/WGSL_TYPED_GENE.md` lines 82-83 and the `b7d7577e` message agree | OK |
| Typed GPU core (phylu) | `65a1cbc7` ... `2b7cc0e9` (11), `c77c9328` SR, `168b2aba`, `6fc6c717`, `20f65a23` goldens, `b844940e`, `039f20a1`, `96a7a91d`, `1241acf0`, `3fd78355`, `f78e5884` (format 3, per-row Scored), `9de90042`, `7e798bf2`, `ed8521e9`, `942fc267` (29 bindings, VP 22 words, uniform sizes), `c0bde1bd`, `1257b834`, `2a4c2b1e` | OK |
| `5ba2ff53` | listed under write-back and cleanse | WRONG ROW, moved |
| Kingdoms moved | regex 11 hashes (`81d0f041` ... `a076dcaa`), WGSL 8 hashes (`94575f3d` ... `67d4fe54`) | OK |
| Removed paths | `a2ecc962` legacy device decode, `f5fc359c` legacy host variation and init, `6cbb876d` untyped fallback and literal machinery, `2cdd40f6` old WGSL path | OK |
| Regex profile builds | v1 `0e69b7d4`, v2 `942fc267`, A `552c6c96` match `regex_fit_new/prof2/prof3.COMMIT`; `19859915` is an ancestor of `942fc267` | OK |
| Link targets | `docs/reviews/2026-10-19_multityped_gep_codex.md`, both phylu reports | exist | OK |

## Totals

Figures checked: about 265 (78 hashes, about 75 in 2.1, about 65 in 2.2, about 15 in 2.3, about 25 in 2.4, about 8 in the conclusion and caveats; counted by table cell, so the total is approximate). Errors found: 7, all corrected (4774cfc0). Unsourced: 1 figure (load 100) and 1 provenance gap (B, C, D env settings), left for the author.

## Follow-up (team-lead)

- "Load reached 100": source is team-lead's `uptime` at 04:45 local (03:45Z): load averages 51.53, 71.71, 100.99 (1, 5, 15 minute), several cargo builds running. Not in any repo log. Report line now reads "the 15-minute load average reached 101 at about 03:45Z, with several builds running".
- B, C, D settings added to the speed section from `gpu_queue9.sh` (scratchpad): `EVOLVE_FUNCTION_SHARE`, `EVOLVE_SYNC_STAGES=1`, `EVOLVE_PROGRAM_SIZE=28`, `EVOLVE_SEED=7013`, `EVOLVE_SECONDS=60`, binary `regex_fit_prof3`. The script's per-run argument mapping (which name gets which share and sync) is as stated by team-lead; the script takes them as arguments and I did not see the call lines.
