# library_ep texts the safe reader does not read (2026-10-21)

11 of 441 `library_ep/texts` have no dump. Findings for the any-WGSL completeness task (T43). First error line from `wgsl_safe_convert`:

| text | first error |
|---|---|
| fit_A2_stopped_1536Z__g001_r10p1.fable | no definition in scope for identifier: `wg_dummy_zero_guard` |
| fit_A2_stopped_1536Z__g001_r18p1.sonnet | expected statement, found "" |
| fit_A2_stopped_1536Z__g002_r16p1.sonnet | expected statement, found "" |
| fit_A2_stopped_1536Z__g002_r18p1.sonnet | expected statement, found "" |
| fit_A2_stopped_1625Z__g002_r10x_union.sonnet | expected statement, found "" |
| fit_A2_stopped_1625Z__g002_r4p1.sonnet | expected statement, found "" |
| fit_A2_stopped_1702Z__g003_r2x_union.sonnet | expected statement, found "" |
| fit_A2_stopped_1729Z__g002_r5p1.codex | a store or load of h is not at this thread's cell: block 4 x 1 in the index, the grid says 4 x 8 |
| fit_repro2__g001_r13p1.sonnet | expected statement, found "" |
| fit_smoke6__g001_r2x_union.fable | expected global item (...) or the end of the file, found "}" |
| fit_smoke6__g001_r8x_union.fable | expected global item (...) or the end of the file, found "}" |
