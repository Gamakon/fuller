# safe96

Saved genomes (Karva dumps) of the texts in logs/HGP/001/gen0, as the safe
reader reads them.

- 2026-10-17 20:17Z: re-encoded at fuller 64401ba (the reader reads a
  thread's row or column in an index as safe.row_as_index / col_as_index),
  with examples/wgsl_safe_convert built in a clean detached worktree at that
  commit: `wgsl_safe_convert gen0 --out safe96 --karva safe96/karva --head 32 --slots 96`.
  The dumps before it are in safe96/karva_2026-10-17_pre_64401ba.
- 2026-10-17 20:50Z: re-encoded at fuller 5a966f3 (a knob reference's
  payload token typed by u32.payload), the same way:
  `wgsl_safe_convert gen0 --out safe96 --karva safe96/karva --head 32 --slots 96`.
  The dumps before it are in safe96/karva_2026-10-17_pre_5a966f3.
