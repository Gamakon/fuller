# futhark-wgpu-shim

The WebGPU C API that Futhark's `webgpu` host runtime calls, implemented over
`wgpu-rs` 30.0.1 — the H2 route of `docs/PLAN_futhark_wgpu_host.md`. Futhark's
generated C compiles with plain `cc` against the header this crate generates,
links against this crate's static library, and runs its kernels on the same
`wgpu`/Metal path as the rest of fuller.

It is a **bridge, not a permanent API** (the plan's §10, question 4): it
implements exactly the 34 functions the generated C calls and nothing else,
and fuller reaches it only through `src/futhark_wgpu_host.rs`. fuller never
links it: it is a separate crate with its own workspace root.

## Layout

| path | what |
|---|---|
| `src/` | the implementation: `types.rs` (the C types), `handles.rs`, `futures.rs` (the future/callback model, plan §4.5), `api.rs` (the entry points), `report.rs` (dispatch counts, GPU timestamps) |
| `build.rs`, `cbindgen.toml` | write `webgpu/webgpu.h` from `src/` on every build |
| `include/webgpu/webgpu.h` | the committed copy of the generated header; a test checks it is byte-identical to the generated one |
| `include/dawn/webgpu.h` | a forwarding include, for the runtime's `-DUSE_DAWN` path |
| `include/emscripten.h` | a native stand-in for the one Emscripten function the runtime calls |
| `surface/` | S1: the surface, taken from the generated artefact (`extract_surface.sh`) |
| `tools/port_futhark_c.py` | the WGSL port (below) |
| `tools/compare_futhark_values.py` | compares two Futhark binary outputs bit for bit |
| `build_futhark.sh` | the link step |
| `tests/s5/` | S5: the same programs through Futhark's C backend and through webgpu + this shim |
| `futhark/rts/c/` | the vendored Futhark C runtime (below) |
| `fetch_futhark_branch.sh` | re-fetches the pinned compiler |

## The pinned compiler

Futhark's webgpu backend is not in a release. The compiler is the CI artefact
of the PR branch (diku-dk/futhark PR "Add WebGPU backend"):

| | |
|---|---|
| CI run | `39795721627` (event `pull_request`, branch `webgpu`) |
| branch head | `14e9ccbf3a47c06ed585d15e6827d186dd74d8a0` |
| the binary's `commit-id` | `02baea337a83885f3f2bfdcd7ae27b4148e5b68a` (the PR's merge commit, which CI builds) |
| artefact | `futhark-nightly-macos-arm64` (`futhark-nightly-macos-arm64.tar.xz`, sha256 `a432f56eff683d1f464751c640d86a19b632e5ccf7a106d6852b594287460f82`) |
| binary | `bin/futhark`, `Futhark 0.29.0 (prerelease)`, GHC 9.14.1, sha256 `3d54dcbe3d79aecce49fb7cbcc64c14c8bf026a3dfba6507533b2294388ea4e6` |

`./fetch_futhark_branch.sh [dir]` downloads it with
`gh run download 39795721627 -R diku-dk/futhark -n futhark-nightly-macos-arm64`
and stops with an error if the binary's sha256 is not the one above. Everything here
defaults to `/tmp/branch_artifact/futhark-nightly-macos-arm64/bin/futhark`;
`FUTHARK_BRANCH` overrides it.

## The vendored runtime

`futhark/rts/c/` is `rts/c/` of the branch head `14e9ccb` (a depth-1 clone of
`https://github.com/diku-dk/futhark.git`, branch `webgpu`). The generated C
embeds this runtime, so the build does not need it; it is vendored as the
reviewable source of what the generated C contains, and it was checked:
`backends/webgpu.h` here is byte-identical to the copy embedded in the
generated `probe.c`.

## The wait paths

The runtime has two ways of waiting for an asynchronous WebGPU request,
chosen by the preprocessor (plan §4.5):

* **default** — the request is started with `WGPUCallbackMode_AllowSpontaneous`
  and the runtime spins on `emscripten_sleep(0)` until the callback has run.
  The shim runs the callback before the starting call returns, so the spin is
  never entered: every report shows `emscripten_sleep` called **0** times.
* **`-DUSE_DAWN`** (`build_futhark.sh --dawn`) — the request is started with
  `WGPUCallbackMode_WaitAnyOnly` and the runtime spins on
  `wgpuInstanceWaitAny(..., 0)`. The callback runs only inside
  `wgpuInstanceWaitAny`.

S5 runs every program on both paths.

## The WGSL port

Two constructs in the generated WGSL stop naga (`tools/port_futhark_c.py`
explains both, and prints every change it makes):

1. helpers taking `ptr<storage, ...>` arguments — WGSL under the
   `unrestricted_pointer_parameters` extension, which naga has not
   implemented. Each call with `&name` or `&name[i]` is **specialised**: a copy
   of the helper that names the variable (or the element) directly. The
   meaning is unchanged. Unspecialised helpers are deleted.
2. the f16 scalar↔vector `bitcast` helpers — deleted, as
   `../tools/port_futhark_wgsl.py` deletes them.

A call the port cannot specialise is listed as NOT PORTED, and naga then
rejects the module by name: nothing is hidden.

## Reports and GPU timestamps

`FUTHARK_WGPU_SHIM_REPORT=<file>` makes the shim write, at exit, a JSON
record: the adapter, the dispatches, the workgroups, the submits, the futures
by mode, the `wgpuInstanceWaitAny` and `emscripten_sleep` calls.
`FUTHARK_WGPU_SHIM_TIMESTAMPS=1` also times every compute pass on the GPU
(the begin and end timestamps the GPU writes when the pass executes) — the
evidence that a kernel was dispatched, not merely built. Timing waits for each
submission, so it is a diagnostic mode, not the mode results are taken in.

The report also carries the **stage clocks** (UNIX nanoseconds): the
instance's creation, the device's readiness, the time spent creating shader
modules and compute pipelines (where Metal compiles them), and the exit. With
the caller's spawn time they split a process into its start, its device
set-up, its pipelines and its runs (`fuller::futhark_wgpu_host::RunStages`).

**The pass query** — `futhark_wgpu_shim_timed_passes()`,
`futhark_wgpu_shim_pass_ns(i)` (-1 for a pass the GPU did not really time),
`futhark_wgpu_shim_dispatches()` — is the shim's own, not WebGPU's: a host
that runs many entry points in one process (fuller's batch driver,
`src/futhark_batch.rs`) reads which passes were one call's. It is left out of
`webgpu.h` (`cbindgen.toml`), which stays exactly the surface; the driver
declares it itself, and the link step's surface check (which looks at
`wgpu*`/`emscripten*` names) does not see it.

## What the runtime leaves behind

Futhark's runtime never releases a handle (the surface has no
`wgpu*Release`). GPU memory is freed by `wgpuBufferDestroy`, which it calls;
the small objects of each kernel launch — the encoder, the pass, the command
buffer, the bind group, and for a kernel with dynamic block sizes its
pipeline — stay until the process exits. For a short-lived executable that is
harmless; for a long-lived host it is an upstream issue to report.

## Building and testing

```bash
RUSTFLAGS="-D warnings" cargo build --release
RUSTFLAGS="-D warnings" cargo clippy --release --all-targets
RUSTFLAGS="-D warnings" cargo test --release     # touches the GPU: take the GPU lock
./build_futhark.sh prog.c prog                   # link a generated program
tests/s5/run.sh /tmp/s5                          # S5, both wait paths
```

After a deliberate change to the surface, refresh the committed header with
`FUTHARK_WGPU_SHIM_BLESS=1 cargo build --release`.
