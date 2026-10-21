"""The interface instance: the Gram worked instance of the design (section 9
preamble): `phi: array<f32>` in, `h: array<f32>` out with shape
`cfg.r_rows x cfg.d`, `cfg { d, m, r_rows, r_off }`, contiguous partition,
strict determinism, no arenas, no `// passes`, not a `full` (symmetric) kernel.
"""

GRAM = dict(
    name="gram",
    fields=["d", "m", "r_rows", "r_off"],
    inputs=[dict(name="phi", e="f32", ty="f32.real")],
    outputs=[dict(name="h", e="f32", ty="f32.real", rows="cfg.r_rows", cols="cfg.d")],
    primary="h",
    # the reference kernel's grid (kingdoms/wgsl/gram/blocked.wgsl): own.grid_grid
    default_grid=(16, 16, 4, 4),
    # the target: the M3 Max oracle device (src/wgsl/target.rs recorded)
    # the shape table's largest workgroup: the work guard (fuller work_guard) at K = 1 allows H * 64 * invocations <= 1.7e8,
    # and a 16 x 16 unit-test dispatch of a 64-thread shape launches at most 1,024 invocations (H = 2,325: 1.5e8)
    max_invocations=64,
    subgroup_sizes=(32,),
    has_subgroups=True,
    has_subgroup_barrier=True,
    has_f16=True,
    workgroup_bytes=32768,
    prologue_top="""enable f16;
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
""",
)
