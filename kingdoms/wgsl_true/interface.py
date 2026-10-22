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


# The tiled dispatch of the image interfaces (skeleton.py): the submit's origin and the whole grid, in workgroups, written
# by the host before each submit (fuller::wgsl::image, group 3 binding 0). `DISPATCH_TILED` is a pipeline-overridable
# constant the host sets when it creates the pipeline: true when a pass takes more than one submit, false for one submit
# (then the entry reads the builtins and never the tile uniform). It is the interface's, never a row's.
TILE_DECL = ("struct DispatchTile {\n    origin: vec4<u32>,\n    grid: vec4<u32>,\n};\n@group(3) @binding(0) var<uniform> dispatch_tile: DispatchTile;\n"
             "override DISPATCH_TILED: bool = true;\n")

TIME_STRUCT = "struct TimeUniform {\n    time: f32,\n    delta: f32,\n    frame: u32,\n    _padding: u32,\n};\n"

# ST9: the cuneus visual shaders (examples/shaders in ~/Dev/gamakon/cuneus). One picture into a write-only storage
# texture from uniforms. The bindings are the shader's own (group 0 time, group 1 the texture and the parameter block);
# the structs are the host's. K = 2: the shader has a counted loop inside a counted loop.
VOLUMEPASSAGE = dict(
    name="img_volumepassage",
    image=True,
    tiled_dispatch=True,
    fuel_budget=2,
    fields=[],
    inputs=[],
    outputs=[dict(name="output", kind="texture", format="rgba16float", ty="vec4<f32>.real", rows="textureDimensions(output).y", cols="textureDimensions(output).x")],
    primary="output",
    uniforms=[
        dict(name="time_data", group=0, binding=0, fields=[("time", "f32"), ("delta", "f32"), ("frame", "u32"), ("_padding", "u32")]),
        dict(name="params", group=1, binding=1, fields=[(n, "f32") for n in ("speed", "intensity", "color1_r", "color1_g", "color1_b", "color2_r", "color2_g", "color2_b", "color3_r", "color3_g", "color3_b", "gamma", "zoom", "_padding1", "_padding2", "_padding3")]),
    ],
    default_grid=(8, 8, 1, 1),
    max_invocations=64,
    subgroup_sizes=(1,),
    has_subgroups=False,
    has_subgroup_barrier=False,
    has_f16=True,
    workgroup_bytes=32768,
    # the shader's own literals, spelled as the shader spells them (append-only: C_T is only ever added to)
    extra_constants={"real": ["0.0", "1.0", "0.5", "2.0", "3.0", "4.0", "0.1", "0.3", "0.6", "0.8", "0.01", "0.005", "0.12", "1.1", "2.3", "2.7",
                              "0.0245786", "0.000090537", "0.983729", "0.4329510", "0.238081",
                              "0.59719", "0.07600", "0.02840", "0.35458", "0.90834", "0.13383", "0.04823", "0.01566", "0.83777",
                              "1.60475", "-0.10208", "-0.00327", "-0.53108", "1.10813", "-0.07276", "-0.07367", "-0.00605", "1.07602"]},
    prologue_top="enable f16;\n" + TIME_STRUCT + """@group(0) @binding(0) var<uniform> time_data: TimeUniform;
struct VolumeParams {
    speed: f32, intensity: f32,
    color1_r: f32, color1_g: f32, color1_b: f32,
    color2_r: f32, color2_g: f32, color2_b: f32,
    color3_r: f32, color3_g: f32, color3_b: f32,
    gamma: f32, zoom: f32, _padding1: f32, _padding2: f32, _padding3: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: VolumeParams;
""" + TILE_DECL,
)

# The same interface at K = 1, for the random-gene loop (examples/wgsl_true_loop.rs --image): at K = 2 the work guard's bound
# H * 2^(6K) * invocations of even a 16 x 16 dispatch is over its cap for every random gene, so no random K = 2 kernel may run.
VOLUMEPASSAGE_K1 = dict(VOLUMEPASSAGE, name="img_volumepassage_k1", fuel_budget=1)

_NEBULA_FIELDS = [("iterations", "i32"), ("formuparam", "f32"), ("volsteps", "i32"), ("stepsize", "f32"), ("zoom", "f32"), ("tile", "f32"), ("speed", "f32"),
                  ("brightness", "f32"), ("dust_intensity", "f32"), ("distfading", "f32"), ("color_variation", "f32"), ("n_boxes", "f32"), ("rotation", "i32"),
                  ("depth", "f32"), ("color_mode", "i32"), ("_padding1", "f32"), ("rotation_x", "f32"), ("rotation_y", "f32"), ("click_state", "i32"), ("scale", "f32"),
                  ("exposure", "f32"), ("gamma", "f32")] + [(f"_padding{i}", "f32") for i in range(4, 11)] + [("time_scale", "f32"), ("visual_mode", "i32"), ("_padding2", "f32"), ("_padding3", "f32")]

# nebula.wgsl: Kali's volumetric nebula; the host runs a stub stage (`volumetric_render`, a bare return) and then `main_image`,
# which draws the picture; the atomic buffer is zeroed by the host and only read.
NEBULA = dict(
    name="img_nebula",
    image=True,
    tiled_dispatch=True,
    fuel_budget=2,
    fields=[],
    inputs=[dict(name="atomic_buffer", kind="atomic_u32", ty="u32.count")],
    outputs=[dict(name="output", kind="texture", format="rgba16float", ty="vec4<f32>.real", rows="textureDimensions(output).y", cols="textureDimensions(output).x")],
    primary="output",
    uniforms=[dict(name="u_time", group=0, binding=0, fields=[("time", "f32"), ("delta", "f32"), ("frame", "u32"), ("_padding", "u32")]),
              dict(name="params", group=1, binding=1, fields=_NEBULA_FIELDS)],
    default_grid=(16, 16, 1, 1),
    max_invocations=256,
    subgroup_sizes=(1,),
    has_subgroups=False,
    has_subgroup_barrier=False,
    has_f16=True,
    workgroup_bytes=32768,
    extra_constants={"real": ["0.0", "1.0", "0.5", "2.0", "-2.0", "0.1", "0.2", "0.25", "0.3", "0.4", "0.6", "0.7", "0.8", "0.9", "0.001", "0.01", "0.03", "0.05",
                              "0.95", "1.05", "1.1", "1.2", "1.5", "4.0", "800.0", "600.0", "0.299", "0.587", "0.114", "2.51", "2.43", "0.59", "0.14"], "index": ["2"]},
    prologue_top="enable f16;\n" + TIME_STRUCT + "@group(0) @binding(0) var<uniform> u_time: TimeUniform;\nstruct NebulaParams {\n"
    + "".join(f"    {n}: {t},\n" for n, t in _NEBULA_FIELDS)
    + """};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: NebulaParams;
@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
""" + TILE_DECL,
)

INTERFACES = {"img_nebula": NEBULA, "gram": GRAM, "img_volumepassage": VOLUMEPASSAGE, "img_volumepassage_k1": VOLUMEPASSAGE_K1}
