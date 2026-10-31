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


# s9 batch 8: the v24 streaming-ingestion interface (problems/v24_ingest_small, problems/v24_ingest5b, contract.md): one
# module of two entry points, `ingest` (the first, carrying the `// passes N` header) and `search`, each a normal gene,
# joined by the fixed homeotic linker of design section 6. The bindings are the contract's table, group 0: the keys and
# queries read-only f32 (a seed's `array<vec4<f32>>` is the vec4 view), the two index arenas (index_u32 atomic: a seed's
# plain `array<u32>` reads through atomicLoad and atomicStore), the candidate rows `cands` (64 cells a workgroup's row,
# safety.contract `rows cands 64`) and the uniform `cfg`. Relaxed determinism: the contract allows atomicExchange and
# atomicLoad on index_u32, whose only use is an order-dependent value.
V24_FIELDS = ["seed_slot", "stage", "arrived_begin", "arrived_end", "n_queries", "pass_index", "n_passes", "c_max"]
V24 = dict(
    name="v24",
    module=[("ingest", True), ("search", False)],
    fuel_budget=6,
    fields=V24_FIELDS,
    inputs=[dict(name="keys", e="f32", ty="f32.real", vec4_view=True), dict(name="queries", e="f32", ty="f32.real", vec4_view=True)],
    arenas=[dict(name="index_u32", e="u32", ty="u32.index", atomic=True), dict(name="index_f32", e="f32", ty="f32.real", vec4_view=True)],
    row_outputs=[dict(name="cands", e="u32", ty="u32.index", row=64)],
    outputs=[],
    primary=None,
    determinism="relaxed",
    # s9 batch 9: every workgroup array of a kernel is a region of the two pools (pool.py)
    wg_pool=True,
    register_scale=4,
    big_private_arrays=True,
    passes=(1, 16),
    default_grid=(64, 1, 1, 1),
    max_invocations=256,
    subgroup_sizes=(32,),
    has_subgroups=False,
    has_subgroup_barrier=False,
    has_f16=True,
    workgroup_bytes=32768,
    extra_constants={
        # the seeds' own literals (fuller src/wgsl/true_reader.rs literal_census), spelled as each family's C_T spells them
        "index": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25', '28', '30', '31', '32', '34', '36', '40', '48', '63', '64', '80', '90', '96', '97', '99', '100', '103', '127', '128', '129', '130', '131', '143', '144', '160', '192', '211', '241', '255', '256', '257', '264', '291', '311', '317', '320', '384', '389', '480', '498', '511', '512', '513', '519', '520', '543', '544', '547', '695', '701', '767', '768', '777', '887', '960', '961', '977', '1000', '1013', '1023', '1024', '1031', '1049', '1088', '1217', '1237', '1280', '1399', '1445', '1535', '1536', '1613', '1792', '1879', '2047', '2048', '2049', '2112', '3072', '3136', '4000', '4095', '4096', '4160', '4608', '7919', '8191', '8192', '8193', '8256', '12288', '12345', '16383', '16384', '20251', '20973', '20992', '24576', '24999', '25000', '25088', '25599', '25600', '31153', '32767', '32768', '34816', '40503', '65535', '65536', '65568', '65600', '77824', '97531', '98368', '114752', '131072', '131584', '262143', '262144', '270000', '270400', '524287', '524288', '524544', '525312', '655360', '786432', '1000000', '1000003', '1048575', '1048576', '2000000', '2097151', '2097152', '2654435', '3145728', '4194303', '4194304', '4196352', '4221952', '4456448', '4980736', '5242880', '5369127', '6400000', '6400256', '6500000', '6504096', '6508192', '6512288', '6520000', '6553600', '6553616', '6557712', '6560000', '6561808', '6587408', '6600000', '6662144', '6815744', '6816000', '6816256', '6841856', '7340032', '7341056', '7602176', '7627776', '7644160', '8000000', '8050000', '8050768', '8051536', '8388352', '8388607', '16777215', '16777216', '19088743', '33554431', '67108863', '134217727', '277803737', '374761393', '475691837', '625341585', '668265263', '695872825', '739982445', '747796405', '1592590361', '1663821227', '2135587861', '2146121005', '2147483647', '2147483648', '2221713035', '2246822507', '2246822519', '2290649224', '2654435761', '2654435769', '2769414579', '2891336453', '3237998085', '3266489909', '3266489917', '4294966272', '4294967295', '6553601'],
        "count": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25', '28', '30', '31', '32', '34', '36', '40', '48', '63', '64', '80', '90', '96', '97', '99', '100', '103', '127', '128', '129', '130', '131', '143', '144', '160', '192', '211', '241', '255', '256', '257', '264', '291', '311', '317', '320', '384', '389', '480', '498', '511', '512', '513', '519', '520', '543', '544', '547', '695', '701', '767', '768', '777', '887', '960', '961', '977', '1000', '1013', '1023', '1024', '1031', '1049', '1088', '1217', '1237', '1280', '1399', '1445', '1535', '1536', '1613', '1792', '1879', '2047', '2048', '2049', '2112', '3072', '3136', '4000', '4095', '4096', '4160', '4608', '7919', '8191', '8192', '8193', '8256', '12288', '12345', '16383', '16384', '20251', '20973', '20992', '24576', '24999', '25000', '25088', '25599', '25600', '31153', '32767', '32768', '34816', '40503', '65535', '65536', '65568', '65600', '77824', '97531', '98368', '114752', '131072', '131584', '262143', '262144', '270000', '270400', '524287', '524288', '524544', '525312', '655360', '786432', '1000000', '1000003', '1048575', '1048576', '2000000', '2097151', '2097152', '2654435', '3145728', '4194303', '4194304', '4196352', '4221952', '4456448', '4980736', '5242880', '5369127', '6400000', '6400256', '6500000', '6504096', '6508192', '6512288', '6520000', '6553600', '6553616', '6557712', '6560000', '6561808', '6587408', '6600000', '6662144', '6815744', '6816000', '6816256', '6841856', '7340032', '7341056', '7602176', '7627776', '7644160', '8000000', '8050000', '8050768', '8051536', '8388352', '8388607', '16777215', '16777216', '19088743', '33554431', '67108863', '134217727', '277803737', '374761393', '475691837', '625341585', '668265263', '695872825', '739982445', '747796405', '1592590361', '1663821227', '2135587861', '2146121005', '2147483647', '2147483648', '2221713035', '2246822507', '2246822519', '2290649224', '2654435761', '2654435769', '2769414579', '2891336453', '3237998085', '3266489909', '3266489917', '4294966272', '4294967295'],
        "bits": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25', '28', '30', '31', '32', '34', '36', '40', '48', '63', '64', '80', '90', '96', '97', '99', '100', '103', '127', '128', '129', '130', '131', '143', '144', '160', '192', '211', '241', '255', '256', '257', '264', '291', '311', '317', '320', '384', '389', '480', '498', '511', '512', '513', '519', '520', '543', '544', '547', '695', '701', '767', '768', '777', '887', '960', '961', '977', '1000', '1013', '1023', '1024', '1031', '1049', '1088', '1217', '1237', '1280', '1399', '1445', '1535', '1536', '1613', '1792', '1879', '2047', '2048', '2049', '2112', '3072', '3136', '4000', '4095', '4096', '4160', '4608', '7919', '8191', '8192', '8193', '8256', '12288', '12345', '16383', '16384', '20251', '20973', '20992', '24576', '24999', '25000', '25088', '25599', '25600', '31153', '32767', '32768', '34816', '40503', '65535', '65536', '65568', '65600', '77824', '97531', '98368', '114752', '131072', '131584', '262143', '262144', '270000', '270400', '524287', '524288', '524544', '525312', '655360', '786432', '1000000', '1000003', '1048575', '1048576', '2000000', '2097151', '2097152', '2654435', '3145728', '4194303', '4194304', '4196352', '4221952', '4456448', '4980736', '5242880', '5369127', '6400000', '6400256', '6500000', '6504096', '6508192', '6512288', '6520000', '6553600', '6553616', '6557712', '6560000', '6561808', '6587408', '6600000', '6662144', '6815744', '6816000', '6816256', '6841856', '7340032', '7341056', '7602176', '7627776', '7644160', '8000000', '8050000', '8050768', '8051536', '8388352', '8388607', '16777215', '16777216', '19088743', '33554431', '67108863', '134217727', '277803737', '374761393', '475691837', '625341585', '668265263', '695872825', '739982445', '747796405', '1592590361', '1663821227', '2135587861', '2146121005', '2147483647', '2147483648', '2221713035', '2246822507', '2246822519', '2290649224', '2654435761', '2654435769', '2769414579', '2891336453', '3237998085', '3266489909', '3266489917', '4294966272', '4294967295'],
        "lit4096": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25', '28', '30', '31', '32', '34', '36', '40', '48', '63', '64', '80', '90', '96', '97', '99', '100', '103', '127', '128', '129', '130', '131', '143', '144', '160', '192', '211', '241', '255', '256', '257', '264', '291', '311', '317', '320', '384', '389', '480', '498', '511', '512', '513', '519', '520', '543', '544', '547', '695', '701', '767', '768', '777', '887', '960', '961', '977', '1000', '1013', '1023', '1024', '1031', '1049', '1088', '1217', '1237', '1280', '1399', '1445', '1535', '1536', '1613', '1792', '1879', '2047', '2048', '2049', '2112', '3072', '3136', '4000', '4095', '4096'],
        "real": ['-3.4e38', '-3e38', '-2e38', '-1.5e38', '-1e38', '-2e30', '-1e30', '-1e29', '-2097150.0', '-1000000.0', '-262144.0', '-131070.0', '-65536.0', '-65535.0', '-32768.0', '-16384.0', '-16000.0', '-2048.0', '-512.0', '-400.0', '-127.0', '-49.0', '-15.0', '-8.0', '-7.5', '-7.0', '-3.5', '-3.0', '-2.5', '-2.0', '-1.0', '-0.9999', '-0.5', '-0.25', '-0.1', '-0.08', '-0.05', '-0.012', '-0.0078125', '-3.0517578e-5', '-6.25e-6', '-1e-6', '-1e-20', '-1e-30', '0.0', '-0.0', '1e-30', '1e-20', '1e-6', '6.25e-6', '3.0517578e-5', '0.0078125', '0.012', '0.05', '0.08', '0.1', '0.25', '0.5', '0.9999', '1.0', '2.0', '2.5', '3.0', '3.5', '7.0', '7.5', '8.0', '15.0', '49.0', '127.0', '400.0', '512.0', '2048.0', '16000.0', '16384.0', '32768.0', '65535.0', '65536.0', '131070.0', '262144.0', '1000000.0', '2097150.0', '1e29', '1e30', '2e30', '1e38', '1.5e38', '2e38', '3e38', '3.4e38'],
        "int": ['-127i', '-8i', '0i', '7i', '8i', '127i'],
    },
    prologue_top="""enable f16;
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
""",
)

# The same interface at K = 1, for the random-gene loop (the work guard bounds every random gene's submit).
V24_K1 = dict(V24, name="v24_k1", fuel_budget=1)

# UltraDim's dense k-means centroid update (qdrant lib/projection-voting-kmeans/src/gpu_kmeans.rs, inventory A7:
# `update_centroids`, its driver update_centroids_gpu; fuller problems/kmeans_update). A module of ONE entry point,
# `update_centroids`, so every kernel of the table is a drop-in for A7's pipeline: the bindings are A7's own text (group 0:
# vectors, assignments, centroids read_write, the uniform `params: vec4<u32>` whose lanes x, y, z, w are n_vectors,
# n_centroids, dimensions, iteration), and no f16 (UltraDim's device manager requests no feature, so `enable f16;` would
# not compile there). The output is the k x d centroid matrix read as one row of k * d cells (A7's invocation owns the cell
# global_id.x = c * d + j): a module has no epilogue, so a cell may be stored conditionally (A7 stores only when its
# cluster has a point; an empty cluster keeps the old centroid the buffer holds). K = 4: A7 has one counted loop (over the
# points), whose trip count n reaches 2^20 = stratum 4 (each stratum holds 2^6 times the trips of the one below). The vec4 view of `vectors` lets a kernel read four lanes at once.
KMEANS_UPDATE = dict(
    name="kmeans_update",
    module=[("update_centroids", False)],
    fuel_budget=4,
    uniform="params",
    fields=["x", "y", "z", "w"],
    # a loop's trip count is bounded by n, k and d only (loops.py `size_bounds`)
    size_bounds=["x", "y", "z"],
    # the items each entry dispatches over at the problem's largest sizes (problems/kmeans_update*/problem.toml: n up to
    # 2^20, k up to 1,024, k * d up to 2^20), so a workgroup width keeps every dispatch within wgpu's 65,535 workgroups
    # (skeleton.py `min_wg_x`): A7's k * d cells
    dispatch_items={"update_centroids": 1 << 20},
    max_workgroups_x=65535,
    inputs=[dict(name="vectors", e="f32", ty="f32.real", vec4_view=True), dict(name="assignments", e="u32", ty="u32.index")],
    outputs=[dict(name="centroids", e="f32", ty="f32.real", rows="1u", cols="(params.y * params.z)")],
    primary=None,
    default_grid=(256, 1, 1, 1),
    max_invocations=256,
    subgroup_sizes=(32,),
    has_subgroups=False,
    has_subgroup_barrier=False,
    has_f16=False,
    workgroup_bytes=32768,
    extra_constants={"index": ["0", "1"], "count": ["0", "1"], "real": ["0.0"]},
    prologue_top="""@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
""",
)

# The same centroid update in the SCATTER FORM (fuller problems/kmeans_update_segmented): a module of five entries in
# dispatch order, A6's stable counting sort by label (hist, offsets, starts, scatter) and the segmented Kahan sum (mean),
# A7's bindings plus binding 4, the module's u32 scratch, an arena (the counting sort's counts, prefixes, lengths, starts
# and grouped ids live in it; a store at any cell is the arena policy's).
KMEANS_UPDATE_SEGMENTED = dict(
    KMEANS_UPDATE,
    name="kmeans_update_segmented",
    module=[("hist", False), ("offsets", False), ("starts", False), ("scatter", False), ("mean", False)],
    # hist: ceil(n / 256) * k blocks x clusters; offsets: k; starts: one workgroup (fixed); scatter: n; mean: k * d
    dispatch_items={"hist": (1 << 20) // 256 * 1024, "offsets": 1024, "scatter": 1 << 20, "mean": 1 << 20},
    arenas=[dict(name="scratch", e="u32", ty="u32.index")],
    extra_constants={"index": ["0", "1", "2", "255", "256"], "count": ["0", "1", "2", "255", "256"], "real": ["0.0"], "lit4096": ["255", "256"]},
    prologue_top=KMEANS_UPDATE["prologue_top"] + "@group(0) @binding(4) var<storage, read_write> scratch: array<u32>;\n",
)

INTERFACES = {"img_nebula": NEBULA, "gram": GRAM, "img_volumepassage": VOLUMEPASSAGE, "img_volumepassage_k1": VOLUMEPASSAGE_K1, "v24": V24, "v24_k1": V24_K1, "kmeans_update": KMEANS_UPDATE, "kmeans_update_segmented": KMEANS_UPDATE_SEGMENTED}
