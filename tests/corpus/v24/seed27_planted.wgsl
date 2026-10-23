// passes 1
struct Cfg {
    seed_slot: u32,
    stage: u32,
    arrived_begin: u32,
    arrived_end: u32,
    n_queries: u32,
    pass_index: u32,
    n_passes: u32,
    c_max: u32,
};

@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;

const FEATURES: u32 = 64u;
const D4: u32 = 512u;
const THREADS: u32 = 16u;
const KEEP: u32 = 4u;
const OUT: u32 = 32u;
const PAD: u32 = 0xffffffffu;
const NONE: f32 = -3.0e38;
const MAX_ROWS: u32 = 25000u;

var<workgroup> tail_score: array<f32, 64>;
var<workgroup> tail_row: array<u32, 64>;
var<workgroup> old_score: array<f32, 64>;
var<workgroup> old_row: array<u32, 64>;

@compute @workgroup_size(1)
fn ingest(@builtin(workgroup_id) wid: vec3<u32>) {
    let r = cfg.arrived_begin + wid.x;
    if (r < cfg.arrived_end && r < MAX_ROWS) {
        let local_row = r - cfg.arrived_begin;
        let key_base = local_row * D4;
        let dst = r * FEATURES;
        for (var j = 0u; j < FEATURES; j += 1u) {
            let v = keys[key_base + j * 8u];
            index_f32[dst + j] = v.x;
        }
    }
}

@compute @workgroup_size(16)
fn search(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let q = wid.x;
    let t = lid.x;
    var tb: array<f32, 4>;
    var tr: array<u32, 4>;
    var ob: array<f32, 4>;
    var orow: array<u32, 4>;

    for (var z = 0u; z < KEEP; z += 1u) {
        tb[z] = NONE;
        tr[z] = PAD;
        ob[z] = NONE;
        orow[z] = PAD;
    }

    if (q < cfg.n_queries) {
        var qn = 0.0;
        for (var j = 0u; j < FEATURES; j += 1u) {
            let qv = queries[q * D4 + j * 8u].x;
            qn = qn + qv * qv;
        }

        for (var base = 0u; base < cfg.arrived_end; base += THREADS) {
            let r = base + t;
            if (r < cfg.arrived_end && r < MAX_ROWS) {
                var dotv = 0.0;
                var kn = 0.0;
                let src = r * FEATURES;
                // PLANTED SLOWDOWN (UI3 planted-win control): the row's score is
                // computed four times over; every pass starts from zero, so the
                // values are the seed's own.
                for (var rep = 0u; rep < 4u; rep += 1u) {
                    dotv = 0.0;
                    kn = 0.0;
                    for (var j = 0u; j < FEATURES; j += 1u) {
                        let qv = queries[q * D4 + j * 8u].x;
                        let kv = index_f32[src + j];
                        dotv = dotv + qv * kv;
                        kn = kn + kv * kv;
                    }
                }
                var score = NONE;
                let den = sqrt(qn * kn);
                if (den > 0.0) {
                    score = dotv / den;
                }

                if (r >= cfg.arrived_begin) {
                    if (score > tb[KEEP - 1u]) {
                        var cs = score;
                        var cr = r;
                        for (var k = 0u; k < KEEP; k += 1u) {
                            if (cs > tb[k]) {
                                let ts = tb[k];
                                let rr = tr[k];
                                tb[k] = cs;
                                tr[k] = cr;
                                cs = ts;
                                cr = rr;
                            }
                        }
                    }
                } else {
                    if (score > ob[KEEP - 1u]) {
                        var cs = score;
                        var cr = r;
                        for (var k = 0u; k < KEEP; k += 1u) {
                            if (cs > ob[k]) {
                                let ts = ob[k];
                                let rr = orow[k];
                                ob[k] = cs;
                                orow[k] = cr;
                                cs = ts;
                                cr = rr;
                            }
                        }
                    }
                }
            }
        }
    }

    for (var k = 0u; k < KEEP; k += 1u) {
        let p = t * KEEP + k;
        tail_score[p] = tb[k];
        tail_row[p] = tr[k];
        old_score[p] = ob[k];
        old_row[p] = orow[k];
    }
    workgroupBarrier();

    if (q < cfg.n_queries && t == 0u) {
        var fb: array<f32, 32>;
        var fr: array<u32, 32>;

        for (var k = 0u; k < OUT; k += 1u) {
            fb[k] = NONE;
            fr[k] = PAD;
        }
        for (var i = 0u; i < 64u; i += 1u) {
            let s = tail_score[i];
            let r = tail_row[i];
            if (r != PAD && s > fb[OUT - 1u]) {
                var cs = s;
                var cr = r;
                for (var k = 0u; k < OUT; k += 1u) {
                    if (cs > fb[k]) {
                        let ts = fb[k];
                        let rr = fr[k];
                        fb[k] = cs;
                        fr[k] = cr;
                        cs = ts;
                        cr = rr;
                    }
                }
            }
        }
        for (var k = 0u; k < OUT; k += 1u) {
            cands[q * 64u + k] = fr[k];
        }

        for (var k = 0u; k < OUT; k += 1u) {
            fb[k] = NONE;
            fr[k] = PAD;
        }
        for (var i = 0u; i < 64u; i += 1u) {
            let s = old_score[i];
            let r = old_row[i];
            if (r != PAD && s > fb[OUT - 1u]) {
                var cs = s;
                var cr = r;
                for (var k = 0u; k < OUT; k += 1u) {
                    if (cs > fb[k]) {
                        let ts = fb[k];
                        let rr = fr[k];
                        fb[k] = cs;
                        fr[k] = cr;
                        cs = ts;
                        cr = rr;
                    }
                }
            }
        }
        for (var k = 0u; k < OUT; k += 1u) {
            cands[q * 64u + OUT + k] = fr[k];
        }
    }
}
