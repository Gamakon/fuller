struct HffParams {
    rows: u32,
    candidates: u32,
    width: u32,
    n_columns: u32,
    wrappers: u32,
    tower: u32,
    redundancy: u32,
    balanced: u32,
    n_extrap: u32,
    gate: u32,
}

const PI: f32 = 3.1415927f;
const LOG_FLOOR: f32 = 0.000000000001f;
const NEG_LOG10_FLOOR: f32 = 12f;

@group(0) @binding(0) 
var<uniform> hp: HffParams;
@group(0) @binding(1) 
var<storage> scores: array<f32>;
@group(0) @binding(2) 
var<storage> columns: array<u32>;
@group(0) @binding(3) 
var<storage> log_scaled: array<u32>;
@group(0) @binding(4) 
var<storage, read_write> col_max: array<f32>;
@group(0) @binding(5) 
var<storage> caps: array<f32>;
@group(0) @binding(6) 
var<storage> tower: array<u32>;
@group(0) @binding(7) 
var<storage, read_write> best_fitness: array<f32>;
@group(0) @binding(8) 
var<storage, read_write> best_candidate: array<u32>;
@group(0) @binding(9) 
var<storage, read_write> best_omr2_: array<f32>;
@group(0) @binding(10) 
var<storage, read_write> best_selection: array<f32>;
@group(0) @binding(11) 
var<storage, read_write> fitness_1: array<f32>;
@group(0) @binding(12) 
var<storage, read_write> best_a: array<f32>;
@group(0) @binding(13) 
var<storage, read_write> best_b: array<f32>;

fn objective(base: u32, k: u32) -> f32 {
    var mse: f32 = 0f;
    var mae: f32 = 0f;

    let metric = (k / 3u);
    let block = (k % 3u);
    if (block == 0u) {
        let _e16 = scores[(base + 2u)];
        mse = _e16;
    } else {
        if (block == 1u) {
            let _e25 = scores[(base + 3u)];
            mse = _e25;
        } else {
            let _e32 = scores[(base + 5u)];
            mse = _e32;
        }
    }
    if (block == 0u) {
        let _e41 = scores[(base + 6u)];
        mae = _e41;
    } else {
        if (block == 1u) {
            let _e50 = scores[(base + 7u)];
            mae = _e50;
        } else {
            let _e57 = scores[(base + 8u)];
            mae = _e57;
        }
    }
    let v = caps[block];
    let m = caps[(3u + block)];
    if (metric == 0u) {
        let _e69 = mse;
        return min(_e69, v);
    }
    if (metric == 1u) {
        if (v > 0f) {
            let _e77 = mse;
            return min((_e77 / v), 1f);
        }
        return 1f;
    }
    let _e82 = mae;
    return min(_e82, m);
}

fn scaled(base_1: u32, i_2: u32, ok_1: ptr<function, bool>) -> f32 {
    var x: f32 = 0f;

    let k_1 = columns[i_2];
    let _e6 = objective(base_1, k_1);
    if ((bitcast<u32>(_e6) & 2139095040u) == 2139095040u) {
        (*ok_1) = false;
        return 0f;
    }
    let mx_1 = col_max[k_1];
    if (mx_1 > 0f) {
        x = min((_e6 / mx_1), 1f);
    }
    let _e27 = log_scaled[i_2];
    if (_e27 == 1u) {
        let _e31 = x;
        if (_e31 <= LOG_FLOOR) {
            x = 0f;
        } else {
            let _e39 = x;
            x = (1f + ((log(_e39) / 2.302585f) / NEG_LOG10_FLOOR));
        }
    }
    let _e45 = x;
    return _e45;
}

@compute @workgroup_size(64, 1, 1) 
fn hff_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var win: f32 = PI;
    var win_truenorth: f32 = PI;
    var win_at: u32 = 0u;
    var win_r0_: f32 = 1f;
    var win_r1_: f32 = 1f;
    var win_r2_: f32 = 1f;
    var win_base: u32 = 0u;
    var block2_live: bool = false;
    var i: u32 = 0u;
    var c: u32 = 0u;
    var ok: bool;
    var energy: f32;
    var abs_sum: f32;
    var i_1: u32;
    var p: f32;
    var fitness: f32;
    var key: f32;

    let row = gid.x;
    let _e14 = hp.rows;
    if (row >= _e14) {
        return;
    }
    let _e18 = hp.redundancy;
    let _e20 = hp.tower;
    let extra = (_e18 + _e20);
    let _e24 = hp.n_columns;
    let m_1 = f32((_e24 + extra));
    loop {
        let _e29 = i;
        let _e31 = hp.n_columns;
        if (_e29 < _e31) {
        } else {
            break;
        }
        {
            let _e37 = i;
            let _e39 = columns[_e37];
            if ((_e39 % 3u) == 2u) {
                block2_live = true;
            }
        }
        continuing {
            let _e46 = i;
            i = (_e46 + 1u);
        }
    }
    loop {
        let _e50 = c;
        let _e52 = hp.candidates;
        if (_e50 < _e52) {
        } else {
            break;
        }
        {
            let _e57 = hp.candidates;
            let _e59 = c;
            let _e62 = hp.width;
            let base_2 = (((row * _e57) + _e59) * _e62);
            let _e66 = scores[base_2];
            let bits = bitcast<u32>(_e66);
            if ((bits & 2139095040u) == 2139095040u) {
                continue;
            }
            ok = true;
            energy = 0f;
            abs_sum = 0f;
            i_1 = 0u;
            loop {
                let _e81 = i_1;
                let _e83 = hp.n_columns;
                if (_e81 < _e83) {
                } else {
                    break;
                }
                {
                    let _e86 = i_1;
                    let _e88 = scaled(base_2, _e86, (&ok));
                    let _e90 = ok;
                    if !(_e90) {
                        break;
                    }
                    let _e93 = energy;
                    energy = (_e93 + (_e88 * _e88));
                    let _e97 = abs_sum;
                    abs_sum = (_e97 + abs(_e88));
                }
                continuing {
                    let _e102 = i_1;
                    i_1 = (_e102 + 1u);
                }
            }
            let _e105 = ok;
            if !(_e105) {
                continue;
            }
            let _e110 = hp.redundancy;
            if (_e110 == 1u) {
                let _e118 = scores[(base_2 + 9u)];
                let r = clamp(_e118, 0f, 1f);
                let _e121 = energy;
                energy = (_e121 + (r * r));
                let _e125 = abs_sum;
                abs_sum = (_e125 + abs(r));
            }
            let _e131 = hp.tower;
            if (_e131 == 1u) {
                let _e137 = hp.candidates;
                let _e139 = c;
                let t = tower[((row * _e137) + _e139)];
                p = 0f;
                if (t > 2u) {
                    p = min((f32((t - 2u)) / 4f), 1f);
                }
                let _e157 = energy;
                let _e158 = p;
                let _e159 = p;
                energy = (_e157 + (_e158 * _e159));
                let _e164 = abs_sum;
                let _e165 = p;
                abs_sum = (_e164 + abs(_e165));
            }
            let _e171 = energy;
            let cos_theta = clamp((1f - min((_e171 / m_1), 1f)), -1f, 1f);
            fitness = 0f;
            if (cos_theta <= 0.9999999f) {
                fitness = acos(cos_theta);
            }
            let _e184 = fitness;
            key = _e184;
            let _e188 = hp.balanced;
            if (_e188 == 1u) {
                let _e192 = energy;
                if (_e192 <= 0.0000000000000002220446f) {
                    key = 0f;
                } else {
                    let _e200 = abs_sum;
                    let _e201 = energy;
                    let cb = clamp(((_e200 / sqrt(_e201)) / sqrt(m_1)), -1f, 1f);
                    key = acos(cb);
                }
            }
            let _e211 = fitness;
            let _e212 = win_truenorth;
            if (_e211 < _e212) {
                let _e216 = key;
                win = _e216;
                let _e219 = fitness;
                win_truenorth = _e219;
                let _e222 = c;
                win_at = _e222;
                win_base = base_2;
                let v0_ = caps[0];
                let v1_ = caps[1];
                let v2_ = caps[2];
                let _e243 = scores[(base_2 + 2u)];
                win_r0_ = select(1000000000000000000000000000000f, (_e243 / v0_), (v0_ > 0f));
                let _e254 = scores[(base_2 + 3u)];
                win_r1_ = select(1000000000000000000000000000000f, (_e254 / v1_), (v1_ > 0f));
                let _e259 = block2_live;
                if !(_e259) {
                    win_r2_ = 0f;
                } else {
                    let _e270 = scores[(base_2 + 5u)];
                    win_r2_ = select(1000000000000000000000000000000f, (_e270 / v2_), (v2_ > 0f));
                }
            }
        }
        continuing {
            let _e276 = c;
            c = (_e276 + 1u);
        }
    }
    let _e281 = win_truenorth;
    best_fitness[row] = _e281;
    let _e285 = win;
    best_selection[row] = _e285;
    let _e289 = win_at;
    best_candidate[row] = _e289;
    let _e295 = win_r0_;
    best_omr2_[(row * 3u)] = _e295;
    let _e303 = win_r1_;
    best_omr2_[((row * 3u) + 1u)] = _e303;
    let _e311 = win_r2_;
    best_omr2_[((row * 3u) + 2u)] = _e311;
    let _e316 = win_base;
    let _e318 = scores[_e316];
    best_a[row] = _e318;
    let _e324 = win_base;
    let _e327 = scores[(_e324 + 1u)];
    best_b[row] = _e327;
    let _e331 = hp.gate;
    if (_e331 == 1u) {
        let f = fitness_1[row];
        if ((bitcast<u32>(f) & 2147483647u) > 2139095040u) {
            let _e344 = win;
            fitness_1[row] = _e344;
            return;
        } else {
            return;
        }
    } else {
        return;
    }
}

@compute @workgroup_size(64, 1, 1) 
fn colmax_main(@builtin(global_invocation_id) gid_1: vec3<u32>) {
    var mx: f32 = 0f;
    var c_1: u32 = 0u;

    let k_2 = gid_1.x;
    if (k_2 >= 9u) {
        return;
    }
    let _e8 = hp.rows;
    let _e10 = hp.candidates;
    let n = (_e8 * _e10);
    loop {
        let _e13 = c_1;
        if (_e13 < n) {
        } else {
            break;
        }
        {
            let _e17 = c_1;
            let _e19 = hp.width;
            let base_3 = (_e17 * _e19);
            let _e23 = scores[base_3];
            let bits_1 = bitcast<u32>(_e23);
            if ((bits_1 & 2139095040u) == 2139095040u) {
                continue;
            }
            let _e28 = objective(base_3, k_2);
            let _e30 = mx;
            mx = max(_e30, _e28);
        }
        continuing {
            let _e34 = c_1;
            c_1 = (_e34 + 1u);
        }
    }
    let _e39 = mx;
    col_max[k_2] = _e39;
    return;
}
