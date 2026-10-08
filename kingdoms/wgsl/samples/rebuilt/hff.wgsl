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

    if ((k % 3u) == 0u) {
        let _e13 = scores[(base + 2u)];
        mse = _e13;
    } else {
        if ((k % 3u) == 1u) {
            let _e25 = scores[(base + 3u)];
            mse = _e25;
        } else {
            let _e32 = scores[(base + 5u)];
            mse = _e32;
        }
    }
    if ((k % 3u) == 0u) {
        let _e44 = scores[(base + 6u)];
        mae = _e44;
    } else {
        if ((k % 3u) == 1u) {
            let _e56 = scores[(base + 7u)];
            mae = _e56;
        } else {
            let _e63 = scores[(base + 8u)];
            mae = _e63;
        }
    }
    if ((k / 3u) == 0u) {
        let _e73 = mse;
        let _e76 = caps[(k % 3u)];
        return min(_e73, _e76);
    }
    if ((k / 3u) == 1u) {
        let _e89 = caps[(k % 3u)];
        if (_e89 > 0f) {
            let _e96 = mse;
            let _e99 = caps[(k % 3u)];
            return min((_e96 / _e99), 1f);
        }
        return 1f;
    }
    let _e107 = mae;
    let _e111 = caps[(3u + (k % 3u))];
    return min(_e107, _e111);
}

fn scaled(base_1: u32, i_2: u32, ok_1: ptr<function, bool>) -> f32 {
    var x: f32 = 0f;

    let _e5 = columns[i_2];
    let _e6 = objective(base_1, _e5);
    if ((bitcast<u32>(_e6) & 2139095040u) == 2139095040u) {
        (*ok_1) = false;
        return 0f;
    }
    let _e19 = columns[i_2];
    let _e21 = col_max[_e19];
    if (_e21 > 0f) {
        let _e29 = columns[i_2];
        let _e31 = col_max[_e29];
        x = min((_e6 / _e31), 1f);
    }
    let _e38 = log_scaled[i_2];
    if (_e38 == 1u) {
        let _e42 = x;
        if (_e42 <= LOG_FLOOR) {
            x = 0f;
        } else {
            let _e50 = x;
            x = (1f + ((log(_e50) / 2.302585f) / NEG_LOG10_FLOOR));
        }
    }
    let _e56 = x;
    return _e56;
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

    let _e14 = hp.rows;
    if (gid.x >= _e14) {
        return;
    }
    loop {
        let _e18 = i;
        let _e20 = hp.n_columns;
        if (_e18 < _e20) {
        } else {
            break;
        }
        {
            let _e26 = i;
            let _e28 = columns[_e26];
            if ((_e28 % 3u) == 2u) {
                block2_live = true;
            }
        }
        continuing {
            let _e35 = i;
            i = (_e35 + 1u);
        }
    }
    loop {
        let _e39 = c;
        let _e41 = hp.candidates;
        if (_e39 < _e41) {
        } else {
            break;
        }
        {
            let _e50 = hp.candidates;
            let _e52 = c;
            let _e55 = hp.width;
            let _e58 = scores[(((gid.x * _e50) + _e52) * _e55)];
            if ((bitcast<u32>(_e58) & 2139095040u) == 2139095040u) {
                continue;
            }
            ok = true;
            energy = 0f;
            abs_sum = 0f;
            i_1 = 0u;
            loop {
                let _e72 = i_1;
                let _e74 = hp.n_columns;
                if (_e72 < _e74) {
                } else {
                    break;
                }
                {
                    let _e81 = hp.candidates;
                    let _e83 = c;
                    let _e86 = hp.width;
                    let _e89 = i_1;
                    let _e91 = scaled((((gid.x * _e81) + _e83) * _e86), _e89, (&ok));
                    let _e93 = ok;
                    if !(_e93) {
                        break;
                    }
                    let _e96 = energy;
                    energy = (_e96 + (_e91 * _e91));
                    let _e100 = abs_sum;
                    abs_sum = (_e100 + abs(_e91));
                }
                continuing {
                    let _e105 = i_1;
                    i_1 = (_e105 + 1u);
                }
            }
            let _e108 = ok;
            if !(_e108) {
                continue;
            }
            let _e113 = hp.redundancy;
            if (_e113 == 1u) {
                let _e123 = energy;
                let _e126 = hp.candidates;
                let _e128 = c;
                let _e131 = hp.width;
                let _e135 = scores[((((gid.x * _e126) + _e128) * _e131) + 9u)];
                let _e139 = hp.candidates;
                let _e141 = c;
                let _e144 = hp.width;
                let _e148 = scores[((((gid.x * _e139) + _e141) * _e144) + 9u)];
                energy = (_e123 + (clamp(_e135, 0f, 1f) * clamp(_e148, 0f, 1f)));
                let _e160 = abs_sum;
                let _e163 = hp.candidates;
                let _e165 = c;
                let _e168 = hp.width;
                let _e172 = scores[((((gid.x * _e163) + _e165) * _e168) + 9u)];
                abs_sum = (_e160 + abs(clamp(_e172, 0f, 1f)));
            }
            let _e179 = hp.tower;
            if (_e179 == 1u) {
                p = 0f;
                let _e190 = hp.candidates;
                let _e192 = c;
                let _e195 = tower[((gid.x * _e190) + _e192)];
                if (_e195 > 2u) {
                    let _e207 = hp.candidates;
                    let _e209 = c;
                    let _e212 = tower[((gid.x * _e207) + _e209)];
                    p = min((f32((_e212 - 2u)) / 4f), 1f);
                }
                let _e219 = energy;
                let _e220 = p;
                let _e221 = p;
                energy = (_e219 + (_e220 * _e221));
                let _e226 = abs_sum;
                let _e227 = p;
                abs_sum = (_e226 + abs(_e227));
            }
            fitness = 0f;
            let _e237 = energy;
            let _e239 = hp.n_columns;
            let _e241 = hp.redundancy;
            let _e243 = hp.tower;
            if (clamp((1f - min((_e237 / f32((_e239 + (_e241 + _e243)))), 1f)), -1f, 1f) <= 0.9999999f) {
                let _e257 = energy;
                let _e259 = hp.n_columns;
                let _e261 = hp.redundancy;
                let _e263 = hp.tower;
                fitness = acos(clamp((1f - min((_e257 / f32((_e259 + (_e261 + _e263)))), 1f)), -1f, 1f));
            }
            let _e274 = fitness;
            key = _e274;
            let _e278 = hp.balanced;
            if (_e278 == 1u) {
                let _e282 = energy;
                if (_e282 <= 0.0000000000000002220446f) {
                    key = 0f;
                } else {
                    let _e292 = abs_sum;
                    let _e293 = energy;
                    let _e297 = hp.n_columns;
                    let _e299 = hp.redundancy;
                    let _e301 = hp.tower;
                    key = acos(clamp(((_e292 / sqrt(_e293)) / sqrt(f32((_e297 + (_e299 + _e301))))), -1f, 1f));
                }
            }
            let _e311 = fitness;
            let _e312 = win_truenorth;
            if (_e311 < _e312) {
                let _e316 = key;
                win = _e316;
                let _e319 = fitness;
                win_truenorth = _e319;
                let _e322 = c;
                win_at = _e322;
                let _e329 = hp.candidates;
                let _e331 = c;
                let _e334 = hp.width;
                win_base = (((gid.x * _e329) + _e331) * _e334);
                let _e348 = hp.candidates;
                let _e350 = c;
                let _e353 = hp.width;
                let _e357 = scores[((((gid.x * _e348) + _e350) * _e353) + 2u)];
                let _e359 = caps[0];
                let _e362 = caps[0];
                win_r0_ = select(1000000000000000000000000000000f, (_e357 / _e359), (_e362 > 0f));
                let _e377 = hp.candidates;
                let _e379 = c;
                let _e382 = hp.width;
                let _e386 = scores[((((gid.x * _e377) + _e379) * _e382) + 3u)];
                let _e388 = caps[1];
                let _e391 = caps[1];
                win_r1_ = select(1000000000000000000000000000000f, (_e386 / _e388), (_e391 > 0f));
                let _e395 = block2_live;
                if !(_e395) {
                    win_r2_ = 0f;
                } else {
                    let _e411 = hp.candidates;
                    let _e413 = c;
                    let _e416 = hp.width;
                    let _e420 = scores[((((gid.x * _e411) + _e413) * _e416) + 5u)];
                    let _e422 = caps[2];
                    let _e425 = caps[2];
                    win_r2_ = select(1000000000000000000000000000000f, (_e420 / _e422), (_e425 > 0f));
                }
            }
        }
        continuing {
            let _e430 = c;
            c = (_e430 + 1u);
        }
    }
    let _e437 = win_truenorth;
    best_fitness[gid.x] = _e437;
    let _e443 = win;
    best_selection[gid.x] = _e443;
    let _e449 = win_at;
    best_candidate[gid.x] = _e449;
    let _e457 = win_r0_;
    best_omr2_[(gid.x * 3u)] = _e457;
    let _e467 = win_r1_;
    best_omr2_[((gid.x * 3u) + 1u)] = _e467;
    let _e477 = win_r2_;
    best_omr2_[((gid.x * 3u) + 2u)] = _e477;
    let _e484 = win_base;
    let _e486 = scores[_e484];
    best_a[gid.x] = _e486;
    let _e494 = win_base;
    let _e497 = scores[(_e494 + 1u)];
    best_b[gid.x] = _e497;
    let _e501 = hp.gate;
    if (_e501 == 1u) {
        let _e509 = fitness_1[gid.x];
        if ((bitcast<u32>(_e509) & 2147483647u) > 2139095040u) {
            let _e518 = win;
            fitness_1[gid.x] = _e518;
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

    if (gid_1.x >= 9u) {
        return;
    }
    loop {
        let _e8 = c_1;
        let _e10 = hp.rows;
        let _e12 = hp.candidates;
        if (_e8 < (_e10 * _e12)) {
        } else {
            break;
        }
        {
            let _e19 = c_1;
            let _e21 = hp.width;
            let _e24 = scores[(_e19 * _e21)];
            if ((bitcast<u32>(_e24) & 2139095040u) == 2139095040u) {
                continue;
            }
            let _e30 = c_1;
            let _e32 = hp.width;
            let _e36 = objective((_e30 * _e32), gid_1.x);
            let _e38 = mx;
            mx = max(_e38, _e36);
        }
        continuing {
            let _e42 = c_1;
            c_1 = (_e42 + 1u);
        }
    }
    let _e49 = mx;
    col_max[gid_1.x] = _e49;
    return;
}
