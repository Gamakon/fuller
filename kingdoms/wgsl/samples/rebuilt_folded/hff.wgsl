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
    let def_b = (block == 1u);
    let def_a = (block == 0u);
    if def_a {
        let _e18 = scores[(base + 2u)];
        mse = _e18;
    } else {
        if def_b {
            let _e25 = scores[(base + 3u)];
            mse = _e25;
        } else {
            let _e32 = scores[(base + 5u)];
            mse = _e32;
        }
    }
    if def_a {
        let _e39 = scores[(base + 6u)];
        mae = _e39;
    } else {
        if def_b {
            let _e46 = scores[(base + 7u)];
            mae = _e46;
        } else {
            let _e53 = scores[(base + 8u)];
            mae = _e53;
        }
    }
    let v = caps[block];
    let m = caps[(3u + block)];
    if (metric == 0u) {
        let _e65 = mse;
        return min(_e65, v);
    }
    if (metric == 1u) {
        if (v > 0f) {
            let _e73 = mse;
            return min((_e73 / v), 1f);
        }
        return 1f;
    }
    let _e78 = mae;
    return min(_e78, m);
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
    let def_b_1 = (row * 3u);
    let _e16 = hp.rows;
    if (row >= _e16) {
        return;
    }
    let _e20 = hp.redundancy;
    let _e22 = hp.tower;
    let extra = (_e20 + _e22);
    let _e26 = hp.n_columns;
    let m_1 = f32((_e26 + extra));
    loop {
        let _e31 = i;
        let _e33 = hp.n_columns;
        if (_e31 < _e33) {
        } else {
            break;
        }
        {
            let _e39 = i;
            let _e41 = columns[_e39];
            if ((_e41 % 3u) == 2u) {
                block2_live = true;
            }
        }
        continuing {
            let _e48 = i;
            i = (_e48 + 1u);
        }
    }
    loop {
        let _e53 = hp.candidates;
        let _e55 = c;
        let def_a_1 = ((row * _e53) + _e55);
        let _e59 = c;
        let _e61 = hp.candidates;
        if (_e59 < _e61) {
        } else {
            break;
        }
        {
            let _e65 = hp.width;
            let base_2 = (def_a_1 * _e65);
            let _e69 = scores[base_2];
            let bits = bitcast<u32>(_e69);
            if ((bits & 2139095040u) == 2139095040u) {
                continue;
            }
            ok = true;
            energy = 0f;
            abs_sum = 0f;
            i_1 = 0u;
            loop {
                let _e84 = i_1;
                let _e86 = hp.n_columns;
                if (_e84 < _e86) {
                } else {
                    break;
                }
                {
                    let _e89 = i_1;
                    let _e91 = scaled(base_2, _e89, (&ok));
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
                let _e121 = scores[(base_2 + 9u)];
                let r = clamp(_e121, 0f, 1f);
                let _e124 = energy;
                energy = (_e124 + (r * r));
                let _e128 = abs_sum;
                abs_sum = (_e128 + abs(r));
            }
            let _e134 = hp.tower;
            if (_e134 == 1u) {
                let t = tower[def_a_1];
                p = 0f;
                if (t > 2u) {
                    p = min((f32((t - 2u)) / 4f), 1f);
                }
                let _e153 = energy;
                let _e154 = p;
                let _e155 = p;
                energy = (_e153 + (_e154 * _e155));
                let _e160 = abs_sum;
                let _e161 = p;
                abs_sum = (_e160 + abs(_e161));
            }
            let _e167 = energy;
            let cos_theta = clamp((1f - min((_e167 / m_1), 1f)), -1f, 1f);
            fitness = 0f;
            if (cos_theta <= 0.9999999f) {
                fitness = acos(cos_theta);
            }
            let _e180 = fitness;
            key = _e180;
            let _e184 = hp.balanced;
            if (_e184 == 1u) {
                let _e188 = energy;
                if (_e188 <= 0.0000000000000002220446f) {
                    key = 0f;
                } else {
                    let _e196 = abs_sum;
                    let _e197 = energy;
                    let cb = clamp(((_e196 / sqrt(_e197)) / sqrt(m_1)), -1f, 1f);
                    key = acos(cb);
                }
            }
            let _e207 = fitness;
            let _e208 = win_truenorth;
            if (_e207 < _e208) {
                let _e212 = key;
                win = _e212;
                let _e215 = fitness;
                win_truenorth = _e215;
                let _e218 = c;
                win_at = _e218;
                win_base = base_2;
                let v0_ = caps[0];
                let v1_ = caps[1];
                let v2_ = caps[2];
                let _e239 = scores[(base_2 + 2u)];
                win_r0_ = select(1000000000000000000000000000000f, (_e239 / v0_), (v0_ > 0f));
                let _e250 = scores[(base_2 + 3u)];
                win_r1_ = select(1000000000000000000000000000000f, (_e250 / v1_), (v1_ > 0f));
                let _e255 = block2_live;
                if !(_e255) {
                    win_r2_ = 0f;
                } else {
                    let _e266 = scores[(base_2 + 5u)];
                    win_r2_ = select(1000000000000000000000000000000f, (_e266 / v2_), (v2_ > 0f));
                }
            }
        }
        continuing {
            let _e272 = c;
            c = (_e272 + 1u);
        }
    }
    let _e277 = win_truenorth;
    best_fitness[row] = _e277;
    let _e281 = win;
    best_selection[row] = _e281;
    let _e285 = win_at;
    best_candidate[row] = _e285;
    let _e289 = win_r0_;
    best_omr2_[def_b_1] = _e289;
    let _e295 = win_r1_;
    best_omr2_[(def_b_1 + 1u)] = _e295;
    let _e301 = win_r2_;
    best_omr2_[(def_b_1 + 2u)] = _e301;
    let _e306 = win_base;
    let _e308 = scores[_e306];
    best_a[row] = _e308;
    let _e314 = win_base;
    let _e317 = scores[(_e314 + 1u)];
    best_b[row] = _e317;
    let _e321 = hp.gate;
    if (_e321 == 1u) {
        let f = fitness_1[row];
        if ((bitcast<u32>(f) & 2147483647u) > 2139095040u) {
            let _e334 = win;
            fitness_1[row] = _e334;
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
