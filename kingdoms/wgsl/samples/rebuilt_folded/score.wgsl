struct Meta {
    n_chromosomes: u32,
    genes_per: u32,
    n_rows: u32,
    n_train: u32,
    n_val: u32,
    n_extrap: u32,
    y_mean_train: f32,
    zero: u32,
    identity: u32,
    gene_base: u32,
    row_base: u32,
    pad1_: u32,
}

const N_WRAPPERS: u32 = 3u;
const WIDTH: u32 = 10u;
const LOO_STRIDE: u32 = 4u;
const NAN_BITS: u32 = 2143289344u;
const CONSTANT_REL_TOL: f32 = 0.000002f;

@group(0) @binding(0) 
var<uniform> sp: Meta;
@group(0) @binding(1) 
var<storage> preds: array<f32>;
@group(0) @binding(2) 
var<storage> gene_ok: array<u32>;
@group(0) @binding(3) 
var<storage> chromosomes: array<u32>;
@group(0) @binding(4) 
var<storage> y_1: array<f32>;
@group(0) @binding(5) 
var<storage, read_write> scores: array<f32>;
@group(0) @binding(6) 
var<storage> combinations: array<u32>;

fn finite(v: f32) -> bool {
    return ((bitcast<u32>(v) & 2139095040u) != 2139095040u);
}

fn gene_index(c: u32, g_2: u32) -> u32 {
    let _e4 = sp.genes_per;
    let def_a = ((c * _e4) + g_2);
    let _e10 = sp.identity;
    if (_e10 == 1u) {
        return def_a;
    }
    let _e14 = chromosomes[def_a];
    return _e14;
}

fn keep(v_1: f32) -> f32 {
    let _e4 = sp.zero;
    return bitcast<f32>((bitcast<u32>(v_1) ^ _e4));
}

fn high(v_2: f32) -> f32 {
    let _e3 = keep((4097f * v_2));
    let _e6 = keep((_e3 - v_2));
    let _e8 = keep((_e3 - _e6));
    return _e8;
}

fn div(x: f32, y: f32) -> f32 {
    let _e3 = keep((x / y));
    let _e6 = keep((_e3 * y));
    let _e7 = high(_e3);
    let _e9 = keep((_e3 - _e7));
    let _e11 = high(y);
    let _e14 = keep((y - _e11));
    let _e16 = keep((_e9 * _e14));
    let _e18 = keep((_e7 * _e11));
    let _e20 = keep((_e6 - _e18));
    let _e22 = keep((_e9 * _e11));
    let _e24 = keep((_e20 - _e22));
    let _e26 = keep((_e7 * _e14));
    let _e28 = keep((_e24 - _e26));
    let _e30 = keep((_e16 - _e28));
    let _e33 = keep((x - _e6));
    let _e35 = keep((_e33 - _e30));
    let _e36 = finite(_e35);
    let _e37 = finite(_e3);
    if (!(_e36) || !(_e37)) {
        return _e3;
    }
    let _e43 = keep((_e35 / y));
    let _e45 = keep((_e3 + _e43));
    return _e45;
}

fn linked(c_1: u32, genes: u32, linker: u32, row_7: u32) -> f32 {
    var acc: f32;
    var g_3: u32 = 0u;

    let def_a_1 = (linker == 1u);
    acc = select(0f, 1f, def_a_1);
    loop {
        let _e10 = g_3;
        let _e12 = sp.genes_per;
        if (_e10 < _e12) {
        } else {
            break;
        }
        {
            let _e18 = g_3;
            if (((genes >> _e18) & 1u) == 0u) {
                continue;
            }
            let _e24 = g_3;
            let _e25 = gene_index(c_1, _e24);
            let _e30 = sp.n_rows;
            let v_6 = preds[((_e25 * _e30) + row_7)];
            if def_a_1 {
                let _e36 = acc;
                let _e38 = keep((_e36 * v_6));
                acc = _e38;
            } else {
                let _e41 = acc;
                let _e43 = keep((_e41 + v_6));
                acc = _e43;
            }
        }
        continuing {
            let _e47 = g_3;
            g_3 = (_e47 + 1u);
        }
    }
    if (linker == 0u) {
        let _e53 = acc;
        let _e57 = div(_e53, f32(countOneBits(genes)));
        acc = _e57;
    }
    let _e60 = acc;
    return _e60;
}

fn linked_without(c_2: u32, genes_1: u32, linker_1: u32, row_8: u32, skip: u32, held: f32) -> f32 {
    var acc_1: f32;
    var g_4: u32 = 0u;
    var v_3: f32;

    let def_a_2 = (linker_1 == 1u);
    acc_1 = select(0f, 1f, def_a_2);
    loop {
        let _e10 = g_4;
        let _e12 = sp.genes_per;
        if (_e10 < _e12) {
        } else {
            break;
        }
        {
            let _e18 = g_4;
            if (((genes_1 >> _e18) & 1u) == 0u) {
                continue;
            }
            let _e30 = sp.genes_per;
            let _e32 = g_4;
            let _e35 = chromosomes[((c_2 * _e30) + _e32)];
            let _e37 = sp.n_rows;
            let _e41 = preds[((_e35 * _e37) + row_8)];
            v_3 = _e41;
            let _e44 = g_4;
            if (_e44 == skip) {
                v_3 = held;
            }
            if def_a_2 {
                let _e50 = acc_1;
                let _e51 = v_3;
                let _e53 = keep((_e50 * _e51));
                acc_1 = _e53;
            } else {
                let _e57 = acc_1;
                let _e58 = v_3;
                let _e60 = keep((_e57 + _e58));
                acc_1 = _e60;
            }
        }
        continuing {
            let _e64 = g_4;
            g_4 = (_e64 + 1u);
        }
    }
    if (linker_1 == 0u) {
        let _e70 = acc_1;
        let _e74 = div(_e70, f32(countOneBits(genes_1)));
        acc_1 = _e74;
    }
    let _e77 = acc_1;
    return _e77;
}

fn root(x_1: f32) -> f32 {
    let _e2 = keep(sqrt(x_1));
    let _e3 = finite(_e2);
    if ((_e2 == 0f) || !(_e3)) {
        return _e2;
    }
    let _e9 = keep((_e2 * _e2));
    let _e10 = high(_e2);
    let _e12 = keep((_e2 - _e10));
    let _e14 = keep((_e12 * _e12));
    let _e16 = keep((_e10 * _e10));
    let _e18 = keep((_e9 - _e16));
    let _e20 = keep((_e12 * _e10));
    let _e22 = keep((_e18 - _e20));
    let _e24 = keep((_e10 * _e12));
    let _e26 = keep((_e22 - _e24));
    let _e28 = keep((_e14 - _e26));
    let _e31 = keep((x_1 - _e9));
    let _e33 = keep((_e31 - _e28));
    let _e34 = finite(_e33);
    if !(_e34) {
        return _e2;
    }
    let _e38 = keep((2f * _e2));
    let _e40 = keep((_e33 / _e38));
    let _e42 = keep((_e2 + _e40));
    return _e42;
}

fn wrapped(v_4: f32, w_16: u32) -> f32 {
    if (w_16 == 1u) {
        return log((abs(v_4) + 0.000000000001f));
    }
    if (w_16 == 2u) {
        let _e13 = root(abs(v_4));
        return _e13;
    }
    return v_4;
}

fn add(s: vec2<f32>, v_5: f32) -> vec2<f32> {
    var c_3: f32;

    let _e4 = keep((s.x + v_5));
    c_3 = s.y;
    if (abs(s.x) >= abs(v_5)) {
        let _e17 = keep((s.x - _e4));
        let _e20 = keep((_e17 + v_5));
        let _e22 = c_3;
        let _e24 = keep((_e22 + _e20));
        c_3 = _e24;
    } else {
        let _e28 = keep((v_5 - _e4));
        let _e32 = keep((_e28 + s.x));
        let _e34 = c_3;
        let _e36 = keep((_e34 + _e32));
        c_3 = _e36;
    }
    let _e39 = c_3;
    return vec2<f32>(_e4, _e39);
}

fn total(s_1: vec2<f32>) -> f32 {
    let _e4 = keep((s_1.x + s_1.y));
    return _e4;
}

@compute @workgroup_size(64, 1, 1) 
fn score_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i: u32 = 0u;
    var g: u32 = 0u;
    var sum: array<vec2<f32>, 3>;
    var ok: array<bool, 3>;
    var x0_: array<f32, 3>;
    var lo: array<f32, 3>;
    var hi: array<f32, 3>;
    var w: u32 = 0u;
    var row: u32 = 0u;
    var w_1: u32;
    var mx: array<f32, 3>;
    var w_2: u32 = 0u;
    var sxx: array<vec2<f32>, 3>;
    var sxy: array<vec2<f32>, 3>;
    var w_3: u32 = 0u;
    var row_1: u32 = 0u;
    var w_4: u32;
    var a: array<f32, 3>;
    var b: array<f32, 3>;
    var w_5: u32 = 0u;
    var sq: array<vec2<f32>, 9>;
    var ab: array<vec2<f32>, 9>;
    var worst: array<f32, 3>;
    var i_1: u32 = 0u;
    var w_6: u32 = 0u;
    var row_2: u32 = 0u;
    var split: u32;
    var w_7: u32;
    var w_8: u32 = 0u;
    var i_2: u32;
    var k: u32;
    var syy: vec2<f32> = vec2<f32>(0f, 0f);
    var row_3: u32 = 0u;
    var loss: array<f32, 24>;
    var n_loss: array<u32, 3>;
    var w_9: u32 = 0u;
    var g_1: u32 = 0u;
    var gsum: vec2<f32>;
    var gmin: f32;
    var ghi: f32;
    var n_read: f32;
    var row_4: u32;
    var m1_: array<vec2<f32>, 3>;
    var lo1_: array<f32, 3>;
    var hi1_: array<f32, 3>;
    var w_10: u32;
    var row_5: u32;
    var w_11: u32;
    var xx: array<vec2<f32>, 3>;
    var xy: array<vec2<f32>, 3>;
    var w_12: u32;
    var row_6: u32;
    var w_13: u32;
    var w_14: u32;
    var without: f32;
    var l: f32;
    var w_15: u32 = 0u;
    var i_3: u32;
    var j: u32;

    let t = gid.x;
    let n_combinations = arrayLength((&combinations));
    let _e24 = sp.n_chromosomes;
    if (t >= (_e24 * n_combinations)) {
        return;
    }
    let c_4 = (t / n_combinations);
    let combination = combinations[(t % n_combinations)];
    let genes_2 = (combination & 16777215u);
    let linker_2 = (combination >> 24u);
    let _e40 = sp.row_base;
    let out = (((((_e40 + c_4) * n_combinations) + (t % n_combinations)) * N_WRAPPERS) * WIDTH);
    let nan = bitcast<f32>(NAN_BITS);
    loop {
        let _e51 = i;
        if (_e51 < 30u) {
        } else {
            break;
        }
        {
            let _e55 = i;
            scores[(out + _e55)] = nan;
        }
        continuing {
            let _e60 = i;
            i = (_e60 + 1u);
        }
    }
    loop {
        let _e64 = g;
        let _e66 = sp.genes_per;
        if (_e64 < _e66) {
        } else {
            break;
        }
        {
            let _e69 = g;
            let _e70 = gene_index(c_4, _e69);
            let _e76 = g;
            let _e81 = sp.gene_base;
            let _e84 = gene_ok[(_e81 + _e70)];
            if ((((genes_2 >> _e76) & 1u) == 1u) && (_e84 == 0u)) {
                return;
            }
        }
        continuing {
            let _e89 = g;
            g = (_e89 + 1u);
        }
    }
    let nt = sp.n_train;
    let _e96 = sp.n_val;
    let v1_ = (nt + _e96);
    let _e99 = linked(c_4, genes_2, linker_2, 0u);
    loop {
        let _e102 = w;
        if (_e102 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e107 = w;
            sum[_e107] = vec2<f32>(0f, 0f);
            let _e113 = w;
            ok[_e113] = true;
            let _e116 = w;
            let _e117 = wrapped(_e99, _e116);
            let _e120 = w;
            x0_[_e120] = _e117;
            let _e125 = w;
            let _e127 = w;
            let _e129 = x0_[_e127];
            lo[_e125] = _e129;
            let _e133 = w;
            let _e135 = w;
            let _e137 = x0_[_e135];
            hi[_e133] = _e137;
        }
        continuing {
            let _e140 = w;
            w = (_e140 + 1u);
        }
    }
    loop {
        let _e144 = row;
        let _e146 = sp.n_rows;
        if (_e144 < _e146) {
        } else {
            break;
        }
        {
            let _e149 = row;
            let _e150 = linked(c_4, genes_2, linker_2, _e149);
            w_1 = 0u;
            loop {
                let _e155 = w_1;
                if (_e155 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e158 = w_1;
                    let _e159 = wrapped(_e150, _e158);
                    let _e160 = finite(_e150);
                    let _e161 = finite(_e159);
                    if (!(_e160) || !(_e161)) {
                        let _e168 = w_1;
                        ok[_e168] = false;
                    } else {
                        let _e171 = row;
                        if (_e171 < nt) {
                            let _e175 = w_1;
                            let _e177 = x0_[_e175];
                            let _e179 = keep((_e159 - _e177));
                            let _e182 = w_1;
                            let _e184 = sum[_e182];
                            let _e185 = add(_e184, _e179);
                            let _e188 = w_1;
                            sum[_e188] = _e185;
                            let _e192 = w_1;
                            let _e194 = w_1;
                            let _e196 = lo[_e194];
                            lo[_e192] = min(_e196, _e159);
                            let _e200 = w_1;
                            let _e202 = w_1;
                            let _e204 = hi[_e202];
                            hi[_e200] = max(_e204, _e159);
                        }
                    }
                }
                continuing {
                    let _e208 = w_1;
                    w_1 = (_e208 + 1u);
                }
            }
        }
        continuing {
            let _e212 = row;
            row = (_e212 + 1u);
        }
    }
    loop {
        let _e216 = w_2;
        if (_e216 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e220 = w_2;
            let _e222 = sum[_e220];
            let _e223 = total(_e222);
            let _e225 = div(_e223, f32(nt));
            let _e228 = w_2;
            let _e230 = x0_[_e228];
            let _e232 = keep((_e230 + _e225));
            let _e235 = w_2;
            mx[_e235] = _e232;
        }
        continuing {
            let _e239 = w_2;
            w_2 = (_e239 + 1u);
        }
    }
    loop {
        let _e243 = w_3;
        if (_e243 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e248 = w_3;
            sxx[_e248] = vec2<f32>(0f, 0f);
            let _e254 = w_3;
            sxy[_e254] = vec2<f32>(0f, 0f);
        }
        continuing {
            let _e259 = w_3;
            w_3 = (_e259 + 1u);
        }
    }
    loop {
        let _e262 = row_1;
        if (_e262 < nt) {
        } else {
            break;
        }
        {
            let _e265 = row_1;
            let _e266 = linked(c_4, genes_2, linker_2, _e265);
            let _e270 = row_1;
            let _e272 = y_1[_e270];
            let _e274 = sp.y_mean_train;
            let _e276 = keep((_e272 - _e274));
            w_4 = 0u;
            loop {
                let _e281 = w_4;
                if (_e281 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e285 = w_4;
                    let _e287 = ok[_e285];
                    if _e287 {
                        let _e289 = w_4;
                        let _e290 = wrapped(_e266, _e289);
                        let _e293 = w_4;
                        let _e295 = mx[_e293];
                        let _e297 = keep((_e290 - _e295));
                        let _e299 = keep((_e297 * _e297));
                        let _e302 = w_4;
                        let _e304 = sxx[_e302];
                        let _e305 = add(_e304, _e299);
                        let _e308 = w_4;
                        sxx[_e308] = _e305;
                        let _e311 = keep((_e297 * _e276));
                        let _e314 = w_4;
                        let _e316 = sxy[_e314];
                        let _e317 = add(_e316, _e311);
                        let _e320 = w_4;
                        sxy[_e320] = _e317;
                    }
                }
                continuing {
                    let _e324 = w_4;
                    w_4 = (_e324 + 1u);
                }
            }
        }
        continuing {
            let _e328 = row_1;
            row_1 = (_e328 + 1u);
        }
    }
    loop {
        let _e332 = w_5;
        if (_e332 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e336 = w_5;
            let _e338 = sxx[_e336];
            let _e339 = total(_e338);
            let _e349 = w_5;
            let _e351 = ok[_e349];
            let _e353 = w_5;
            let _e355 = hi[_e353];
            let _e356 = w_5;
            let _e358 = lo[_e356];
            let _e360 = w_5;
            let _e362 = mx[_e360];
            if ((!(_e351) || ((_e355 - _e358) <= (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e362)))))) || (_e339 <= 0f)) {
                let _e374 = w_5;
                ok[_e374] = false;
                continue;
            }
            let _e378 = w_5;
            let _e380 = sxy[_e378];
            let _e381 = total(_e380);
            let _e382 = div(_e381, _e339);
            let _e385 = w_5;
            a[_e385] = _e382;
            let _e390 = w_5;
            let _e392 = a[_e390];
            let _e393 = w_5;
            let _e395 = mx[_e393];
            let _e397 = keep((_e392 * _e395));
            let _e400 = sp.y_mean_train;
            let _e402 = keep((_e400 - _e397));
            let _e405 = w_5;
            b[_e405] = _e402;
            let _e409 = w_5;
            let _e411 = a[_e409];
            let _e412 = finite(_e411);
            let _e415 = w_5;
            let _e417 = b[_e415];
            let _e418 = finite(_e417);
            if (!(_e412) || !(_e418)) {
                let _e425 = w_5;
                ok[_e425] = false;
            }
        }
        continuing {
            let _e429 = w_5;
            w_5 = (_e429 + 1u);
        }
    }
    loop {
        let _e433 = i_1;
        if (_e433 < 9u) {
        } else {
            break;
        }
        {
            let _e438 = i_1;
            sq[_e438] = vec2<f32>(0f, 0f);
            let _e444 = i_1;
            ab[_e444] = vec2<f32>(0f, 0f);
        }
        continuing {
            let _e449 = i_1;
            i_1 = (_e449 + 1u);
        }
    }
    loop {
        let _e453 = w_6;
        if (_e453 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e458 = w_6;
            worst[_e458] = 0f;
        }
        continuing {
            let _e462 = w_6;
            w_6 = (_e462 + 1u);
        }
    }
    loop {
        let _e466 = row_2;
        let _e468 = sp.n_rows;
        if (_e466 < _e468) {
        } else {
            break;
        }
        {
            let _e471 = row_2;
            let _e472 = linked(c_4, genes_2, linker_2, _e471);
            split = 2u;
            let _e476 = row_2;
            if (_e476 < nt) {
                split = 0u;
            } else {
                let _e481 = row_2;
                if (_e481 < v1_) {
                    split = 1u;
                }
            }
            w_7 = 0u;
            loop {
                let _e490 = w_7;
                let _e492 = split;
                let def_d = ((_e490 * 3u) + _e492);
                let _e496 = w_7;
                if (_e496 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e500 = w_7;
                    let _e502 = ok[_e500];
                    if _e502 {
                        let _e504 = w_7;
                        let _e505 = wrapped(_e472, _e504);
                        let _e508 = w_7;
                        let _e510 = a[_e508];
                        let _e512 = keep((_e510 * _e505));
                        let _e515 = w_7;
                        let _e517 = b[_e515];
                        let _e519 = keep((_e512 + _e517));
                        let _e522 = row_2;
                        let _e524 = y_1[_e522];
                        let _e526 = keep((_e524 - _e519));
                        let _e528 = keep((_e526 * _e526));
                        let _e531 = sq[def_d];
                        let _e532 = add(_e531, _e528);
                        sq[def_d] = _e532;
                        let _e537 = ab[def_d];
                        let _e539 = add(_e537, abs(_e526));
                        ab[def_d] = _e539;
                        let _e544 = split;
                        if (_e544 == 1u) {
                            let _e548 = w_7;
                            let _e550 = w_7;
                            let _e552 = worst[_e550];
                            worst[_e548] = max(_e552, abs(_e526));
                        }
                    }
                }
                continuing {
                    let _e557 = w_7;
                    w_7 = (_e557 + 1u);
                }
            }
        }
        continuing {
            let _e561 = row_2;
            row_2 = (_e561 + 1u);
        }
    }
    loop {
        let _e565 = w_8;
        let def_e = (_e565 * 3u);
        let def_c = (def_e + 2u);
        let def_b = (def_e + 1u);
        let _e573 = w_8;
        if (_e573 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e577 = w_8;
            let _e579 = ok[_e577];
            if !(_e579) {
                continue;
            }
            let _e583 = w_8;
            let o = (out + (_e583 * WIDTH));
            let _e590 = sp.n_val;
            let _e593 = sp.n_extrap;
            let n = array<f32, 3>(f32(nt), f32(_e590), f32(max(_e593, 1u)));
            let _e601 = w_8;
            let _e603 = a[_e601];
            scores[o] = _e603;
            let _e610 = w_8;
            let _e612 = b[_e610];
            scores[(o + 1u)] = _e612;
            let _e615 = sq[def_e];
            let _e616 = total(_e615);
            let _e618 = div(_e616, n[0]);
            scores[(o + 2u)] = _e618;
            let _e625 = sq[def_b];
            let _e626 = total(_e625);
            let _e628 = div(_e626, n[1]);
            scores[(o + 3u)] = _e628;
            let _e639 = w_8;
            let _e641 = worst[_e639];
            scores[(o + 4u)] = _e641;
            let _e644 = sq[def_c];
            let _e645 = total(_e644);
            let _e647 = div(_e645, n[2]);
            scores[(o + 5u)] = _e647;
            let _e654 = ab[def_e];
            let _e655 = total(_e654);
            let _e657 = div(_e655, n[0]);
            scores[(o + 6u)] = _e657;
            let _e664 = ab[def_b];
            let _e665 = total(_e664);
            let _e667 = div(_e665, n[1]);
            scores[(o + 7u)] = _e667;
            let _e674 = ab[def_c];
            let _e675 = total(_e674);
            let _e677 = div(_e675, n[2]);
            scores[(o + 8u)] = _e677;
            scores[(o + 9u)] = 0f;
            i_2 = 0u;
            loop {
                let _e691 = i_2;
                if (_e691 < WIDTH) {
                } else {
                    break;
                }
                {
                    let _e695 = i_2;
                    let _e698 = scores[(o + _e695)];
                    let _e699 = finite(_e698);
                    if !(_e699) {
                        k = 0u;
                        loop {
                            let _e705 = k;
                            if (_e705 < WIDTH) {
                            } else {
                                break;
                            }
                            {
                                let _e709 = k;
                                scores[(o + _e709)] = nan;
                            }
                            continuing {
                                let _e714 = k;
                                k = (_e714 + 1u);
                            }
                        }
                        let _e719 = w_8;
                        ok[_e719] = false;
                        break;
                    }
                }
                continuing {
                    let _e723 = i_2;
                    i_2 = (_e723 + 1u);
                }
            }
        }
        continuing {
            let _e727 = w_8;
            w_8 = (_e727 + 1u);
        }
    }
    if (countOneBits(genes_2) == 1u) {
        return;
    }
    loop {
        let _e733 = row_3;
        if (_e733 < nt) {
        } else {
            break;
        }
        {
            let _e738 = row_3;
            let _e740 = y_1[_e738];
            let _e742 = sp.y_mean_train;
            let _e744 = keep((_e740 - _e742));
            let _e746 = keep((_e744 * _e744));
            let _e748 = syy;
            let _e749 = add(_e748, _e746);
            syy = _e749;
        }
        continuing {
            let _e753 = row_3;
            row_3 = (_e753 + LOO_STRIDE);
        }
    }
    let _e756 = syy;
    let _e757 = total(_e756);
    loop {
        let _e760 = w_9;
        if (_e760 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e765 = w_9;
            n_loss[_e765] = 0u;
        }
        continuing {
            let _e769 = w_9;
            w_9 = (_e769 + 1u);
        }
    }
    loop {
        let _e774 = g_1;
        let _e776 = sp.genes_per;
        if (_e774 < min(_e776, 8u)) {
        } else {
            break;
        }
        {
            let _e782 = g_1;
            if (((genes_2 >> _e782) & 1u) == 0u) {
                continue;
            }
            let _e787 = g_1;
            let _e788 = gene_index(c_4, _e787);
            let _e792 = sp.n_rows;
            let g0_ = preds[(_e788 * _e792)];
            gsum = vec2<f32>(0f, 0f);
            gmin = g0_;
            ghi = g0_;
            n_read = 0f;
            row_4 = 0u;
            loop {
                let _e806 = row_4;
                if (_e806 < nt) {
                } else {
                    break;
                }
                {
                    let _e812 = sp.n_rows;
                    let _e814 = row_4;
                    let v_7 = preds[((_e788 * _e812) + _e814)];
                    let _e819 = keep((v_7 - g0_));
                    let _e821 = gsum;
                    let _e822 = add(_e821, _e819);
                    gsum = _e822;
                    let _e825 = gmin;
                    gmin = min(_e825, v_7);
                    let _e828 = ghi;
                    ghi = max(_e828, v_7);
                    let _e832 = n_read;
                    n_read = (_e832 + 1f);
                }
                continuing {
                    let _e836 = row_4;
                    row_4 = (_e836 + LOO_STRIDE);
                }
            }
            let _e839 = gsum;
            let _e840 = total(_e839);
            let _e842 = n_read;
            let _e843 = div(_e840, _e842);
            let _e845 = keep((g0_ + _e843));
            let _e851 = ghi;
            let _e852 = gmin;
            if ((_e851 - _e852) <= (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e845))))) {
                continue;
            }
            let _e861 = g_1;
            let _e862 = linked_without(c_4, genes_2, linker_2, 0u, _e861, _e845);
            w_10 = 0u;
            loop {
                let _e867 = w_10;
                if (_e867 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e872 = w_10;
                    m1_[_e872] = vec2<f32>(0f, 0f);
                    let _e876 = w_10;
                    let _e877 = wrapped(_e862, _e876);
                    let _e880 = w_10;
                    lo1_[_e880] = _e877;
                    let _e885 = w_10;
                    let _e887 = w_10;
                    let _e889 = lo1_[_e887];
                    hi1_[_e885] = _e889;
                }
                continuing {
                    let _e892 = w_10;
                    w_10 = (_e892 + 1u);
                }
            }
            row_5 = 0u;
            loop {
                let _e897 = row_5;
                if (_e897 < nt) {
                } else {
                    break;
                }
                {
                    let _e900 = row_5;
                    let _e902 = g_1;
                    let _e903 = linked_without(c_4, genes_2, linker_2, _e900, _e902, _e845);
                    w_11 = 0u;
                    loop {
                        let _e908 = w_11;
                        if (_e908 < N_WRAPPERS) {
                        } else {
                            break;
                        }
                        {
                            let _e911 = w_11;
                            let _e912 = wrapped(_e903, _e911);
                            let _e915 = w_11;
                            let _e917 = m1_[_e915];
                            let _e918 = add(_e917, _e912);
                            let _e921 = w_11;
                            m1_[_e921] = _e918;
                            let _e925 = w_11;
                            let _e927 = w_11;
                            let _e929 = lo1_[_e927];
                            lo1_[_e925] = min(_e929, _e912);
                            let _e933 = w_11;
                            let _e935 = w_11;
                            let _e937 = hi1_[_e935];
                            hi1_[_e933] = max(_e937, _e912);
                        }
                        continuing {
                            let _e941 = w_11;
                            w_11 = (_e941 + 1u);
                        }
                    }
                }
                continuing {
                    let _e945 = row_5;
                    row_5 = (_e945 + LOO_STRIDE);
                }
            }
            w_12 = 0u;
            loop {
                let _e951 = w_12;
                if (_e951 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e956 = w_12;
                    xx[_e956] = vec2<f32>(0f, 0f);
                    let _e962 = w_12;
                    xy[_e962] = vec2<f32>(0f, 0f);
                }
                continuing {
                    let _e967 = w_12;
                    w_12 = (_e967 + 1u);
                }
            }
            row_6 = 0u;
            loop {
                let _e972 = row_6;
                if (_e972 < nt) {
                } else {
                    break;
                }
                {
                    let _e975 = row_6;
                    let _e977 = g_1;
                    let _e978 = linked_without(c_4, genes_2, linker_2, _e975, _e977, _e845);
                    let _e982 = row_6;
                    let _e984 = y_1[_e982];
                    let _e986 = sp.y_mean_train;
                    let _e988 = keep((_e984 - _e986));
                    w_13 = 0u;
                    loop {
                        let _e993 = w_13;
                        if (_e993 < N_WRAPPERS) {
                        } else {
                            break;
                        }
                        {
                            let _e996 = w_13;
                            let _e997 = wrapped(_e978, _e996);
                            let _e1000 = w_13;
                            let _e1002 = m1_[_e1000];
                            let _e1003 = total(_e1002);
                            let _e1005 = n_read;
                            let _e1006 = div(_e1003, _e1005);
                            let _e1008 = keep((_e997 - _e1006));
                            let _e1010 = keep((_e1008 * _e1008));
                            let _e1013 = w_13;
                            let _e1015 = xx[_e1013];
                            let _e1016 = add(_e1015, _e1010);
                            let _e1019 = w_13;
                            xx[_e1019] = _e1016;
                            let _e1022 = keep((_e1008 * _e988));
                            let _e1025 = w_13;
                            let _e1027 = xy[_e1025];
                            let _e1028 = add(_e1027, _e1022);
                            let _e1031 = w_13;
                            xy[_e1031] = _e1028;
                        }
                        continuing {
                            let _e1035 = w_13;
                            w_13 = (_e1035 + 1u);
                        }
                    }
                }
                continuing {
                    let _e1039 = row_6;
                    row_6 = (_e1039 + LOO_STRIDE);
                }
            }
            w_14 = 0u;
            loop {
                let _e1045 = w_14;
                if (_e1045 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e1049 = w_14;
                    let _e1051 = ok[_e1049];
                    if !(_e1051) {
                        continue;
                    }
                    let _e1055 = w_14;
                    let o_1 = (out + (_e1055 * WIDTH));
                    let _e1062 = scores[(o_1 + 2u)];
                    let _e1065 = keep((_e1062 * f32(nt)));
                    let _e1068 = keep((_e757 * 4f));
                    let _e1071 = div(_e1065, max(_e1068, 0.000000000000000000000000000001f));
                    let _e1074 = keep((1f - _e1071));
                    let _e1077 = w_14;
                    let _e1079 = xx[_e1077];
                    let _e1080 = total(_e1079);
                    let _e1083 = w_14;
                    let _e1085 = xy[_e1083];
                    let _e1086 = total(_e1085);
                    without = 0f;
                    let _e1091 = w_14;
                    let _e1093 = m1_[_e1091];
                    let _e1094 = total(_e1093);
                    let _e1096 = n_read;
                    let _e1097 = div(_e1094, _e1096);
                    let _e1104 = w_14;
                    let _e1106 = hi1_[_e1104];
                    let _e1107 = w_14;
                    let _e1109 = lo1_[_e1107];
                    let varies = ((_e1106 - _e1109) > (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e1097)))));
                    let _e1116 = finite(_e1080);
                    let _e1117 = finite(_e1086);
                    if ((((varies && (_e1080 > 0f)) && (_e757 > 0f)) && _e1116) && _e1117) {
                        let _e1126 = keep((_e1086 * _e1086));
                        let _e1128 = keep((_e1080 * _e757));
                        let _e1129 = div(_e1126, _e1128);
                        without = clamp(_e1129, 0f, 1f);
                    }
                    l = 1f;
                    if (_e1074 > 0.000001f) {
                        let _e1139 = without;
                        let _e1140 = div(_e1139, _e1074);
                        let _e1143 = keep((1f - _e1140));
                        l = clamp(_e1143, 0f, 1f);
                    }
                    let _e1153 = w_14;
                    let _e1155 = w_14;
                    let _e1157 = n_loss[_e1155];
                    let _e1160 = l;
                    loss[((_e1153 * 8u) + _e1157)] = _e1160;
                    let _e1164 = w_14;
                    let _e1166 = w_14;
                    let _e1168 = n_loss[_e1166];
                    n_loss[_e1164] = (_e1168 + 1u);
                }
                continuing {
                    let _e1172 = w_14;
                    w_14 = (_e1172 + 1u);
                }
            }
        }
        continuing {
            let _e1176 = g_1;
            g_1 = (_e1176 + 1u);
        }
    }
    loop {
        let _e1180 = w_15;
        let def_f = (_e1180 * 8u);
        let _e1184 = w_15;
        if (_e1184 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e1190 = w_15;
            let _e1192 = ok[_e1190];
            let _e1194 = w_15;
            let _e1196 = n_loss[_e1194];
            if (!(_e1192) || (_e1196 == 0u)) {
                continue;
            }
            let _e1201 = w_15;
            let n_1 = n_loss[_e1201];
            i_3 = 1u;
            loop {
                let _e1207 = i_3;
                if (_e1207 < n_1) {
                } else {
                    break;
                }
                {
                    let _e1211 = i_3;
                    let v_8 = loss[(def_f + _e1211)];
                    let _e1217 = i_3;
                    j = _e1217;
                    loop {
                        let _e1221 = j;
                        let def_a_3 = loss[((def_f + _e1221) - 1u)];
                        let _e1228 = j;
                        if ((_e1228 == 0u) || (def_a_3 <= v_8)) {
                            break;
                        }
                        let _e1234 = j;
                        loss[(def_f + _e1234)] = def_a_3;
                        let _e1239 = j;
                        j = (_e1239 - 1u);
                    }
                    let _e1243 = j;
                    loss[(def_f + _e1243)] = v_8;
                }
                continuing {
                    let _e1248 = i_3;
                    i_3 = (_e1248 + 1u);
                }
            }
            let _e1257 = loss[(def_f + ((n_1 - 1u) / 2u))];
            let _e1261 = loss[(def_f + (n_1 / 2u))];
            let _e1263 = keep((_e1257 + _e1261));
            let _e1266 = keep((0.5f * _e1263));
            let _e1269 = keep((1f - _e1266));
            let _e1274 = w_15;
            scores[((out + (_e1274 * WIDTH)) + 9u)] = _e1269;
        }
        continuing {
            let _e1281 = w_15;
            w_15 = (_e1281 + 1u);
        }
    }
    return;
}
