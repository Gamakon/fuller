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
    let _e3 = sp.identity;
    if (_e3 == 1u) {
        let _e9 = sp.genes_per;
        return ((c * _e9) + g_2);
    }
    let _e17 = sp.genes_per;
    let _e21 = chromosomes[((c * _e17) + g_2)];
    return _e21;
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

    acc = select(0f, 1f, (linker == 1u));
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
            if (linker == 1u) {
                let _e39 = acc;
                let _e41 = keep((_e39 * v_6));
                acc = _e41;
            } else {
                let _e44 = acc;
                let _e46 = keep((_e44 + v_6));
                acc = _e46;
            }
        }
        continuing {
            let _e50 = g_3;
            g_3 = (_e50 + 1u);
        }
    }
    if (linker == 0u) {
        let _e56 = acc;
        let _e60 = div(_e56, f32(countOneBits(genes)));
        acc = _e60;
    }
    let _e63 = acc;
    return _e63;
}

fn linked_without(c_2: u32, genes_1: u32, linker_1: u32, row_8: u32, skip: u32, held: f32) -> f32 {
    var acc_1: f32;
    var g_4: u32 = 0u;
    var v_3: f32;

    acc_1 = select(0f, 1f, (linker_1 == 1u));
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
            if (linker_1 == 1u) {
                let _e53 = acc_1;
                let _e54 = v_3;
                let _e56 = keep((_e53 * _e54));
                acc_1 = _e56;
            } else {
                let _e60 = acc_1;
                let _e61 = v_3;
                let _e63 = keep((_e60 + _e61));
                acc_1 = _e63;
            }
        }
        continuing {
            let _e67 = g_4;
            g_4 = (_e67 + 1u);
        }
    }
    if (linker_1 == 0u) {
        let _e73 = acc_1;
        let _e77 = div(_e73, f32(countOneBits(genes_1)));
        acc_1 = _e77;
    }
    let _e80 = acc_1;
    return _e80;
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
                let _e489 = w_7;
                if (_e489 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e493 = w_7;
                    let _e495 = ok[_e493];
                    if _e495 {
                        let _e497 = w_7;
                        let _e498 = wrapped(_e472, _e497);
                        let _e501 = w_7;
                        let _e503 = a[_e501];
                        let _e505 = keep((_e503 * _e498));
                        let _e508 = w_7;
                        let _e510 = b[_e508];
                        let _e512 = keep((_e505 + _e510));
                        let _e515 = row_2;
                        let _e517 = y_1[_e515];
                        let _e519 = keep((_e517 - _e512));
                        let _e521 = keep((_e519 * _e519));
                        let _e526 = w_7;
                        let _e528 = split;
                        let _e531 = sq[((_e526 * 3u) + _e528)];
                        let _e532 = add(_e531, _e521);
                        let _e537 = w_7;
                        let _e539 = split;
                        sq[((_e537 * 3u) + _e539)] = _e532;
                        let _e546 = w_7;
                        let _e548 = split;
                        let _e551 = ab[((_e546 * 3u) + _e548)];
                        let _e553 = add(_e551, abs(_e519));
                        let _e558 = w_7;
                        let _e560 = split;
                        ab[((_e558 * 3u) + _e560)] = _e553;
                        let _e565 = split;
                        if (_e565 == 1u) {
                            let _e569 = w_7;
                            let _e571 = w_7;
                            let _e573 = worst[_e571];
                            worst[_e569] = max(_e573, abs(_e519));
                        }
                    }
                }
                continuing {
                    let _e578 = w_7;
                    w_7 = (_e578 + 1u);
                }
            }
        }
        continuing {
            let _e582 = row_2;
            row_2 = (_e582 + 1u);
        }
    }
    loop {
        let _e586 = w_8;
        if (_e586 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e590 = w_8;
            let _e592 = ok[_e590];
            if !(_e592) {
                continue;
            }
            let _e596 = w_8;
            let o = (out + (_e596 * WIDTH));
            let _e603 = sp.n_val;
            let _e606 = sp.n_extrap;
            let n = array<f32, 3>(f32(nt), f32(_e603), f32(max(_e606, 1u)));
            let _e614 = w_8;
            let _e616 = a[_e614];
            scores[o] = _e616;
            let _e623 = w_8;
            let _e625 = b[_e623];
            scores[(o + 1u)] = _e625;
            let _e629 = w_8;
            let _e632 = sq[(_e629 * 3u)];
            let _e633 = total(_e632);
            let _e635 = div(_e633, n[0]);
            scores[(o + 2u)] = _e635;
            let _e644 = w_8;
            let _e648 = sq[((_e644 * 3u) + 1u)];
            let _e649 = total(_e648);
            let _e651 = div(_e649, n[1]);
            scores[(o + 3u)] = _e651;
            let _e662 = w_8;
            let _e664 = worst[_e662];
            scores[(o + 4u)] = _e664;
            let _e669 = w_8;
            let _e673 = sq[((_e669 * 3u) + 2u)];
            let _e674 = total(_e673);
            let _e676 = div(_e674, n[2]);
            scores[(o + 5u)] = _e676;
            let _e684 = w_8;
            let _e687 = ab[(_e684 * 3u)];
            let _e688 = total(_e687);
            let _e690 = div(_e688, n[0]);
            scores[(o + 6u)] = _e690;
            let _e699 = w_8;
            let _e703 = ab[((_e699 * 3u) + 1u)];
            let _e704 = total(_e703);
            let _e706 = div(_e704, n[1]);
            scores[(o + 7u)] = _e706;
            let _e715 = w_8;
            let _e719 = ab[((_e715 * 3u) + 2u)];
            let _e720 = total(_e719);
            let _e722 = div(_e720, n[2]);
            scores[(o + 8u)] = _e722;
            scores[(o + 9u)] = 0f;
            i_2 = 0u;
            loop {
                let _e736 = i_2;
                if (_e736 < WIDTH) {
                } else {
                    break;
                }
                {
                    let _e740 = i_2;
                    let _e743 = scores[(o + _e740)];
                    let _e744 = finite(_e743);
                    if !(_e744) {
                        k = 0u;
                        loop {
                            let _e750 = k;
                            if (_e750 < WIDTH) {
                            } else {
                                break;
                            }
                            {
                                let _e754 = k;
                                scores[(o + _e754)] = nan;
                            }
                            continuing {
                                let _e759 = k;
                                k = (_e759 + 1u);
                            }
                        }
                        let _e764 = w_8;
                        ok[_e764] = false;
                        break;
                    }
                }
                continuing {
                    let _e768 = i_2;
                    i_2 = (_e768 + 1u);
                }
            }
        }
        continuing {
            let _e772 = w_8;
            w_8 = (_e772 + 1u);
        }
    }
    if (countOneBits(genes_2) == 1u) {
        return;
    }
    loop {
        let _e778 = row_3;
        if (_e778 < nt) {
        } else {
            break;
        }
        {
            let _e783 = row_3;
            let _e785 = y_1[_e783];
            let _e787 = sp.y_mean_train;
            let _e789 = keep((_e785 - _e787));
            let _e791 = keep((_e789 * _e789));
            let _e793 = syy;
            let _e794 = add(_e793, _e791);
            syy = _e794;
        }
        continuing {
            let _e798 = row_3;
            row_3 = (_e798 + LOO_STRIDE);
        }
    }
    let _e801 = syy;
    let _e802 = total(_e801);
    loop {
        let _e805 = w_9;
        if (_e805 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e810 = w_9;
            n_loss[_e810] = 0u;
        }
        continuing {
            let _e814 = w_9;
            w_9 = (_e814 + 1u);
        }
    }
    loop {
        let _e819 = g_1;
        let _e821 = sp.genes_per;
        if (_e819 < min(_e821, 8u)) {
        } else {
            break;
        }
        {
            let _e827 = g_1;
            if (((genes_2 >> _e827) & 1u) == 0u) {
                continue;
            }
            let _e832 = g_1;
            let _e833 = gene_index(c_4, _e832);
            let _e837 = sp.n_rows;
            let g0_ = preds[(_e833 * _e837)];
            gsum = vec2<f32>(0f, 0f);
            gmin = g0_;
            ghi = g0_;
            n_read = 0f;
            row_4 = 0u;
            loop {
                let _e851 = row_4;
                if (_e851 < nt) {
                } else {
                    break;
                }
                {
                    let _e857 = sp.n_rows;
                    let _e859 = row_4;
                    let v_7 = preds[((_e833 * _e857) + _e859)];
                    let _e864 = keep((v_7 - g0_));
                    let _e866 = gsum;
                    let _e867 = add(_e866, _e864);
                    gsum = _e867;
                    let _e870 = gmin;
                    gmin = min(_e870, v_7);
                    let _e873 = ghi;
                    ghi = max(_e873, v_7);
                    let _e877 = n_read;
                    n_read = (_e877 + 1f);
                }
                continuing {
                    let _e881 = row_4;
                    row_4 = (_e881 + LOO_STRIDE);
                }
            }
            let _e884 = gsum;
            let _e885 = total(_e884);
            let _e887 = n_read;
            let _e888 = div(_e885, _e887);
            let _e890 = keep((g0_ + _e888));
            let _e896 = ghi;
            let _e897 = gmin;
            if ((_e896 - _e897) <= (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e890))))) {
                continue;
            }
            let _e906 = g_1;
            let _e907 = linked_without(c_4, genes_2, linker_2, 0u, _e906, _e890);
            w_10 = 0u;
            loop {
                let _e912 = w_10;
                if (_e912 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e917 = w_10;
                    m1_[_e917] = vec2<f32>(0f, 0f);
                    let _e921 = w_10;
                    let _e922 = wrapped(_e907, _e921);
                    let _e925 = w_10;
                    lo1_[_e925] = _e922;
                    let _e930 = w_10;
                    let _e932 = w_10;
                    let _e934 = lo1_[_e932];
                    hi1_[_e930] = _e934;
                }
                continuing {
                    let _e937 = w_10;
                    w_10 = (_e937 + 1u);
                }
            }
            row_5 = 0u;
            loop {
                let _e942 = row_5;
                if (_e942 < nt) {
                } else {
                    break;
                }
                {
                    let _e945 = row_5;
                    let _e947 = g_1;
                    let _e948 = linked_without(c_4, genes_2, linker_2, _e945, _e947, _e890);
                    w_11 = 0u;
                    loop {
                        let _e953 = w_11;
                        if (_e953 < N_WRAPPERS) {
                        } else {
                            break;
                        }
                        {
                            let _e956 = w_11;
                            let _e957 = wrapped(_e948, _e956);
                            let _e960 = w_11;
                            let _e962 = m1_[_e960];
                            let _e963 = add(_e962, _e957);
                            let _e966 = w_11;
                            m1_[_e966] = _e963;
                            let _e970 = w_11;
                            let _e972 = w_11;
                            let _e974 = lo1_[_e972];
                            lo1_[_e970] = min(_e974, _e957);
                            let _e978 = w_11;
                            let _e980 = w_11;
                            let _e982 = hi1_[_e980];
                            hi1_[_e978] = max(_e982, _e957);
                        }
                        continuing {
                            let _e986 = w_11;
                            w_11 = (_e986 + 1u);
                        }
                    }
                }
                continuing {
                    let _e990 = row_5;
                    row_5 = (_e990 + LOO_STRIDE);
                }
            }
            w_12 = 0u;
            loop {
                let _e996 = w_12;
                if (_e996 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e1001 = w_12;
                    xx[_e1001] = vec2<f32>(0f, 0f);
                    let _e1007 = w_12;
                    xy[_e1007] = vec2<f32>(0f, 0f);
                }
                continuing {
                    let _e1012 = w_12;
                    w_12 = (_e1012 + 1u);
                }
            }
            row_6 = 0u;
            loop {
                let _e1017 = row_6;
                if (_e1017 < nt) {
                } else {
                    break;
                }
                {
                    let _e1020 = row_6;
                    let _e1022 = g_1;
                    let _e1023 = linked_without(c_4, genes_2, linker_2, _e1020, _e1022, _e890);
                    let _e1027 = row_6;
                    let _e1029 = y_1[_e1027];
                    let _e1031 = sp.y_mean_train;
                    let _e1033 = keep((_e1029 - _e1031));
                    w_13 = 0u;
                    loop {
                        let _e1038 = w_13;
                        if (_e1038 < N_WRAPPERS) {
                        } else {
                            break;
                        }
                        {
                            let _e1041 = w_13;
                            let _e1042 = wrapped(_e1023, _e1041);
                            let _e1045 = w_13;
                            let _e1047 = m1_[_e1045];
                            let _e1048 = total(_e1047);
                            let _e1050 = n_read;
                            let _e1051 = div(_e1048, _e1050);
                            let _e1053 = keep((_e1042 - _e1051));
                            let _e1055 = keep((_e1053 * _e1053));
                            let _e1058 = w_13;
                            let _e1060 = xx[_e1058];
                            let _e1061 = add(_e1060, _e1055);
                            let _e1064 = w_13;
                            xx[_e1064] = _e1061;
                            let _e1067 = keep((_e1053 * _e1033));
                            let _e1070 = w_13;
                            let _e1072 = xy[_e1070];
                            let _e1073 = add(_e1072, _e1067);
                            let _e1076 = w_13;
                            xy[_e1076] = _e1073;
                        }
                        continuing {
                            let _e1080 = w_13;
                            w_13 = (_e1080 + 1u);
                        }
                    }
                }
                continuing {
                    let _e1084 = row_6;
                    row_6 = (_e1084 + LOO_STRIDE);
                }
            }
            w_14 = 0u;
            loop {
                let _e1090 = w_14;
                if (_e1090 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e1094 = w_14;
                    let _e1096 = ok[_e1094];
                    if !(_e1096) {
                        continue;
                    }
                    let _e1100 = w_14;
                    let o_1 = (out + (_e1100 * WIDTH));
                    let _e1107 = scores[(o_1 + 2u)];
                    let _e1110 = keep((_e1107 * f32(nt)));
                    let _e1113 = keep((_e802 * 4f));
                    let _e1116 = div(_e1110, max(_e1113, 0.000000000000000000000000000001f));
                    let _e1119 = keep((1f - _e1116));
                    let _e1122 = w_14;
                    let _e1124 = xx[_e1122];
                    let _e1125 = total(_e1124);
                    let _e1128 = w_14;
                    let _e1130 = xy[_e1128];
                    let _e1131 = total(_e1130);
                    without = 0f;
                    let _e1136 = w_14;
                    let _e1138 = m1_[_e1136];
                    let _e1139 = total(_e1138);
                    let _e1141 = n_read;
                    let _e1142 = div(_e1139, _e1141);
                    let _e1149 = w_14;
                    let _e1151 = hi1_[_e1149];
                    let _e1152 = w_14;
                    let _e1154 = lo1_[_e1152];
                    let varies = ((_e1151 - _e1154) > (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e1142)))));
                    let _e1161 = finite(_e1125);
                    let _e1162 = finite(_e1131);
                    if ((((varies && (_e1125 > 0f)) && (_e802 > 0f)) && _e1161) && _e1162) {
                        let _e1171 = keep((_e1131 * _e1131));
                        let _e1173 = keep((_e1125 * _e802));
                        let _e1174 = div(_e1171, _e1173);
                        without = clamp(_e1174, 0f, 1f);
                    }
                    l = 1f;
                    if (_e1119 > 0.000001f) {
                        let _e1184 = without;
                        let _e1185 = div(_e1184, _e1119);
                        let _e1188 = keep((1f - _e1185));
                        l = clamp(_e1188, 0f, 1f);
                    }
                    let _e1198 = w_14;
                    let _e1200 = w_14;
                    let _e1202 = n_loss[_e1200];
                    let _e1205 = l;
                    loss[((_e1198 * 8u) + _e1202)] = _e1205;
                    let _e1209 = w_14;
                    let _e1211 = w_14;
                    let _e1213 = n_loss[_e1211];
                    n_loss[_e1209] = (_e1213 + 1u);
                }
                continuing {
                    let _e1217 = w_14;
                    w_14 = (_e1217 + 1u);
                }
            }
        }
        continuing {
            let _e1221 = g_1;
            g_1 = (_e1221 + 1u);
        }
    }
    loop {
        let _e1225 = w_15;
        if (_e1225 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e1231 = w_15;
            let _e1233 = ok[_e1231];
            let _e1235 = w_15;
            let _e1237 = n_loss[_e1235];
            if (!(_e1233) || (_e1237 == 0u)) {
                continue;
            }
            let _e1242 = w_15;
            let n_1 = n_loss[_e1242];
            i_3 = 1u;
            loop {
                let _e1248 = i_3;
                if (_e1248 < n_1) {
                } else {
                    break;
                }
                {
                    let _e1254 = w_15;
                    let _e1256 = i_3;
                    let v_8 = loss[((_e1254 * 8u) + _e1256)];
                    let _e1262 = i_3;
                    j = _e1262;
                    loop {
                        let _e1269 = j;
                        let _e1271 = w_15;
                        let _e1273 = j;
                        let _e1277 = loss[(((_e1271 * 8u) + _e1273) - 1u)];
                        if ((_e1269 == 0u) || (_e1277 <= v_8)) {
                            break;
                        }
                        let _e1285 = w_15;
                        let _e1287 = j;
                        let _e1290 = w_15;
                        let _e1292 = j;
                        let _e1296 = loss[(((_e1290 * 8u) + _e1292) - 1u)];
                        loss[((_e1285 * 8u) + _e1287)] = _e1296;
                        let _e1299 = j;
                        j = (_e1299 - 1u);
                    }
                    let _e1305 = w_15;
                    let _e1307 = j;
                    loss[((_e1305 * 8u) + _e1307)] = v_8;
                }
                continuing {
                    let _e1312 = i_3;
                    i_3 = (_e1312 + 1u);
                }
            }
            let _e1319 = w_15;
            let _e1325 = loss[((_e1319 * 8u) + ((n_1 - 1u) / 2u))];
            let _e1326 = w_15;
            let _e1331 = loss[((_e1326 * 8u) + (n_1 / 2u))];
            let _e1333 = keep((_e1325 + _e1331));
            let _e1336 = keep((0.5f * _e1333));
            let _e1339 = keep((1f - _e1336));
            let _e1344 = w_15;
            scores[((out + (_e1344 * WIDTH)) + 9u)] = _e1339;
        }
        continuing {
            let _e1351 = w_15;
            w_15 = (_e1351 + 1u);
        }
    }
    return;
}
