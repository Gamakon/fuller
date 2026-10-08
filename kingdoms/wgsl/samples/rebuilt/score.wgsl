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
            if (linker == 1u) {
                let _e33 = acc;
                let _e35 = sp.n_rows;
                let _e39 = preds[((_e25 * _e35) + row_7)];
                let _e41 = keep((_e33 * _e39));
                acc = _e41;
            } else {
                let _e47 = acc;
                let _e49 = sp.n_rows;
                let _e53 = preds[((_e25 * _e49) + row_7)];
                let _e55 = keep((_e47 + _e53));
                acc = _e55;
            }
        }
        continuing {
            let _e59 = g_3;
            g_3 = (_e59 + 1u);
        }
    }
    if (linker == 0u) {
        let _e65 = acc;
        let _e69 = div(_e65, f32(countOneBits(genes)));
        acc = _e69;
    }
    let _e72 = acc;
    return _e72;
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

    let _e23 = sp.n_chromosomes;
    if (gid.x >= (_e23 * arrayLength((&combinations)))) {
        return;
    }
    loop {
        let _e29 = i;
        if (_e29 < 30u) {
        } else {
            break;
        }
        {
            let _e40 = sp.row_base;
            let _e53 = i;
            scores[((((((_e40 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + _e53)] = bitcast<f32>(NAN_BITS);
        }
        continuing {
            let _e59 = i;
            i = (_e59 + 1u);
        }
    }
    loop {
        let _e63 = g;
        let _e65 = sp.genes_per;
        if (_e63 < _e65) {
        } else {
            break;
        }
        {
            let _e73 = g;
            let _e74 = gene_index((gid.x / arrayLength((&combinations))), _e73);
            let _e88 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e90 = g;
            let _e95 = sp.gene_base;
            let _e98 = gene_ok[(_e95 + _e74)];
            if (((((_e88 & 16777215u) >> _e90) & 1u) == 1u) && (_e98 == 0u)) {
                return;
            }
        }
        continuing {
            let _e103 = g;
            g = (_e103 + 1u);
        }
    }
    let _e118 = combinations[(gid.x % arrayLength((&combinations)))];
    let _e128 = combinations[(gid.x % arrayLength((&combinations)))];
    let _e131 = linked((gid.x / arrayLength((&combinations))), (_e118 & 16777215u), (_e128 >> 24u), 0u);
    loop {
        let _e134 = w;
        if (_e134 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e139 = w;
            sum[_e139] = vec2<f32>(0f, 0f);
            let _e145 = w;
            ok[_e145] = true;
            let _e148 = w;
            let _e149 = wrapped(_e131, _e148);
            let _e152 = w;
            x0_[_e152] = _e149;
            let _e157 = w;
            let _e159 = w;
            let _e161 = x0_[_e159];
            lo[_e157] = _e161;
            let _e165 = w;
            let _e167 = w;
            let _e169 = x0_[_e167];
            hi[_e165] = _e169;
        }
        continuing {
            let _e172 = w;
            w = (_e172 + 1u);
        }
    }
    loop {
        let _e176 = row;
        let _e178 = sp.n_rows;
        if (_e176 < _e178) {
        } else {
            break;
        }
        {
            let _e193 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e203 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e206 = row;
            let _e207 = linked((gid.x / arrayLength((&combinations))), (_e193 & 16777215u), (_e203 >> 24u), _e206);
            w_1 = 0u;
            loop {
                let _e212 = w_1;
                if (_e212 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e215 = w_1;
                    let _e216 = wrapped(_e207, _e215);
                    let _e217 = finite(_e207);
                    let _e218 = finite(_e216);
                    if (!(_e217) || !(_e218)) {
                        let _e225 = w_1;
                        ok[_e225] = false;
                    } else {
                        let _e229 = row;
                        let _e231 = sp.n_train;
                        if (_e229 < _e231) {
                            let _e235 = w_1;
                            let _e237 = x0_[_e235];
                            let _e239 = keep((_e216 - _e237));
                            let _e242 = w_1;
                            let _e244 = sum[_e242];
                            let _e245 = add(_e244, _e239);
                            let _e248 = w_1;
                            sum[_e248] = _e245;
                            let _e252 = w_1;
                            let _e254 = w_1;
                            let _e256 = lo[_e254];
                            lo[_e252] = min(_e256, _e216);
                            let _e260 = w_1;
                            let _e262 = w_1;
                            let _e264 = hi[_e262];
                            hi[_e260] = max(_e264, _e216);
                        }
                    }
                }
                continuing {
                    let _e268 = w_1;
                    w_1 = (_e268 + 1u);
                }
            }
        }
        continuing {
            let _e272 = row;
            row = (_e272 + 1u);
        }
    }
    loop {
        let _e276 = w_2;
        if (_e276 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e280 = w_2;
            let _e282 = sum[_e280];
            let _e283 = total(_e282);
            let _e286 = sp.n_train;
            let _e288 = div(_e283, f32(_e286));
            let _e291 = w_2;
            let _e293 = x0_[_e291];
            let _e295 = keep((_e293 + _e288));
            let _e298 = w_2;
            mx[_e298] = _e295;
        }
        continuing {
            let _e302 = w_2;
            w_2 = (_e302 + 1u);
        }
    }
    loop {
        let _e306 = w_3;
        if (_e306 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e311 = w_3;
            sxx[_e311] = vec2<f32>(0f, 0f);
            let _e317 = w_3;
            sxy[_e317] = vec2<f32>(0f, 0f);
        }
        continuing {
            let _e322 = w_3;
            w_3 = (_e322 + 1u);
        }
    }
    loop {
        let _e326 = row_1;
        let _e328 = sp.n_train;
        if (_e326 < _e328) {
        } else {
            break;
        }
        {
            let _e343 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e353 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e356 = row_1;
            let _e357 = linked((gid.x / arrayLength((&combinations))), (_e343 & 16777215u), (_e353 >> 24u), _e356);
            let _e361 = row_1;
            let _e363 = y_1[_e361];
            let _e365 = sp.y_mean_train;
            let _e367 = keep((_e363 - _e365));
            w_4 = 0u;
            loop {
                let _e372 = w_4;
                if (_e372 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e376 = w_4;
                    let _e378 = ok[_e376];
                    if _e378 {
                        let _e380 = w_4;
                        let _e381 = wrapped(_e357, _e380);
                        let _e384 = w_4;
                        let _e386 = mx[_e384];
                        let _e388 = keep((_e381 - _e386));
                        let _e390 = keep((_e388 * _e388));
                        let _e393 = w_4;
                        let _e395 = sxx[_e393];
                        let _e396 = add(_e395, _e390);
                        let _e399 = w_4;
                        sxx[_e399] = _e396;
                        let _e402 = keep((_e388 * _e367));
                        let _e405 = w_4;
                        let _e407 = sxy[_e405];
                        let _e408 = add(_e407, _e402);
                        let _e411 = w_4;
                        sxy[_e411] = _e408;
                    }
                }
                continuing {
                    let _e415 = w_4;
                    w_4 = (_e415 + 1u);
                }
            }
        }
        continuing {
            let _e419 = row_1;
            row_1 = (_e419 + 1u);
        }
    }
    loop {
        let _e423 = w_5;
        if (_e423 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e427 = w_5;
            let _e429 = sxx[_e427];
            let _e430 = total(_e429);
            let _e440 = w_5;
            let _e442 = ok[_e440];
            let _e444 = w_5;
            let _e446 = hi[_e444];
            let _e447 = w_5;
            let _e449 = lo[_e447];
            let _e451 = w_5;
            let _e453 = mx[_e451];
            if ((!(_e442) || ((_e446 - _e449) <= (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e453)))))) || (_e430 <= 0f)) {
                let _e465 = w_5;
                ok[_e465] = false;
                continue;
            }
            let _e469 = w_5;
            let _e471 = sxy[_e469];
            let _e472 = total(_e471);
            let _e473 = div(_e472, _e430);
            let _e476 = w_5;
            a[_e476] = _e473;
            let _e481 = w_5;
            let _e483 = a[_e481];
            let _e484 = w_5;
            let _e486 = mx[_e484];
            let _e488 = keep((_e483 * _e486));
            let _e491 = sp.y_mean_train;
            let _e493 = keep((_e491 - _e488));
            let _e496 = w_5;
            b[_e496] = _e493;
            let _e500 = w_5;
            let _e502 = a[_e500];
            let _e503 = finite(_e502);
            let _e506 = w_5;
            let _e508 = b[_e506];
            let _e509 = finite(_e508);
            if (!(_e503) || !(_e509)) {
                let _e516 = w_5;
                ok[_e516] = false;
            }
        }
        continuing {
            let _e520 = w_5;
            w_5 = (_e520 + 1u);
        }
    }
    loop {
        let _e524 = i_1;
        if (_e524 < 9u) {
        } else {
            break;
        }
        {
            let _e529 = i_1;
            sq[_e529] = vec2<f32>(0f, 0f);
            let _e535 = i_1;
            ab[_e535] = vec2<f32>(0f, 0f);
        }
        continuing {
            let _e540 = i_1;
            i_1 = (_e540 + 1u);
        }
    }
    loop {
        let _e544 = w_6;
        if (_e544 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e549 = w_6;
            worst[_e549] = 0f;
        }
        continuing {
            let _e553 = w_6;
            w_6 = (_e553 + 1u);
        }
    }
    loop {
        let _e557 = row_2;
        let _e559 = sp.n_rows;
        if (_e557 < _e559) {
        } else {
            break;
        }
        {
            let _e574 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e584 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e587 = row_2;
            let _e588 = linked((gid.x / arrayLength((&combinations))), (_e574 & 16777215u), (_e584 >> 24u), _e587);
            split = 2u;
            let _e593 = row_2;
            let _e595 = sp.n_train;
            if (_e593 < _e595) {
                split = 0u;
            } else {
                let _e601 = row_2;
                let _e603 = sp.n_train;
                let _e605 = sp.n_val;
                if (_e601 < (_e603 + _e605)) {
                    split = 1u;
                }
            }
            w_7 = 0u;
            loop {
                let _e614 = w_7;
                if (_e614 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e618 = w_7;
                    let _e620 = ok[_e618];
                    if _e620 {
                        let _e622 = w_7;
                        let _e623 = wrapped(_e588, _e622);
                        let _e626 = w_7;
                        let _e628 = a[_e626];
                        let _e630 = keep((_e628 * _e623));
                        let _e633 = w_7;
                        let _e635 = b[_e633];
                        let _e637 = keep((_e630 + _e635));
                        let _e640 = row_2;
                        let _e642 = y_1[_e640];
                        let _e644 = keep((_e642 - _e637));
                        let _e646 = keep((_e644 * _e644));
                        let _e651 = w_7;
                        let _e653 = split;
                        let _e656 = sq[((_e651 * 3u) + _e653)];
                        let _e657 = add(_e656, _e646);
                        let _e662 = w_7;
                        let _e664 = split;
                        sq[((_e662 * 3u) + _e664)] = _e657;
                        let _e671 = w_7;
                        let _e673 = split;
                        let _e676 = ab[((_e671 * 3u) + _e673)];
                        let _e678 = add(_e676, abs(_e644));
                        let _e683 = w_7;
                        let _e685 = split;
                        ab[((_e683 * 3u) + _e685)] = _e678;
                        let _e690 = split;
                        if (_e690 == 1u) {
                            let _e694 = w_7;
                            let _e696 = w_7;
                            let _e698 = worst[_e696];
                            worst[_e694] = max(_e698, abs(_e644));
                        }
                    }
                }
                continuing {
                    let _e703 = w_7;
                    w_7 = (_e703 + 1u);
                }
            }
        }
        continuing {
            let _e707 = row_2;
            row_2 = (_e707 + 1u);
        }
    }
    loop {
        let _e711 = w_8;
        if (_e711 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e715 = w_8;
            let _e717 = ok[_e715];
            if !(_e717) {
                continue;
            }
            let _e728 = sp.row_base;
            let _e741 = w_8;
            let _e745 = w_8;
            let _e747 = a[_e745];
            scores[((((((_e728 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e741 * WIDTH))] = _e747;
            let _e758 = sp.row_base;
            let _e771 = w_8;
            let _e776 = w_8;
            let _e778 = b[_e776];
            scores[(((((((_e758 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e771 * WIDTH)) + 1u)] = _e778;
            let _e782 = w_8;
            let _e785 = sq[(_e782 * 3u)];
            let _e786 = total(_e785);
            let _e790 = sp.n_train;
            let _e793 = sp.n_val;
            let _e796 = sp.n_extrap;
            let _e801 = div(_e786, array<f32, 3>(f32(_e790), f32(_e793), f32(max(_e796, 1u)))[0]);
            let _e811 = sp.row_base;
            let _e824 = w_8;
            scores[(((((((_e811 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e824 * WIDTH)) + 2u)] = _e801;
            let _e833 = w_8;
            let _e837 = sq[((_e833 * 3u) + 1u)];
            let _e838 = total(_e837);
            let _e842 = sp.n_train;
            let _e845 = sp.n_val;
            let _e848 = sp.n_extrap;
            let _e853 = div(_e838, array<f32, 3>(f32(_e842), f32(_e845), f32(max(_e848, 1u)))[1]);
            let _e863 = sp.row_base;
            let _e876 = w_8;
            scores[(((((((_e863 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e876 * WIDTH)) + 3u)] = _e853;
            let _e891 = sp.row_base;
            let _e904 = w_8;
            let _e909 = w_8;
            let _e911 = worst[_e909];
            scores[(((((((_e891 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e904 * WIDTH)) + 4u)] = _e911;
            let _e916 = w_8;
            let _e920 = sq[((_e916 * 3u) + 2u)];
            let _e921 = total(_e920);
            let _e925 = sp.n_train;
            let _e928 = sp.n_val;
            let _e931 = sp.n_extrap;
            let _e936 = div(_e921, array<f32, 3>(f32(_e925), f32(_e928), f32(max(_e931, 1u)))[2]);
            let _e946 = sp.row_base;
            let _e959 = w_8;
            scores[(((((((_e946 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e959 * WIDTH)) + 5u)] = _e936;
            let _e967 = w_8;
            let _e970 = ab[(_e967 * 3u)];
            let _e971 = total(_e970);
            let _e975 = sp.n_train;
            let _e978 = sp.n_val;
            let _e981 = sp.n_extrap;
            let _e986 = div(_e971, array<f32, 3>(f32(_e975), f32(_e978), f32(max(_e981, 1u)))[0]);
            let _e996 = sp.row_base;
            let _e1009 = w_8;
            scores[(((((((_e996 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1009 * WIDTH)) + 6u)] = _e986;
            let _e1018 = w_8;
            let _e1022 = ab[((_e1018 * 3u) + 1u)];
            let _e1023 = total(_e1022);
            let _e1027 = sp.n_train;
            let _e1030 = sp.n_val;
            let _e1033 = sp.n_extrap;
            let _e1038 = div(_e1023, array<f32, 3>(f32(_e1027), f32(_e1030), f32(max(_e1033, 1u)))[1]);
            let _e1048 = sp.row_base;
            let _e1061 = w_8;
            scores[(((((((_e1048 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1061 * WIDTH)) + 7u)] = _e1038;
            let _e1070 = w_8;
            let _e1074 = ab[((_e1070 * 3u) + 2u)];
            let _e1075 = total(_e1074);
            let _e1079 = sp.n_train;
            let _e1082 = sp.n_val;
            let _e1085 = sp.n_extrap;
            let _e1090 = div(_e1075, array<f32, 3>(f32(_e1079), f32(_e1082), f32(max(_e1085, 1u)))[2]);
            let _e1100 = sp.row_base;
            let _e1113 = w_8;
            scores[(((((((_e1100 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1113 * WIDTH)) + 8u)] = _e1090;
            let _e1128 = sp.row_base;
            let _e1141 = w_8;
            scores[(((((((_e1128 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1141 * WIDTH)) + 9u)] = 0f;
            i_2 = 0u;
            loop {
                let _e1150 = i_2;
                if (_e1150 < WIDTH) {
                } else {
                    break;
                }
                {
                    let _e1161 = sp.row_base;
                    let _e1174 = w_8;
                    let _e1177 = i_2;
                    let _e1180 = scores[(((((((_e1161 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1174 * WIDTH)) + _e1177)];
                    let _e1181 = finite(_e1180);
                    if !(_e1181) {
                        k = 0u;
                        loop {
                            let _e1187 = k;
                            if (_e1187 < WIDTH) {
                            } else {
                                break;
                            }
                            {
                                let _e1199 = sp.row_base;
                                let _e1212 = w_8;
                                let _e1215 = k;
                                scores[(((((((_e1199 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1212 * WIDTH)) + _e1215)] = bitcast<f32>(NAN_BITS);
                            }
                            continuing {
                                let _e1221 = k;
                                k = (_e1221 + 1u);
                            }
                        }
                        let _e1226 = w_8;
                        ok[_e1226] = false;
                        break;
                    }
                }
                continuing {
                    let _e1230 = i_2;
                    i_2 = (_e1230 + 1u);
                }
            }
        }
        continuing {
            let _e1234 = w_8;
            w_8 = (_e1234 + 1u);
        }
    }
    let _e1245 = combinations[(gid.x % arrayLength((&combinations)))];
    if (countOneBits((_e1245 & 16777215u)) == 1u) {
        return;
    }
    loop {
        let _e1251 = row_3;
        let _e1253 = sp.n_train;
        if (_e1251 < _e1253) {
        } else {
            break;
        }
        {
            let _e1258 = row_3;
            let _e1260 = y_1[_e1258];
            let _e1262 = sp.y_mean_train;
            let _e1264 = keep((_e1260 - _e1262));
            let _e1266 = keep((_e1264 * _e1264));
            let _e1268 = syy;
            let _e1269 = add(_e1268, _e1266);
            syy = _e1269;
        }
        continuing {
            let _e1273 = row_3;
            row_3 = (_e1273 + LOO_STRIDE);
        }
    }
    let _e1276 = syy;
    let _e1277 = total(_e1276);
    loop {
        let _e1280 = w_9;
        if (_e1280 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e1285 = w_9;
            n_loss[_e1285] = 0u;
        }
        continuing {
            let _e1289 = w_9;
            w_9 = (_e1289 + 1u);
        }
    }
    loop {
        let _e1294 = g_1;
        let _e1296 = sp.genes_per;
        if (_e1294 < min(_e1296, 8u)) {
        } else {
            break;
        }
        {
            let _e1310 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e1312 = g_1;
            if ((((_e1310 & 16777215u) >> _e1312) & 1u) == 0u) {
                continue;
            }
            let _e1322 = g_1;
            let _e1323 = gene_index((gid.x / arrayLength((&combinations))), _e1322);
            gsum = vec2<f32>(0f, 0f);
            let _e1331 = sp.n_rows;
            let _e1334 = preds[(_e1323 * _e1331)];
            gmin = _e1334;
            let _e1339 = sp.n_rows;
            let _e1342 = preds[(_e1323 * _e1339)];
            ghi = _e1342;
            n_read = 0f;
            row_4 = 0u;
            loop {
                let _e1349 = row_4;
                let _e1351 = sp.n_train;
                if (_e1349 < _e1351) {
                } else {
                    break;
                }
                {
                    let _e1357 = sp.n_rows;
                    let _e1359 = row_4;
                    let _e1362 = preds[((_e1323 * _e1357) + _e1359)];
                    let _e1364 = sp.n_rows;
                    let _e1367 = preds[(_e1323 * _e1364)];
                    let _e1369 = keep((_e1362 - _e1367));
                    let _e1371 = gsum;
                    let _e1372 = add(_e1371, _e1369);
                    gsum = _e1372;
                    let _e1378 = gmin;
                    let _e1380 = sp.n_rows;
                    let _e1382 = row_4;
                    let _e1385 = preds[((_e1323 * _e1380) + _e1382)];
                    gmin = min(_e1378, _e1385);
                    let _e1391 = ghi;
                    let _e1393 = sp.n_rows;
                    let _e1395 = row_4;
                    let _e1398 = preds[((_e1323 * _e1393) + _e1395)];
                    ghi = max(_e1391, _e1398);
                    let _e1402 = n_read;
                    n_read = (_e1402 + 1f);
                }
                continuing {
                    let _e1406 = row_4;
                    row_4 = (_e1406 + LOO_STRIDE);
                }
            }
            let _e1409 = gsum;
            let _e1410 = total(_e1409);
            let _e1412 = n_read;
            let _e1413 = div(_e1410, _e1412);
            let _e1417 = sp.n_rows;
            let _e1420 = preds[(_e1323 * _e1417)];
            let _e1422 = keep((_e1420 + _e1413));
            let _e1428 = ghi;
            let _e1429 = gmin;
            if ((_e1428 - _e1429) <= (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e1422))))) {
                continue;
            }
            let _e1449 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e1459 = combinations[(gid.x % arrayLength((&combinations)))];
            let _e1463 = g_1;
            let _e1464 = linked_without((gid.x / arrayLength((&combinations))), (_e1449 & 16777215u), (_e1459 >> 24u), 0u, _e1463, _e1422);
            w_10 = 0u;
            loop {
                let _e1469 = w_10;
                if (_e1469 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e1474 = w_10;
                    m1_[_e1474] = vec2<f32>(0f, 0f);
                    let _e1478 = w_10;
                    let _e1479 = wrapped(_e1464, _e1478);
                    let _e1482 = w_10;
                    lo1_[_e1482] = _e1479;
                    let _e1487 = w_10;
                    let _e1489 = w_10;
                    let _e1491 = lo1_[_e1489];
                    hi1_[_e1487] = _e1491;
                }
                continuing {
                    let _e1494 = w_10;
                    w_10 = (_e1494 + 1u);
                }
            }
            row_5 = 0u;
            loop {
                let _e1500 = row_5;
                let _e1502 = sp.n_train;
                if (_e1500 < _e1502) {
                } else {
                    break;
                }
                {
                    let _e1517 = combinations[(gid.x % arrayLength((&combinations)))];
                    let _e1527 = combinations[(gid.x % arrayLength((&combinations)))];
                    let _e1530 = row_5;
                    let _e1532 = g_1;
                    let _e1533 = linked_without((gid.x / arrayLength((&combinations))), (_e1517 & 16777215u), (_e1527 >> 24u), _e1530, _e1532, _e1422);
                    w_11 = 0u;
                    loop {
                        let _e1538 = w_11;
                        if (_e1538 < N_WRAPPERS) {
                        } else {
                            break;
                        }
                        {
                            let _e1541 = w_11;
                            let _e1542 = wrapped(_e1533, _e1541);
                            let _e1545 = w_11;
                            let _e1547 = m1_[_e1545];
                            let _e1548 = add(_e1547, _e1542);
                            let _e1551 = w_11;
                            m1_[_e1551] = _e1548;
                            let _e1555 = w_11;
                            let _e1557 = w_11;
                            let _e1559 = lo1_[_e1557];
                            lo1_[_e1555] = min(_e1559, _e1542);
                            let _e1563 = w_11;
                            let _e1565 = w_11;
                            let _e1567 = hi1_[_e1565];
                            hi1_[_e1563] = max(_e1567, _e1542);
                        }
                        continuing {
                            let _e1571 = w_11;
                            w_11 = (_e1571 + 1u);
                        }
                    }
                }
                continuing {
                    let _e1575 = row_5;
                    row_5 = (_e1575 + LOO_STRIDE);
                }
            }
            w_12 = 0u;
            loop {
                let _e1581 = w_12;
                if (_e1581 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e1586 = w_12;
                    xx[_e1586] = vec2<f32>(0f, 0f);
                    let _e1592 = w_12;
                    xy[_e1592] = vec2<f32>(0f, 0f);
                }
                continuing {
                    let _e1597 = w_12;
                    w_12 = (_e1597 + 1u);
                }
            }
            row_6 = 0u;
            loop {
                let _e1603 = row_6;
                let _e1605 = sp.n_train;
                if (_e1603 < _e1605) {
                } else {
                    break;
                }
                {
                    let _e1620 = combinations[(gid.x % arrayLength((&combinations)))];
                    let _e1630 = combinations[(gid.x % arrayLength((&combinations)))];
                    let _e1633 = row_6;
                    let _e1635 = g_1;
                    let _e1636 = linked_without((gid.x / arrayLength((&combinations))), (_e1620 & 16777215u), (_e1630 >> 24u), _e1633, _e1635, _e1422);
                    let _e1640 = row_6;
                    let _e1642 = y_1[_e1640];
                    let _e1644 = sp.y_mean_train;
                    let _e1646 = keep((_e1642 - _e1644));
                    w_13 = 0u;
                    loop {
                        let _e1651 = w_13;
                        if (_e1651 < N_WRAPPERS) {
                        } else {
                            break;
                        }
                        {
                            let _e1654 = w_13;
                            let _e1655 = wrapped(_e1636, _e1654);
                            let _e1658 = w_13;
                            let _e1660 = m1_[_e1658];
                            let _e1661 = total(_e1660);
                            let _e1663 = n_read;
                            let _e1664 = div(_e1661, _e1663);
                            let _e1666 = keep((_e1655 - _e1664));
                            let _e1668 = keep((_e1666 * _e1666));
                            let _e1671 = w_13;
                            let _e1673 = xx[_e1671];
                            let _e1674 = add(_e1673, _e1668);
                            let _e1677 = w_13;
                            xx[_e1677] = _e1674;
                            let _e1680 = keep((_e1666 * _e1646));
                            let _e1683 = w_13;
                            let _e1685 = xy[_e1683];
                            let _e1686 = add(_e1685, _e1680);
                            let _e1689 = w_13;
                            xy[_e1689] = _e1686;
                        }
                        continuing {
                            let _e1693 = w_13;
                            w_13 = (_e1693 + 1u);
                        }
                    }
                }
                continuing {
                    let _e1697 = row_6;
                    row_6 = (_e1697 + LOO_STRIDE);
                }
            }
            w_14 = 0u;
            loop {
                let _e1703 = w_14;
                if (_e1703 < N_WRAPPERS) {
                } else {
                    break;
                }
                {
                    let _e1707 = w_14;
                    let _e1709 = ok[_e1707];
                    if !(_e1709) {
                        continue;
                    }
                    let _e1720 = sp.row_base;
                    let _e1733 = w_14;
                    let _e1738 = scores[(((((((_e1720 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e1733 * WIDTH)) + 2u)];
                    let _e1740 = sp.n_train;
                    let _e1743 = keep((_e1738 * f32(_e1740)));
                    let _e1746 = keep((_e1277 * 4f));
                    let _e1749 = div(_e1743, max(_e1746, 0.000000000000000000000000000001f));
                    let _e1752 = keep((1f - _e1749));
                    let _e1755 = w_14;
                    let _e1757 = xx[_e1755];
                    let _e1758 = total(_e1757);
                    let _e1761 = w_14;
                    let _e1763 = xy[_e1761];
                    let _e1764 = total(_e1763);
                    without = 0f;
                    let _e1769 = w_14;
                    let _e1771 = m1_[_e1769];
                    let _e1772 = total(_e1771);
                    let _e1774 = n_read;
                    let _e1775 = div(_e1772, _e1774);
                    let _e1776 = finite(_e1758);
                    let _e1777 = finite(_e1764);
                    let _e1785 = w_14;
                    let _e1787 = hi1_[_e1785];
                    let _e1788 = w_14;
                    let _e1790 = lo1_[_e1788];
                    if ((((((_e1787 - _e1790) > (2f * (0.00000001f + (CONSTANT_REL_TOL * abs(_e1775))))) && (_e1758 > 0f)) && (_e1277 > 0f)) && _e1776) && _e1777) {
                        let _e1804 = keep((_e1764 * _e1764));
                        let _e1806 = keep((_e1758 * _e1277));
                        let _e1807 = div(_e1804, _e1806);
                        without = clamp(_e1807, 0f, 1f);
                    }
                    l = 1f;
                    if (_e1752 > 0.000001f) {
                        let _e1817 = without;
                        let _e1818 = div(_e1817, _e1752);
                        let _e1821 = keep((1f - _e1818));
                        l = clamp(_e1821, 0f, 1f);
                    }
                    let _e1831 = w_14;
                    let _e1833 = w_14;
                    let _e1835 = n_loss[_e1833];
                    let _e1838 = l;
                    loss[((_e1831 * 8u) + _e1835)] = _e1838;
                    let _e1842 = w_14;
                    let _e1844 = w_14;
                    let _e1846 = n_loss[_e1844];
                    n_loss[_e1842] = (_e1846 + 1u);
                }
                continuing {
                    let _e1850 = w_14;
                    w_14 = (_e1850 + 1u);
                }
            }
        }
        continuing {
            let _e1854 = g_1;
            g_1 = (_e1854 + 1u);
        }
    }
    loop {
        let _e1858 = w_15;
        if (_e1858 < N_WRAPPERS) {
        } else {
            break;
        }
        {
            let _e1864 = w_15;
            let _e1866 = ok[_e1864];
            let _e1868 = w_15;
            let _e1870 = n_loss[_e1868];
            if (!(_e1866) || (_e1870 == 0u)) {
                continue;
            }
            i_3 = 1u;
            loop {
                let _e1878 = i_3;
                let _e1879 = w_15;
                let _e1881 = n_loss[_e1879];
                if (_e1878 < _e1881) {
                } else {
                    break;
                }
                {
                    let _e1885 = i_3;
                    j = _e1885;
                    loop {
                        let _e1893 = j;
                        let _e1895 = w_15;
                        let _e1897 = j;
                        let _e1901 = loss[(((_e1895 * 8u) + _e1897) - 1u)];
                        let _e1902 = w_15;
                        let _e1904 = i_3;
                        let _e1907 = loss[((_e1902 * 8u) + _e1904)];
                        if ((_e1893 == 0u) || (_e1901 <= _e1907)) {
                            break;
                        }
                        let _e1915 = w_15;
                        let _e1917 = j;
                        let _e1920 = w_15;
                        let _e1922 = j;
                        let _e1926 = loss[(((_e1920 * 8u) + _e1922) - 1u)];
                        loss[((_e1915 * 8u) + _e1917)] = _e1926;
                        let _e1929 = j;
                        j = (_e1929 - 1u);
                    }
                    let _e1936 = w_15;
                    let _e1938 = j;
                    let _e1941 = w_15;
                    let _e1943 = i_3;
                    let _e1946 = loss[((_e1941 * 8u) + _e1943)];
                    loss[((_e1936 * 8u) + _e1938)] = _e1946;
                }
                continuing {
                    let _e1949 = i_3;
                    i_3 = (_e1949 + 1u);
                }
            }
            let _e1957 = w_15;
            let _e1959 = w_15;
            let _e1961 = n_loss[_e1959];
            let _e1966 = loss[((_e1957 * 8u) + ((_e1961 - 1u) / 2u))];
            let _e1967 = w_15;
            let _e1969 = w_15;
            let _e1971 = n_loss[_e1969];
            let _e1975 = loss[((_e1967 * 8u) + (_e1971 / 2u))];
            let _e1977 = keep((_e1966 + _e1975));
            let _e1980 = keep((0.5f * _e1977));
            let _e1983 = keep((1f - _e1980));
            let _e1993 = sp.row_base;
            let _e2006 = w_15;
            scores[(((((((_e1993 + (gid.x / arrayLength((&combinations)))) * arrayLength((&combinations))) + (gid.x % arrayLength((&combinations)))) * N_WRAPPERS) * WIDTH) + (_e2006 * WIDTH)) + 9u)] = _e1983;
        }
        continuing {
            let _e2013 = w_15;
            w_15 = (_e2013 + 1u);
        }
    }
    return;
}
