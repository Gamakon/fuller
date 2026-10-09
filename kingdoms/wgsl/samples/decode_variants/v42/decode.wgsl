struct Params {
    pop: u32,
    n_genes: u32,
    head: u32,
    tail: u32,
    n_rnc: u32,
    typed: u32,
    n_ids: u32,
    order_ceiling: u32,
    typed_kingdom: u32,
    root_ty: u32,
    k_in: u32,
    pad: u32,
}

const KIND_FUNCTION: u32 = 0u;
const KIND_COMPOUND: u32 = 1u;
const KIND_INPUT: u32 = 2u;
const KIND_CONSTANT: u32 = 3u;
const KIND_RNC: u32 = 4u;
const KIND_NAMED: u32 = 5u;
const KIND_GENEREF: u32 = 6u;
const OP_VAR: u32 = 0u;
const OP_NUM: u32 = 1u;
const OP_GENEREF: u32 = 47u;
const MAX_NODES: u32 = 64u;
const MAX_HT: u32 = 512u;
const TY_F: u32 = 0u;
const TY_T1_: u32 = 1u;
const TY_T2_: u32 = 2u;
const TY_NONE: u32 = 3u;
const OP_ADD: u32 = 2u;
const OP_SUB: u32 = 3u;
const OP_MUL: u32 = 4u;
const OP_DIV: u32 = 5u;
const OP_NEG: u32 = 6u;
const OP_ABS: u32 = 7u;
const OP_POW: u32 = 15u;
const OP_PDIV: u32 = 19u;

@group(0) @binding(0) 
var<uniform> params: Params;
@group(0) @binding(1) 
var<storage> genome: array<u32>;
@group(0) @binding(2) 
var<storage> rnc: array<f32>;
@group(0) @binding(3) 
var<storage> kind: array<u32>;
@group(0) @binding(4) 
var<storage> code: array<u32>;
@group(0) @binding(5) 
var<storage> konst: array<f32>;
@group(0) @binding(6) 
var<storage> arity: array<u32>;
@group(0) @binding(7) 
var<storage> op_raiser: array<u32>;
@group(0) @binding(8) 
var<storage> compound_ops: array<u32>;
@group(0) @binding(9) 
var<storage> compound_len: array<u32>;
@group(0) @binding(10) 
var<storage, read_write> nodes: array<u32>;
@group(0) @binding(11) 
var<storage, read_write> offsets: array<u32>;
@group(0) @binding(12) 
var<storage, read_write> lengths: array<u32>;
@group(0) @binding(13) 
var<storage, read_write> gene_ok: array<u32>;
@group(0) @binding(14) 
var<storage, read_write> gene_depth: array<u32>;
@group(0) @binding(15) 
var<storage, read_write> gene_hash: array<u32>;
@group(0) @binding(16) 
var<storage> op_order: array<u32>;
@group(0) @binding(17) 
var<storage> op_stack: array<u32>;
@group(0) @binding(18) 
var<storage> out_ty: array<u32>;
@group(0) @binding(19) 
var<storage> in_ty: array<u32>;
@group(0) @binding(20) 
var<storage> fallback_op: array<u32>;
@group(0) @binding(21) 
var<storage> fallback_k: array<u32>;
var<private> kid_first: array<u32, 512>;
var<private> node_op: array<u32, 64>;
var<private> node_a0_: array<u32, 64>;
var<private> node_tc: array<u32, 64>;
var<private> node_k: array<u32, 64>;
var<private> node_nk: array<u32, 64>;
var<private> node_ty: array<u32, 64>;
var<private> node_depth: array<u32, 64>;
var<private> node_best: array<u32, 64>;
var<private> node_ties: array<u32, 64>;
var<private> node_order: array<u32, 64>;
var<private> q_pos: array<u32, 512>;
var<private> q_left: array<u32, 512>;
var<private> q_demand: array<u32, 512>;

fn is_raiser_op(op_1: u32) -> bool {
    let _e4 = op_raiser[op_1];
    return (_e4 == 1u);
}

fn family(op_2: u32) -> u32 {
    if ((op_2 == OP_ADD) || (op_2 == OP_SUB)) {
        return 1u;
    }
    if (((op_2 == OP_MUL) || (op_2 == OP_DIV)) || (op_2 == OP_PDIV)) {
        return 2u;
    }
    return 0u;
}

fn is_transparent(op_3: u32) -> bool {
    return ((op_3 == OP_NEG) || (op_3 == OP_ABS));
}

fn is_unary(k_1: u32) -> bool {
    let _e4 = node_nk[k_1];
    return (_e4 == 1u);
}

fn through_transparent(k0_: u32) -> u32 {
    var k_2: u32;
    var i_3: u32 = 0u;

    k_2 = k0_;
    loop {
        let _e5 = i_3;
        if (_e5 < MAX_NODES) {
        } else {
            break;
        }
        {
            let _e8 = k_2;
            let _e9 = is_unary(_e8);
            let _e12 = k_2;
            let _e14 = node_op[_e12];
            let _e15 = is_transparent(_e14);
            if !((_e9 && _e15)) {
                break;
            }
            let _e20 = k_2;
            let _e22 = node_a0_[_e20];
            k_2 = _e22;
        }
        continuing {
            let _e25 = i_3;
            i_3 = (_e25 + 1u);
        }
    }
    let _e28 = k_2;
    return _e28;
}

fn apply(op_4: u32, m_1: u32) -> u32 {
    let _e1 = is_raiser_op(op_4);
    if _e1 {
        if (m_1 >= TY_T2_) {
            return TY_NONE;
        }
        return (m_1 + 1u);
    }
    return m_1;
}

fn fail(gid: u32) {
    lengths[gid] = 0u;
    gene_ok[gid] = 0u;
    gene_depth[gid] = 0u;
    gene_hash[gid] = 0u;
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn decode_main(@builtin(global_invocation_id) id: vec3<u32>) {
    var need: i32 = 1i;
    var n: u32 = 0u;
    var child: u32 = 1u;
    var n_q: u32 = 0u;
    var i: u32 = 0u;
    var head_i: u32 = 0u;
    var tail_i: u32 = 1u;
    var count: u32 = 0u;
    var next: u32 = 1u;
    var n_rnc_seen: u32 = 0u;
    var masked: u32 = 0u;
    var op: u32;
    var a0_: u32;
    var kv: u32;
    var nk: u32;
    var l: u32;
    var j: u32;
    var idx: u32;
    var i_1: u32;
    var k2_: u32;
    var ok: bool = true;
    var ty: u32;
    var d: u32;
    var m: u32;
    var c: u32;
    var best: u32;
    var ties: u32;
    var c_1: u32;
    var o: u32;
    var t: u32;
    var k: u32 = 0u;
    var c_2: u32;
    var h: u32;
    var i_2: u32 = 0u;

    let gid_1 = id.x;
    let _e18 = params.pop;
    let _e20 = params.n_genes;
    if (gid_1 >= (_e18 * _e20)) {
        return;
    }
    let _e25 = params.head;
    let _e27 = params.tail;
    let ht = (_e25 + _e27);
    let _e32 = params.head;
    let _e34 = params.tail;
    let width = (_e32 + (2u * _e34));
    let base = (gid_1 * width);
    let _e40 = params.n_rnc;
    let rbase = (gid_1 * _e40);
    offsets[gid_1] = (gid_1 * MAX_NODES);
    loop {
        let _e49 = need;
        let _e51 = n;
        if ((_e49 <= 0i) || (_e51 >= ht)) {
            break;
        }
        let _e59 = need;
        let _e60 = n;
        let _e63 = genome[(base + _e60)];
        let _e65 = arity[_e63];
        need = ((_e59 + i32(_e65)) - 1i);
        let _e71 = n;
        n = (_e71 + 1u);
    }
    let _e76 = need;
    if ((_e76 > 0i) || (ht > MAX_HT)) {
        fail(gid_1);
        return;
    }
    loop {
        let _e82 = i;
        let _e83 = n;
        if (_e82 < _e83) {
        } else {
            break;
        }
        {
            let _e87 = i;
            let sid = genome[(base + _e87)];
            let a = arity[sid];
            let _e99 = i;
            let _e101 = child;
            kid_first[_e99] = select(0u, _e101, (a >= 1u));
            let _e105 = child;
            child = (_e105 + a);
            let _e110 = kind[sid];
            if (_e110 == KIND_RNC) {
                let _e114 = n_q;
                n_q = (_e114 + 1u);
            }
        }
        continuing {
            let _e118 = i;
            i = (_e118 + 1u);
        }
    }
    let _e122 = n_q;
    let _e124 = params.n_rnc;
    if (_e122 > _e124) {
        fail(gid_1);
        return;
    }
    q_pos[0] = 0u;
    q_left[0] = 0u;
    let _e137 = params.root_ty;
    q_demand[0] = _e137;
    loop {
        let _e140 = count;
        let def_d = (_e140 + 1u);
        let _e144 = head_i;
        let _e145 = tail_i;
        if (_e144 >= _e145) {
            break;
        }
        let _e149 = head_i;
        let pos = q_pos[_e149];
        let _e154 = head_i;
        let left = q_left[_e154];
        let _e159 = head_i;
        let demand = q_demand[_e159];
        let _e164 = head_i;
        head_i = (_e164 + 1u);
        let _e168 = count;
        if (_e168 >= MAX_NODES) {
            fail(gid_1);
            return;
        }
        let sid_1 = genome[(base + pos)];
        let _e179 = params.typed_kingdom;
        let _e184 = out_ty[sid_1];
        if (((_e179 == 1u) && (left == 0u)) && (_e184 != demand)) {
            let _e190 = fallback_op[demand];
            if (_e190 == 4294967295u) {
                fail(gid_1);
                return;
            }
            let _e195 = count;
            let _e198 = fallback_op[demand];
            node_op[_e195] = _e198;
            let _e202 = count;
            node_a0_[_e202] = 0u;
            let _e206 = count;
            node_tc[_e206] = demand;
            let _e211 = count;
            let _e214 = fallback_k[demand];
            node_k[_e211] = _e214;
            let _e218 = count;
            node_nk[_e218] = 0u;
            count = def_d;
            let _e223 = masked;
            masked = (_e223 + 1u);
            continue;
        }
        let k_3 = kind[sid_1];
        op = 0u;
        a0_ = 0u;
        let ty_1 = out_ty[sid_1];
        kv = 0u;
        nk = 0u;
        if (k_3 == KIND_COMPOUND) {
            let c_3 = code[sid_1];
            let len = compound_len[c_3];
            l = left;
            let _e250 = l;
            if (_e250 == 0u) {
                l = len;
            }
            let _e255 = l;
            let def_f = (_e255 - 1u);
            let _e263 = compound_ops[((c_3 * 3u) + def_f)];
            op = _e263;
            let _e266 = l;
            if (_e266 == 1u) {
                let _e271 = tail_i;
                let _e274 = kid_first[pos];
                q_pos[_e271] = _e274;
                let _e278 = tail_i;
                q_left[_e278] = 0u;
                let _e282 = tail_i;
                q_demand[_e282] = demand;
                let _e286 = tail_i;
                tail_i = (_e286 + 1u);
                let _e292 = tail_i;
                let _e295 = kid_first[pos];
                q_pos[_e292] = (_e295 + 1u);
                let _e300 = tail_i;
                q_left[_e300] = 0u;
                let _e304 = tail_i;
                q_demand[_e304] = demand;
                let _e308 = tail_i;
                tail_i = (_e308 + 1u);
                let _e312 = next;
                next = (_e312 + 2u);
                let _e317 = next;
                a0_ = (_e317 - 2u);
                nk = 2u;
            } else {
                let _e323 = tail_i;
                q_pos[_e323] = pos;
                let _e327 = tail_i;
                q_left[_e327] = def_f;
                let _e331 = tail_i;
                q_demand[_e331] = demand;
                let _e335 = tail_i;
                tail_i = (_e335 + 1u);
                let _e339 = next;
                next = (_e339 + 1u);
                let _e344 = next;
                a0_ = (_e344 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e353 = code[sid_1];
                op = _e353;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e360 = j;
                    if (_e360 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e366 = tail_i;
                        let _e369 = kid_first[pos];
                        let _e370 = j;
                        q_pos[_e366] = (_e369 + _e370);
                        let _e375 = tail_i;
                        q_left[_e375] = 0u;
                        let _e382 = tail_i;
                        let _e385 = params.k_in;
                        let _e387 = j;
                        let _e390 = in_ty[((sid_1 * _e385) + _e387)];
                        q_demand[_e382] = _e390;
                        let _e393 = tail_i;
                        tail_i = (_e393 + 1u);
                    }
                    continuing {
                        let _e397 = j;
                        j = (_e397 + 1u);
                    }
                }
                let _e400 = next;
                next = (_e400 + a_1);
                let _e404 = next;
                a0_ = (_e404 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e414 = code[sid_1];
                    a0_ = _e414;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e422 = konst[sid_1];
                        kv = bitcast<u32>(_e422);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e433 = i_1;
                                if (_e433 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e439 = i_1;
                                    let _e442 = genome[(base + _e439)];
                                    let _e444 = kind[_e442];
                                    if (_e444 == KIND_RNC) {
                                        let _e448 = idx;
                                        idx = (_e448 + 1u);
                                    }
                                }
                                continuing {
                                    let _e452 = i_1;
                                    i_1 = (_e452 + 1u);
                                }
                            }
                            let _e457 = idx;
                            let dc = genome[((base + ht) + _e457)];
                            let _e463 = params.n_rnc;
                            if (dc >= _e463) {
                                fail(gid_1);
                                return;
                            }
                            let _e469 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e469);
                            let _e473 = n_rnc_seen;
                            n_rnc_seen = (_e473 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e482 = code[sid_1];
                                a0_ = _e482;
                            } else {
                                op = OP_NUM;
                                let _e489 = code[sid_1];
                                a0_ = (_e489 + 1u);
                                let _e494 = konst[sid_1];
                                kv = bitcast<u32>(_e494);
                            }
                        }
                    }
                }
            }
        }
        let _e499 = count;
        let _e501 = op;
        node_op[_e499] = _e501;
        let _e505 = count;
        let _e507 = a0_;
        node_a0_[_e505] = _e507;
        let _e510 = count;
        node_tc[_e510] = ty_1;
        let _e515 = count;
        let _e517 = kv;
        node_k[_e515] = _e517;
        let _e521 = count;
        let _e523 = nk;
        node_nk[_e521] = _e523;
        count = def_d;
    }
    let _e527 = count;
    k2_ = _e527;
    loop {
        let _e530 = k2_;
        if (_e530 == 0u) {
            break;
        }
        let _e534 = k2_;
        k2_ = (_e534 - 1u);
        let _e538 = k2_;
        let op_5 = node_op[_e538];
        let _e543 = k2_;
        let nk_1 = node_nk[_e543];
        let def_b = (nk_1 == 0u);
        ty = TY_F;
        d = 0u;
        if def_b {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e559 = c;
                if (_e559 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e564 = k2_;
                    let _e566 = node_a0_[_e564];
                    let _e567 = c;
                    let kid = (_e566 + _e567);
                    let _e571 = m;
                    let _e573 = node_ty[kid];
                    m = max(_e571, _e573);
                    let _e577 = d;
                    let _e579 = node_depth[kid];
                    d = max(_e577, _e579);
                }
                continuing {
                    let _e583 = c;
                    c = (_e583 + 1u);
                }
            }
            let _e586 = m;
            let _e587 = apply(op_5, _e586);
            ty = _e587;
            let _e591 = ty;
            if (_e591 == TY_NONE) {
                ok = false;
            }
            let _e595 = is_raiser_op(op_5);
            if _e595 {
                let _e598 = d;
                d = (_e598 + 1u);
            }
        }
        let _e603 = k2_;
        let _e605 = ty;
        node_ty[_e603] = _e605;
        let _e609 = k2_;
        let _e611 = d;
        node_depth[_e609] = _e611;
        if def_b {
            let _e615 = k2_;
            node_best[_e615] = 1u;
            let _e620 = k2_;
            node_ties[_e620] = 0u;
            let _e625 = k2_;
            node_order[_e625] = 1u;
        } else {
            let _e627 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e635 = c_1;
                if (_e635 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e640 = k2_;
                    let _e642 = node_a0_[_e640];
                    let _e643 = c_1;
                    let kid_1 = (_e642 + _e643);
                    let _e647 = node_op[kid_1];
                    let _e648 = family(_e647);
                    if ((_e627 != 0u) && (_e648 == _e627)) {
                        let _e656 = node_best[kid_1];
                        o = _e656;
                        let _e660 = node_ties[kid_1];
                        t = _e660;
                    } else {
                        let _e664 = node_order[kid_1];
                        o = _e664;
                        t = 1u;
                    }
                    let _e669 = o;
                    let _e670 = best;
                    if (_e669 > _e670) {
                        let _e674 = o;
                        best = _e674;
                        let _e677 = t;
                        ties = _e677;
                    } else {
                        let _e680 = o;
                        let _e681 = best;
                        if (_e680 == _e681) {
                            let _e685 = ties;
                            let _e686 = t;
                            ties = (_e685 + _e686);
                        }
                    }
                }
                continuing {
                    let _e690 = c_1;
                    c_1 = (_e690 + 1u);
                }
            }
            let _e695 = k2_;
            let _e697 = best;
            node_best[_e695] = _e697;
            let _e701 = k2_;
            let _e703 = ties;
            node_ties[_e701] = _e703;
            let _e711 = k2_;
            let _e713 = best;
            let _e714 = ties;
            node_order[_e711] = (_e713 + select(0u, 1u, (_e714 >= 2u)));
        }
    }
    let _e722 = params.typed;
    let _e724 = ok;
    if ((_e722 == 1u) && !(_e724)) {
        fail(gid_1);
        return;
    }
    let _e730 = params.order_ceiling;
    if (_e730 > 0u) {
        let _e736 = node_order[0];
        let _e738 = params.order_ceiling;
        if (_e736 > _e738) {
            fail(gid_1);
            return;
        }
        loop {
            let _e742 = k;
            let _e743 = count;
            if (_e742 < _e743) {
            } else {
                break;
            }
            {
                let _e747 = k;
                let op_6 = node_op[_e747];
                let _e753 = k;
                let _e755 = node_nk[_e753];
                if (_e755 == 0u) {
                    continue;
                }
                let _e758 = k;
                let _e759 = is_unary(_e758);
                let _e760 = is_transparent(op_6);
                if (_e759 && !(_e760)) {
                    let _e765 = k;
                    let _e767 = node_a0_[_e765];
                    let _e768 = through_transparent(_e767);
                    let _e769 = is_unary(_e768);
                    let _e775 = op_stack[op_6];
                    let _e777 = node_op[_e768];
                    if (_e769 && ((_e775 & (1u << _e777)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e785 = op_order[op_6];
                let lo = (_e785 & 255u);
                let _e790 = op_order[op_6];
                let hi = (_e790 >> 8u);
                c_2 = 0u;
                loop {
                    let _e797 = c_2;
                    let _e798 = k;
                    let _e800 = node_nk[_e798];
                    if (_e797 < _e800) {
                    } else {
                        break;
                    }
                    {
                        let _e806 = c_2;
                        if ((op_6 == OP_POW) && (_e806 == 1u)) {
                            continue;
                        }
                        let _e812 = k;
                        let _e814 = node_a0_[_e812];
                        let _e815 = c_2;
                        let kid_2 = (_e814 + _e815);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e825 = c_2;
                        c_2 = (_e825 + 1u);
                    }
                }
            }
            continuing {
                let _e829 = k;
                k = (_e829 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e838 = count;
    h = (2166136261u ^ _e838);
    loop {
        let _e842 = i_2;
        let _e843 = count;
        if (_e842 < _e843) {
        } else {
            break;
        }
        {
            let _e849 = i_2;
            let _e853 = i_2;
            let _e855 = node_op[_e853];
            nodes[(out + (_e849 * 4u))] = _e855;
            let _e861 = i_2;
            let _e866 = i_2;
            let _e868 = node_a0_[_e866];
            nodes[((out + (_e861 * 4u)) + 1u)] = _e868;
            let _e874 = i_2;
            let _e879 = i_2;
            let _e881 = node_tc[_e879];
            nodes[((out + (_e874 * 4u)) + 2u)] = _e881;
            let _e887 = i_2;
            let _e892 = i_2;
            let _e894 = node_k[_e892];
            nodes[((out + (_e887 * 4u)) + 3u)] = _e894;
            let _e899 = h;
            let _e900 = i_2;
            let _e902 = node_op[_e900];
            h = ((_e899 ^ _e902) * 16777619u);
            let _e909 = h;
            let _e910 = i_2;
            let _e912 = node_a0_[_e910];
            h = ((_e909 ^ _e912) * 16777619u);
            let _e919 = h;
            let _e920 = i_2;
            let _e922 = node_tc[_e920];
            h = ((_e919 ^ _e922) * 16777619u);
            let _e929 = h;
            let _e930 = i_2;
            let _e932 = node_k[_e930];
            h = ((_e929 ^ _e932) * 16777619u);
        }
        continuing {
            let _e937 = i_2;
            i_2 = (_e937 + 1u);
        }
    }
    let _e942 = count;
    lengths[gid_1] = _e942;
    gene_ok[gid_1] = 1u;
    let _e951 = node_depth[0];
    gene_depth[gid_1] = _e951;
    let _e955 = h;
    gene_hash[gid_1] = _e955;
    return;
}
