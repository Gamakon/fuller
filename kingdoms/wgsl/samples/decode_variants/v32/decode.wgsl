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
        let _e140 = head_i;
        let _e141 = tail_i;
        if (_e140 >= _e141) {
            break;
        }
        let _e145 = head_i;
        let pos = q_pos[_e145];
        let _e150 = head_i;
        let left = q_left[_e150];
        let _e155 = head_i;
        let demand = q_demand[_e155];
        let _e160 = head_i;
        head_i = (_e160 + 1u);
        let _e164 = count;
        if (_e164 >= MAX_NODES) {
            fail(gid_1);
            return;
        }
        let sid_1 = genome[(base + pos)];
        let _e175 = params.typed_kingdom;
        let _e180 = out_ty[sid_1];
        if (((_e175 == 1u) && (left == 0u)) && (_e180 != demand)) {
            let _e186 = fallback_op[demand];
            if (_e186 == 4294967295u) {
                fail(gid_1);
                return;
            }
            let _e191 = count;
            let _e194 = fallback_op[demand];
            node_op[_e191] = _e194;
            let _e198 = count;
            node_a0_[_e198] = 0u;
            let _e202 = count;
            node_tc[_e202] = demand;
            let _e207 = count;
            let _e210 = fallback_k[demand];
            node_k[_e207] = _e210;
            let _e214 = count;
            node_nk[_e214] = 0u;
            let _e218 = count;
            count = (_e218 + 1u);
            let _e222 = masked;
            masked = (_e222 + 1u);
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
            let _e249 = l;
            if (_e249 == 0u) {
                l = len;
            }
            let _e254 = l;
            let def_f = (_e254 - 1u);
            let _e262 = compound_ops[((c_3 * 3u) + def_f)];
            op = _e262;
            let _e265 = l;
            if (_e265 == 1u) {
                let _e270 = tail_i;
                let _e273 = kid_first[pos];
                q_pos[_e270] = _e273;
                let _e277 = tail_i;
                q_left[_e277] = 0u;
                let _e281 = tail_i;
                q_demand[_e281] = demand;
                let _e285 = tail_i;
                tail_i = (_e285 + 1u);
                let _e291 = tail_i;
                let _e294 = kid_first[pos];
                q_pos[_e291] = (_e294 + 1u);
                let _e299 = tail_i;
                q_left[_e299] = 0u;
                let _e303 = tail_i;
                q_demand[_e303] = demand;
                let _e307 = tail_i;
                tail_i = (_e307 + 1u);
                let _e311 = next;
                next = (_e311 + 2u);
                let _e316 = next;
                a0_ = (_e316 - 2u);
                nk = 2u;
            } else {
                let _e322 = tail_i;
                q_pos[_e322] = pos;
                let _e326 = tail_i;
                q_left[_e326] = def_f;
                let _e330 = tail_i;
                q_demand[_e330] = demand;
                let _e334 = tail_i;
                tail_i = (_e334 + 1u);
                let _e338 = next;
                next = (_e338 + 1u);
                let _e343 = next;
                a0_ = (_e343 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e352 = code[sid_1];
                op = _e352;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e359 = j;
                    if (_e359 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e365 = tail_i;
                        let _e368 = kid_first[pos];
                        let _e369 = j;
                        q_pos[_e365] = (_e368 + _e369);
                        let _e374 = tail_i;
                        q_left[_e374] = 0u;
                        let _e381 = tail_i;
                        let _e384 = params.k_in;
                        let _e386 = j;
                        let _e389 = in_ty[((sid_1 * _e384) + _e386)];
                        q_demand[_e381] = _e389;
                        let _e392 = tail_i;
                        tail_i = (_e392 + 1u);
                    }
                    continuing {
                        let _e396 = j;
                        j = (_e396 + 1u);
                    }
                }
                let _e399 = next;
                next = (_e399 + a_1);
                let _e403 = next;
                a0_ = (_e403 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e413 = code[sid_1];
                    a0_ = _e413;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e421 = konst[sid_1];
                        kv = bitcast<u32>(_e421);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e432 = i_1;
                                if (_e432 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e438 = i_1;
                                    let _e441 = genome[(base + _e438)];
                                    let _e443 = kind[_e441];
                                    if (_e443 == KIND_RNC) {
                                        let _e447 = idx;
                                        idx = (_e447 + 1u);
                                    }
                                }
                                continuing {
                                    let _e451 = i_1;
                                    i_1 = (_e451 + 1u);
                                }
                            }
                            let _e456 = idx;
                            let dc = genome[((base + ht) + _e456)];
                            let _e462 = params.n_rnc;
                            if (dc >= _e462) {
                                fail(gid_1);
                                return;
                            }
                            let _e468 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e468);
                            let _e472 = n_rnc_seen;
                            n_rnc_seen = (_e472 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e481 = code[sid_1];
                                a0_ = _e481;
                            } else {
                                op = OP_NUM;
                                let _e488 = code[sid_1];
                                a0_ = (_e488 + 1u);
                                let _e493 = konst[sid_1];
                                kv = bitcast<u32>(_e493);
                            }
                        }
                    }
                }
            }
        }
        let _e498 = count;
        let _e500 = op;
        node_op[_e498] = _e500;
        let _e504 = count;
        let _e506 = a0_;
        node_a0_[_e504] = _e506;
        let _e509 = count;
        node_tc[_e509] = ty_1;
        let _e514 = count;
        let _e516 = kv;
        node_k[_e514] = _e516;
        let _e520 = count;
        let _e522 = nk;
        node_nk[_e520] = _e522;
        let _e525 = count;
        count = (_e525 + 1u);
    }
    let _e529 = count;
    k2_ = _e529;
    loop {
        let _e532 = k2_;
        if (_e532 == 0u) {
            break;
        }
        let _e536 = k2_;
        k2_ = (_e536 - 1u);
        let _e540 = k2_;
        let op_5 = node_op[_e540];
        let _e545 = k2_;
        let nk_1 = node_nk[_e545];
        ty = TY_F;
        d = 0u;
        if (nk_1 == 0u) {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e561 = c;
                if (_e561 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e566 = k2_;
                    let _e568 = node_a0_[_e566];
                    let _e569 = c;
                    let kid = (_e568 + _e569);
                    let _e573 = m;
                    let _e575 = node_ty[kid];
                    m = max(_e573, _e575);
                    let _e579 = d;
                    let _e581 = node_depth[kid];
                    d = max(_e579, _e581);
                }
                continuing {
                    let _e585 = c;
                    c = (_e585 + 1u);
                }
            }
            let _e588 = m;
            let _e589 = apply(op_5, _e588);
            ty = _e589;
            let _e593 = ty;
            if (_e593 == TY_NONE) {
                ok = false;
            }
            let _e597 = is_raiser_op(op_5);
            if _e597 {
                let _e600 = d;
                d = (_e600 + 1u);
            }
        }
        let _e605 = k2_;
        let _e607 = ty;
        node_ty[_e605] = _e607;
        let _e611 = k2_;
        let _e613 = d;
        node_depth[_e611] = _e613;
        if (nk_1 == 0u) {
            let _e619 = k2_;
            node_best[_e619] = 1u;
            let _e624 = k2_;
            node_ties[_e624] = 0u;
            let _e629 = k2_;
            node_order[_e629] = 1u;
        } else {
            let _e631 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e639 = c_1;
                if (_e639 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e644 = k2_;
                    let _e646 = node_a0_[_e644];
                    let _e647 = c_1;
                    let kid_1 = (_e646 + _e647);
                    let _e651 = node_op[kid_1];
                    let _e652 = family(_e651);
                    if ((_e631 != 0u) && (_e652 == _e631)) {
                        let _e660 = node_best[kid_1];
                        o = _e660;
                        let _e664 = node_ties[kid_1];
                        t = _e664;
                    } else {
                        let _e668 = node_order[kid_1];
                        o = _e668;
                        t = 1u;
                    }
                    let _e673 = o;
                    let _e674 = best;
                    if (_e673 > _e674) {
                        let _e678 = o;
                        best = _e678;
                        let _e681 = t;
                        ties = _e681;
                    } else {
                        let _e684 = o;
                        let _e685 = best;
                        if (_e684 == _e685) {
                            let _e689 = ties;
                            let _e690 = t;
                            ties = (_e689 + _e690);
                        }
                    }
                }
                continuing {
                    let _e694 = c_1;
                    c_1 = (_e694 + 1u);
                }
            }
            let _e699 = k2_;
            let _e701 = best;
            node_best[_e699] = _e701;
            let _e705 = k2_;
            let _e707 = ties;
            node_ties[_e705] = _e707;
            let _e715 = k2_;
            let _e717 = best;
            let _e718 = ties;
            node_order[_e715] = (_e717 + select(0u, 1u, (_e718 >= 2u)));
        }
    }
    let _e726 = params.typed;
    let _e728 = ok;
    if ((_e726 == 1u) && !(_e728)) {
        fail(gid_1);
        return;
    }
    let _e734 = params.order_ceiling;
    if (_e734 > 0u) {
        let _e740 = node_order[0];
        let _e742 = params.order_ceiling;
        if (_e740 > _e742) {
            fail(gid_1);
            return;
        }
        loop {
            let _e746 = k;
            let _e747 = count;
            if (_e746 < _e747) {
            } else {
                break;
            }
            {
                let _e751 = k;
                let op_6 = node_op[_e751];
                let _e757 = k;
                let _e759 = node_nk[_e757];
                if (_e759 == 0u) {
                    continue;
                }
                let _e762 = k;
                let _e763 = is_unary(_e762);
                let _e764 = is_transparent(op_6);
                if (_e763 && !(_e764)) {
                    let _e769 = k;
                    let _e771 = node_a0_[_e769];
                    let _e772 = through_transparent(_e771);
                    let _e773 = is_unary(_e772);
                    let _e779 = op_stack[op_6];
                    let _e781 = node_op[_e772];
                    if (_e773 && ((_e779 & (1u << _e781)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e789 = op_order[op_6];
                let lo = (_e789 & 255u);
                let _e794 = op_order[op_6];
                let hi = (_e794 >> 8u);
                c_2 = 0u;
                loop {
                    let _e801 = c_2;
                    let _e802 = k;
                    let _e804 = node_nk[_e802];
                    if (_e801 < _e804) {
                    } else {
                        break;
                    }
                    {
                        let _e810 = c_2;
                        if ((op_6 == OP_POW) && (_e810 == 1u)) {
                            continue;
                        }
                        let _e816 = k;
                        let _e818 = node_a0_[_e816];
                        let _e819 = c_2;
                        let kid_2 = (_e818 + _e819);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e829 = c_2;
                        c_2 = (_e829 + 1u);
                    }
                }
            }
            continuing {
                let _e833 = k;
                k = (_e833 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e842 = count;
    h = (2166136261u ^ _e842);
    loop {
        let _e846 = i_2;
        let _e847 = count;
        if (_e846 < _e847) {
        } else {
            break;
        }
        {
            let _e853 = i_2;
            let _e857 = i_2;
            let _e859 = node_op[_e857];
            nodes[(out + (_e853 * 4u))] = _e859;
            let _e865 = i_2;
            let _e870 = i_2;
            let _e872 = node_a0_[_e870];
            nodes[((out + (_e865 * 4u)) + 1u)] = _e872;
            let _e878 = i_2;
            let _e883 = i_2;
            let _e885 = node_tc[_e883];
            nodes[((out + (_e878 * 4u)) + 2u)] = _e885;
            let _e891 = i_2;
            let _e896 = i_2;
            let _e898 = node_k[_e896];
            nodes[((out + (_e891 * 4u)) + 3u)] = _e898;
            let _e903 = h;
            let _e904 = i_2;
            let _e906 = node_op[_e904];
            h = ((_e903 ^ _e906) * 16777619u);
            let _e913 = h;
            let _e914 = i_2;
            let _e916 = node_a0_[_e914];
            h = ((_e913 ^ _e916) * 16777619u);
            let _e923 = h;
            let _e924 = i_2;
            let _e926 = node_tc[_e924];
            h = ((_e923 ^ _e926) * 16777619u);
            let _e933 = h;
            let _e934 = i_2;
            let _e936 = node_k[_e934];
            h = ((_e933 ^ _e936) * 16777619u);
        }
        continuing {
            let _e941 = i_2;
            i_2 = (_e941 + 1u);
        }
    }
    let _e946 = count;
    lengths[gid_1] = _e946;
    gene_ok[gid_1] = 1u;
    let _e955 = node_depth[0];
    gene_depth[gid_1] = _e955;
    let _e959 = h;
    gene_hash[gid_1] = _e959;
    return;
}
