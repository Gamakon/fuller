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
            let _e259 = l;
            let _e263 = compound_ops[((c_3 * 3u) + (_e259 - 1u))];
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
                let _e329 = tail_i;
                let _e331 = l;
                q_left[_e329] = (_e331 - 1u);
                let _e335 = tail_i;
                q_demand[_e335] = demand;
                let _e339 = tail_i;
                tail_i = (_e339 + 1u);
                let _e343 = next;
                next = (_e343 + 1u);
                let _e348 = next;
                a0_ = (_e348 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e357 = code[sid_1];
                op = _e357;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e364 = j;
                    if (_e364 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e370 = tail_i;
                        let _e373 = kid_first[pos];
                        let _e374 = j;
                        q_pos[_e370] = (_e373 + _e374);
                        let _e379 = tail_i;
                        q_left[_e379] = 0u;
                        let _e386 = tail_i;
                        let _e389 = params.k_in;
                        let _e391 = j;
                        let _e394 = in_ty[((sid_1 * _e389) + _e391)];
                        q_demand[_e386] = _e394;
                        let _e397 = tail_i;
                        tail_i = (_e397 + 1u);
                    }
                    continuing {
                        let _e401 = j;
                        j = (_e401 + 1u);
                    }
                }
                let _e404 = next;
                next = (_e404 + a_1);
                let _e408 = next;
                a0_ = (_e408 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e418 = code[sid_1];
                    a0_ = _e418;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e426 = konst[sid_1];
                        kv = bitcast<u32>(_e426);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e437 = i_1;
                                if (_e437 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e443 = i_1;
                                    let _e446 = genome[(base + _e443)];
                                    let _e448 = kind[_e446];
                                    if (_e448 == KIND_RNC) {
                                        let _e452 = idx;
                                        idx = (_e452 + 1u);
                                    }
                                }
                                continuing {
                                    let _e456 = i_1;
                                    i_1 = (_e456 + 1u);
                                }
                            }
                            let _e461 = idx;
                            let dc = genome[((base + ht) + _e461)];
                            let _e467 = params.n_rnc;
                            if (dc >= _e467) {
                                fail(gid_1);
                                return;
                            }
                            let _e473 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e473);
                            let _e477 = n_rnc_seen;
                            n_rnc_seen = (_e477 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e486 = code[sid_1];
                                a0_ = _e486;
                            } else {
                                op = OP_NUM;
                                let _e493 = code[sid_1];
                                a0_ = (_e493 + 1u);
                                let _e498 = konst[sid_1];
                                kv = bitcast<u32>(_e498);
                            }
                        }
                    }
                }
            }
        }
        let _e503 = count;
        let _e505 = op;
        node_op[_e503] = _e505;
        let _e509 = count;
        let _e511 = a0_;
        node_a0_[_e509] = _e511;
        let _e514 = count;
        node_tc[_e514] = ty_1;
        let _e519 = count;
        let _e521 = kv;
        node_k[_e519] = _e521;
        let _e525 = count;
        let _e527 = nk;
        node_nk[_e525] = _e527;
        count = def_d;
    }
    let _e531 = count;
    k2_ = _e531;
    loop {
        let _e534 = k2_;
        if (_e534 == 0u) {
            break;
        }
        let _e538 = k2_;
        k2_ = (_e538 - 1u);
        let _e542 = k2_;
        let op_5 = node_op[_e542];
        let _e547 = k2_;
        let nk_1 = node_nk[_e547];
        let def_b = (nk_1 == 0u);
        ty = TY_F;
        d = 0u;
        if def_b {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e563 = c;
                if (_e563 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e568 = k2_;
                    let _e570 = node_a0_[_e568];
                    let _e571 = c;
                    let kid = (_e570 + _e571);
                    let _e575 = m;
                    let _e577 = node_ty[kid];
                    m = max(_e575, _e577);
                    let _e581 = d;
                    let _e583 = node_depth[kid];
                    d = max(_e581, _e583);
                }
                continuing {
                    let _e587 = c;
                    c = (_e587 + 1u);
                }
            }
            let _e590 = m;
            let _e591 = apply(op_5, _e590);
            ty = _e591;
            let _e595 = ty;
            if (_e595 == TY_NONE) {
                ok = false;
            }
            let _e599 = is_raiser_op(op_5);
            if _e599 {
                let _e602 = d;
                d = (_e602 + 1u);
            }
        }
        let _e607 = k2_;
        let _e609 = ty;
        node_ty[_e607] = _e609;
        let _e613 = k2_;
        let _e615 = d;
        node_depth[_e613] = _e615;
        if def_b {
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
