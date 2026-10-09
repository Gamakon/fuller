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
        let _e140 = tail_i;
        let def_e = (_e140 + 1u);
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
            let _e222 = count;
            count = (_e222 + 1u);
            let _e226 = masked;
            masked = (_e226 + 1u);
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
            let _e253 = l;
            if (_e253 == 0u) {
                l = len;
            }
            let _e262 = l;
            let _e266 = compound_ops[((c_3 * 3u) + (_e262 - 1u))];
            op = _e266;
            let _e269 = l;
            if (_e269 == 1u) {
                let _e274 = tail_i;
                let _e277 = kid_first[pos];
                q_pos[_e274] = _e277;
                let _e281 = tail_i;
                q_left[_e281] = 0u;
                let _e285 = tail_i;
                q_demand[_e285] = demand;
                tail_i = def_e;
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
                tail_i = def_e;
                let _e340 = next;
                next = (_e340 + 1u);
                let _e345 = next;
                a0_ = (_e345 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e354 = code[sid_1];
                op = _e354;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e361 = j;
                    if (_e361 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e367 = tail_i;
                        let _e370 = kid_first[pos];
                        let _e371 = j;
                        q_pos[_e367] = (_e370 + _e371);
                        let _e376 = tail_i;
                        q_left[_e376] = 0u;
                        let _e383 = tail_i;
                        let _e386 = params.k_in;
                        let _e388 = j;
                        let _e391 = in_ty[((sid_1 * _e386) + _e388)];
                        q_demand[_e383] = _e391;
                        let _e394 = tail_i;
                        tail_i = (_e394 + 1u);
                    }
                    continuing {
                        let _e398 = j;
                        j = (_e398 + 1u);
                    }
                }
                let _e401 = next;
                next = (_e401 + a_1);
                let _e405 = next;
                a0_ = (_e405 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e415 = code[sid_1];
                    a0_ = _e415;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e423 = konst[sid_1];
                        kv = bitcast<u32>(_e423);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e434 = i_1;
                                if (_e434 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e440 = i_1;
                                    let _e443 = genome[(base + _e440)];
                                    let _e445 = kind[_e443];
                                    if (_e445 == KIND_RNC) {
                                        let _e449 = idx;
                                        idx = (_e449 + 1u);
                                    }
                                }
                                continuing {
                                    let _e453 = i_1;
                                    i_1 = (_e453 + 1u);
                                }
                            }
                            let _e458 = idx;
                            let dc = genome[((base + ht) + _e458)];
                            let _e464 = params.n_rnc;
                            if (dc >= _e464) {
                                fail(gid_1);
                                return;
                            }
                            let _e470 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e470);
                            let _e474 = n_rnc_seen;
                            n_rnc_seen = (_e474 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e483 = code[sid_1];
                                a0_ = _e483;
                            } else {
                                op = OP_NUM;
                                let _e490 = code[sid_1];
                                a0_ = (_e490 + 1u);
                                let _e495 = konst[sid_1];
                                kv = bitcast<u32>(_e495);
                            }
                        }
                    }
                }
            }
        }
        let _e500 = count;
        let _e502 = op;
        node_op[_e500] = _e502;
        let _e506 = count;
        let _e508 = a0_;
        node_a0_[_e506] = _e508;
        let _e511 = count;
        node_tc[_e511] = ty_1;
        let _e516 = count;
        let _e518 = kv;
        node_k[_e516] = _e518;
        let _e522 = count;
        let _e524 = nk;
        node_nk[_e522] = _e524;
        let _e527 = count;
        count = (_e527 + 1u);
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
        ty = TY_F;
        d = 0u;
        if (nk_1 == 0u) {
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
        if (nk_1 == 0u) {
            let _e621 = k2_;
            node_best[_e621] = 1u;
            let _e626 = k2_;
            node_ties[_e626] = 0u;
            let _e631 = k2_;
            node_order[_e631] = 1u;
        } else {
            let _e633 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e641 = c_1;
                if (_e641 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e646 = k2_;
                    let _e648 = node_a0_[_e646];
                    let _e649 = c_1;
                    let kid_1 = (_e648 + _e649);
                    let _e653 = node_op[kid_1];
                    let _e654 = family(_e653);
                    if ((_e633 != 0u) && (_e654 == _e633)) {
                        let _e662 = node_best[kid_1];
                        o = _e662;
                        let _e666 = node_ties[kid_1];
                        t = _e666;
                    } else {
                        let _e670 = node_order[kid_1];
                        o = _e670;
                        t = 1u;
                    }
                    let _e675 = o;
                    let _e676 = best;
                    if (_e675 > _e676) {
                        let _e680 = o;
                        best = _e680;
                        let _e683 = t;
                        ties = _e683;
                    } else {
                        let _e686 = o;
                        let _e687 = best;
                        if (_e686 == _e687) {
                            let _e691 = ties;
                            let _e692 = t;
                            ties = (_e691 + _e692);
                        }
                    }
                }
                continuing {
                    let _e696 = c_1;
                    c_1 = (_e696 + 1u);
                }
            }
            let _e701 = k2_;
            let _e703 = best;
            node_best[_e701] = _e703;
            let _e707 = k2_;
            let _e709 = ties;
            node_ties[_e707] = _e709;
            let _e717 = k2_;
            let _e719 = best;
            let _e720 = ties;
            node_order[_e717] = (_e719 + select(0u, 1u, (_e720 >= 2u)));
        }
    }
    let _e728 = params.typed;
    let _e730 = ok;
    if ((_e728 == 1u) && !(_e730)) {
        fail(gid_1);
        return;
    }
    let _e736 = params.order_ceiling;
    if (_e736 > 0u) {
        let _e742 = node_order[0];
        let _e744 = params.order_ceiling;
        if (_e742 > _e744) {
            fail(gid_1);
            return;
        }
        loop {
            let _e748 = k;
            let _e749 = count;
            if (_e748 < _e749) {
            } else {
                break;
            }
            {
                let _e753 = k;
                let op_6 = node_op[_e753];
                let _e759 = k;
                let _e761 = node_nk[_e759];
                if (_e761 == 0u) {
                    continue;
                }
                let _e764 = k;
                let _e765 = is_unary(_e764);
                let _e766 = is_transparent(op_6);
                if (_e765 && !(_e766)) {
                    let _e771 = k;
                    let _e773 = node_a0_[_e771];
                    let _e774 = through_transparent(_e773);
                    let _e775 = is_unary(_e774);
                    let _e781 = op_stack[op_6];
                    let _e783 = node_op[_e774];
                    if (_e775 && ((_e781 & (1u << _e783)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e791 = op_order[op_6];
                let lo = (_e791 & 255u);
                let _e796 = op_order[op_6];
                let hi = (_e796 >> 8u);
                c_2 = 0u;
                loop {
                    let _e803 = c_2;
                    let _e804 = k;
                    let _e806 = node_nk[_e804];
                    if (_e803 < _e806) {
                    } else {
                        break;
                    }
                    {
                        let _e812 = c_2;
                        if ((op_6 == OP_POW) && (_e812 == 1u)) {
                            continue;
                        }
                        let _e818 = k;
                        let _e820 = node_a0_[_e818];
                        let _e821 = c_2;
                        let kid_2 = (_e820 + _e821);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e831 = c_2;
                        c_2 = (_e831 + 1u);
                    }
                }
            }
            continuing {
                let _e835 = k;
                k = (_e835 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e844 = count;
    h = (2166136261u ^ _e844);
    loop {
        let _e848 = i_2;
        let def_a = (out + (_e848 * 4u));
        let _e853 = i_2;
        let _e854 = count;
        if (_e853 < _e854) {
        } else {
            break;
        }
        {
            let _e860 = i_2;
            let _e862 = node_op[_e860];
            nodes[def_a] = _e862;
            let _e869 = i_2;
            let _e871 = node_a0_[_e869];
            nodes[(def_a + 1u)] = _e871;
            let _e878 = i_2;
            let _e880 = node_tc[_e878];
            nodes[(def_a + 2u)] = _e880;
            let _e887 = i_2;
            let _e889 = node_k[_e887];
            nodes[(def_a + 3u)] = _e889;
            let _e894 = h;
            let _e895 = i_2;
            let _e897 = node_op[_e895];
            h = ((_e894 ^ _e897) * 16777619u);
            let _e904 = h;
            let _e905 = i_2;
            let _e907 = node_a0_[_e905];
            h = ((_e904 ^ _e907) * 16777619u);
            let _e914 = h;
            let _e915 = i_2;
            let _e917 = node_tc[_e915];
            h = ((_e914 ^ _e917) * 16777619u);
            let _e924 = h;
            let _e925 = i_2;
            let _e927 = node_k[_e925];
            h = ((_e924 ^ _e927) * 16777619u);
        }
        continuing {
            let _e932 = i_2;
            i_2 = (_e932 + 1u);
        }
    }
    let _e937 = count;
    lengths[gid_1] = _e937;
    gene_ok[gid_1] = 1u;
    let _e946 = node_depth[0];
    gene_depth[gid_1] = _e946;
    let _e950 = h;
    gene_hash[gid_1] = _e950;
    return;
}
