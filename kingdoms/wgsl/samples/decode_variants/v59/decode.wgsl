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
        let _e144 = count;
        let def_d = (_e144 + 1u);
        let _e148 = head_i;
        let _e149 = tail_i;
        if (_e148 >= _e149) {
            break;
        }
        let _e153 = head_i;
        let pos = q_pos[_e153];
        let _e158 = head_i;
        let left = q_left[_e158];
        let _e163 = head_i;
        let demand = q_demand[_e163];
        let _e168 = head_i;
        head_i = (_e168 + 1u);
        let _e172 = count;
        if (_e172 >= MAX_NODES) {
            fail(gid_1);
            return;
        }
        let sid_1 = genome[(base + pos)];
        let _e183 = params.typed_kingdom;
        let _e188 = out_ty[sid_1];
        if (((_e183 == 1u) && (left == 0u)) && (_e188 != demand)) {
            let _e194 = fallback_op[demand];
            if (_e194 == 4294967295u) {
                fail(gid_1);
                return;
            }
            let _e199 = count;
            let _e202 = fallback_op[demand];
            node_op[_e199] = _e202;
            let _e206 = count;
            node_a0_[_e206] = 0u;
            let _e210 = count;
            node_tc[_e210] = demand;
            let _e215 = count;
            let _e218 = fallback_k[demand];
            node_k[_e215] = _e218;
            let _e222 = count;
            node_nk[_e222] = 0u;
            count = def_d;
            let _e227 = masked;
            masked = (_e227 + 1u);
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
            let _e254 = l;
            if (_e254 == 0u) {
                l = len;
            }
            let _e259 = l;
            let def_f = (_e259 - 1u);
            let _e267 = compound_ops[((c_3 * 3u) + def_f)];
            op = _e267;
            let _e270 = l;
            if (_e270 == 1u) {
                let _e275 = tail_i;
                let _e278 = kid_first[pos];
                q_pos[_e275] = _e278;
                let _e282 = tail_i;
                q_left[_e282] = 0u;
                let _e286 = tail_i;
                q_demand[_e286] = demand;
                tail_i = def_e;
                let _e293 = tail_i;
                let _e296 = kid_first[pos];
                q_pos[_e293] = (_e296 + 1u);
                let _e301 = tail_i;
                q_left[_e301] = 0u;
                let _e305 = tail_i;
                q_demand[_e305] = demand;
                let _e309 = tail_i;
                tail_i = (_e309 + 1u);
                let _e313 = next;
                next = (_e313 + 2u);
                let _e318 = next;
                a0_ = (_e318 - 2u);
                nk = 2u;
            } else {
                let _e324 = tail_i;
                q_pos[_e324] = pos;
                let _e328 = tail_i;
                q_left[_e328] = def_f;
                let _e332 = tail_i;
                q_demand[_e332] = demand;
                tail_i = def_e;
                let _e337 = next;
                next = (_e337 + 1u);
                let _e342 = next;
                a0_ = (_e342 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e351 = code[sid_1];
                op = _e351;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e358 = j;
                    if (_e358 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e364 = tail_i;
                        let _e367 = kid_first[pos];
                        let _e368 = j;
                        q_pos[_e364] = (_e367 + _e368);
                        let _e373 = tail_i;
                        q_left[_e373] = 0u;
                        let _e380 = tail_i;
                        let _e383 = params.k_in;
                        let _e385 = j;
                        let _e388 = in_ty[((sid_1 * _e383) + _e385)];
                        q_demand[_e380] = _e388;
                        let _e391 = tail_i;
                        tail_i = (_e391 + 1u);
                    }
                    continuing {
                        let _e395 = j;
                        j = (_e395 + 1u);
                    }
                }
                let _e398 = next;
                next = (_e398 + a_1);
                let _e402 = next;
                a0_ = (_e402 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e412 = code[sid_1];
                    a0_ = _e412;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e420 = konst[sid_1];
                        kv = bitcast<u32>(_e420);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e431 = i_1;
                                if (_e431 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e437 = i_1;
                                    let _e440 = genome[(base + _e437)];
                                    let _e442 = kind[_e440];
                                    if (_e442 == KIND_RNC) {
                                        let _e446 = idx;
                                        idx = (_e446 + 1u);
                                    }
                                }
                                continuing {
                                    let _e450 = i_1;
                                    i_1 = (_e450 + 1u);
                                }
                            }
                            let _e455 = idx;
                            let dc = genome[((base + ht) + _e455)];
                            let _e461 = params.n_rnc;
                            if (dc >= _e461) {
                                fail(gid_1);
                                return;
                            }
                            let _e467 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e467);
                            let _e471 = n_rnc_seen;
                            n_rnc_seen = (_e471 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e480 = code[sid_1];
                                a0_ = _e480;
                            } else {
                                op = OP_NUM;
                                let _e487 = code[sid_1];
                                a0_ = (_e487 + 1u);
                                let _e492 = konst[sid_1];
                                kv = bitcast<u32>(_e492);
                            }
                        }
                    }
                }
            }
        }
        let _e497 = count;
        let _e499 = op;
        node_op[_e497] = _e499;
        let _e503 = count;
        let _e505 = a0_;
        node_a0_[_e503] = _e505;
        let _e508 = count;
        node_tc[_e508] = ty_1;
        let _e513 = count;
        let _e515 = kv;
        node_k[_e513] = _e515;
        let _e519 = count;
        let _e521 = nk;
        node_nk[_e519] = _e521;
        count = def_d;
    }
    let _e525 = count;
    k2_ = _e525;
    loop {
        let _e528 = k2_;
        if (_e528 == 0u) {
            break;
        }
        let _e532 = k2_;
        k2_ = (_e532 - 1u);
        let _e536 = k2_;
        let op_5 = node_op[_e536];
        let _e541 = k2_;
        let nk_1 = node_nk[_e541];
        let def_b = (nk_1 == 0u);
        ty = TY_F;
        d = 0u;
        if def_b {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e557 = c;
                if (_e557 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e562 = k2_;
                    let _e564 = node_a0_[_e562];
                    let _e565 = c;
                    let kid = (_e564 + _e565);
                    let _e569 = m;
                    let _e571 = node_ty[kid];
                    m = max(_e569, _e571);
                    let _e575 = d;
                    let _e577 = node_depth[kid];
                    d = max(_e575, _e577);
                }
                continuing {
                    let _e581 = c;
                    c = (_e581 + 1u);
                }
            }
            let _e584 = m;
            let _e585 = apply(op_5, _e584);
            ty = _e585;
            let _e589 = ty;
            if (_e589 == TY_NONE) {
                ok = false;
            }
            let _e593 = is_raiser_op(op_5);
            if _e593 {
                let _e596 = d;
                d = (_e596 + 1u);
            }
        }
        let _e601 = k2_;
        let _e603 = ty;
        node_ty[_e601] = _e603;
        let _e607 = k2_;
        let _e609 = d;
        node_depth[_e607] = _e609;
        if def_b {
            let _e613 = k2_;
            node_best[_e613] = 1u;
            let _e618 = k2_;
            node_ties[_e618] = 0u;
            let _e623 = k2_;
            node_order[_e623] = 1u;
        } else {
            let _e625 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e633 = c_1;
                if (_e633 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e638 = k2_;
                    let _e640 = node_a0_[_e638];
                    let _e641 = c_1;
                    let kid_1 = (_e640 + _e641);
                    let _e645 = node_op[kid_1];
                    let _e646 = family(_e645);
                    if ((_e625 != 0u) && (_e646 == _e625)) {
                        let _e654 = node_best[kid_1];
                        o = _e654;
                        let _e658 = node_ties[kid_1];
                        t = _e658;
                    } else {
                        let _e662 = node_order[kid_1];
                        o = _e662;
                        t = 1u;
                    }
                    let _e667 = o;
                    let _e668 = best;
                    if (_e667 > _e668) {
                        let _e672 = o;
                        best = _e672;
                        let _e675 = t;
                        ties = _e675;
                    } else {
                        let _e678 = o;
                        let _e679 = best;
                        if (_e678 == _e679) {
                            let _e683 = ties;
                            let _e684 = t;
                            ties = (_e683 + _e684);
                        }
                    }
                }
                continuing {
                    let _e688 = c_1;
                    c_1 = (_e688 + 1u);
                }
            }
            let _e693 = k2_;
            let _e695 = best;
            node_best[_e693] = _e695;
            let _e699 = k2_;
            let _e701 = ties;
            node_ties[_e699] = _e701;
            let _e709 = k2_;
            let _e711 = best;
            let _e712 = ties;
            node_order[_e709] = (_e711 + select(0u, 1u, (_e712 >= 2u)));
        }
    }
    let _e720 = params.typed;
    let _e722 = ok;
    if ((_e720 == 1u) && !(_e722)) {
        fail(gid_1);
        return;
    }
    let _e728 = params.order_ceiling;
    if (_e728 > 0u) {
        let _e734 = node_order[0];
        let _e736 = params.order_ceiling;
        if (_e734 > _e736) {
            fail(gid_1);
            return;
        }
        loop {
            let _e740 = k;
            let _e741 = count;
            if (_e740 < _e741) {
            } else {
                break;
            }
            {
                let _e745 = k;
                let op_6 = node_op[_e745];
                let _e751 = k;
                let _e753 = node_nk[_e751];
                if (_e753 == 0u) {
                    continue;
                }
                let _e756 = k;
                let _e757 = is_unary(_e756);
                let _e758 = is_transparent(op_6);
                if (_e757 && !(_e758)) {
                    let _e763 = k;
                    let _e765 = node_a0_[_e763];
                    let _e766 = through_transparent(_e765);
                    let _e767 = is_unary(_e766);
                    let _e773 = op_stack[op_6];
                    let _e775 = node_op[_e766];
                    if (_e767 && ((_e773 & (1u << _e775)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e783 = op_order[op_6];
                let lo = (_e783 & 255u);
                let _e788 = op_order[op_6];
                let hi = (_e788 >> 8u);
                c_2 = 0u;
                loop {
                    let _e795 = c_2;
                    let _e796 = k;
                    let _e798 = node_nk[_e796];
                    if (_e795 < _e798) {
                    } else {
                        break;
                    }
                    {
                        let _e804 = c_2;
                        if ((op_6 == OP_POW) && (_e804 == 1u)) {
                            continue;
                        }
                        let _e810 = k;
                        let _e812 = node_a0_[_e810];
                        let _e813 = c_2;
                        let kid_2 = (_e812 + _e813);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e823 = c_2;
                        c_2 = (_e823 + 1u);
                    }
                }
            }
            continuing {
                let _e827 = k;
                k = (_e827 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e836 = count;
    h = (2166136261u ^ _e836);
    loop {
        let _e840 = i_2;
        let def_a = (out + (_e840 * 4u));
        let _e845 = i_2;
        let _e846 = count;
        if (_e845 < _e846) {
        } else {
            break;
        }
        {
            let _e852 = i_2;
            let _e854 = node_op[_e852];
            nodes[def_a] = _e854;
            let _e861 = i_2;
            let _e863 = node_a0_[_e861];
            nodes[(def_a + 1u)] = _e863;
            let _e870 = i_2;
            let _e872 = node_tc[_e870];
            nodes[(def_a + 2u)] = _e872;
            let _e879 = i_2;
            let _e881 = node_k[_e879];
            nodes[(def_a + 3u)] = _e881;
            let _e886 = h;
            let _e887 = i_2;
            let _e889 = node_op[_e887];
            h = ((_e886 ^ _e889) * 16777619u);
            let _e896 = h;
            let _e897 = i_2;
            let _e899 = node_a0_[_e897];
            h = ((_e896 ^ _e899) * 16777619u);
            let _e906 = h;
            let _e907 = i_2;
            let _e909 = node_tc[_e907];
            h = ((_e906 ^ _e909) * 16777619u);
            let _e916 = h;
            let _e917 = i_2;
            let _e919 = node_k[_e917];
            h = ((_e916 ^ _e919) * 16777619u);
        }
        continuing {
            let _e924 = i_2;
            i_2 = (_e924 + 1u);
        }
    }
    let _e929 = count;
    lengths[gid_1] = _e929;
    gene_ok[gid_1] = 1u;
    let _e938 = node_depth[0];
    gene_depth[gid_1] = _e938;
    let _e942 = h;
    gene_hash[gid_1] = _e942;
    return;
}
