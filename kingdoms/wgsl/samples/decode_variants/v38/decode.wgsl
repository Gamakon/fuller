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
        let _e172 = konst[sid_1];
        let def_c = bitcast<u32>(_e172);
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
            let _e258 = l;
            let def_f = (_e258 - 1u);
            let _e266 = compound_ops[((c_3 * 3u) + def_f)];
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
                let _e289 = tail_i;
                tail_i = (_e289 + 1u);
                let _e295 = tail_i;
                let _e298 = kid_first[pos];
                q_pos[_e295] = (_e298 + 1u);
                let _e303 = tail_i;
                q_left[_e303] = 0u;
                let _e307 = tail_i;
                q_demand[_e307] = demand;
                let _e311 = tail_i;
                tail_i = (_e311 + 1u);
                let _e315 = next;
                next = (_e315 + 2u);
                let _e320 = next;
                a0_ = (_e320 - 2u);
                nk = 2u;
            } else {
                let _e326 = tail_i;
                q_pos[_e326] = pos;
                let _e330 = tail_i;
                q_left[_e330] = def_f;
                let _e334 = tail_i;
                q_demand[_e334] = demand;
                let _e338 = tail_i;
                tail_i = (_e338 + 1u);
                let _e342 = next;
                next = (_e342 + 1u);
                let _e347 = next;
                a0_ = (_e347 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e356 = code[sid_1];
                op = _e356;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e363 = j;
                    if (_e363 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e369 = tail_i;
                        let _e372 = kid_first[pos];
                        let _e373 = j;
                        q_pos[_e369] = (_e372 + _e373);
                        let _e378 = tail_i;
                        q_left[_e378] = 0u;
                        let _e385 = tail_i;
                        let _e388 = params.k_in;
                        let _e390 = j;
                        let _e393 = in_ty[((sid_1 * _e388) + _e390)];
                        q_demand[_e385] = _e393;
                        let _e396 = tail_i;
                        tail_i = (_e396 + 1u);
                    }
                    continuing {
                        let _e400 = j;
                        j = (_e400 + 1u);
                    }
                }
                let _e403 = next;
                next = (_e403 + a_1);
                let _e407 = next;
                a0_ = (_e407 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e417 = code[sid_1];
                    a0_ = _e417;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        kv = def_c;
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
                                kv = def_c;
                            }
                        }
                    }
                }
            }
        }
        let _e494 = count;
        let _e496 = op;
        node_op[_e494] = _e496;
        let _e500 = count;
        let _e502 = a0_;
        node_a0_[_e500] = _e502;
        let _e505 = count;
        node_tc[_e505] = ty_1;
        let _e510 = count;
        let _e512 = kv;
        node_k[_e510] = _e512;
        let _e516 = count;
        let _e518 = nk;
        node_nk[_e516] = _e518;
        let _e521 = count;
        count = (_e521 + 1u);
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
        let _e841 = count;
        if (_e840 < _e841) {
        } else {
            break;
        }
        {
            let _e847 = i_2;
            let _e851 = i_2;
            let _e853 = node_op[_e851];
            nodes[(out + (_e847 * 4u))] = _e853;
            let _e859 = i_2;
            let _e864 = i_2;
            let _e866 = node_a0_[_e864];
            nodes[((out + (_e859 * 4u)) + 1u)] = _e866;
            let _e872 = i_2;
            let _e877 = i_2;
            let _e879 = node_tc[_e877];
            nodes[((out + (_e872 * 4u)) + 2u)] = _e879;
            let _e885 = i_2;
            let _e890 = i_2;
            let _e892 = node_k[_e890];
            nodes[((out + (_e885 * 4u)) + 3u)] = _e892;
            let _e897 = h;
            let _e898 = i_2;
            let _e900 = node_op[_e898];
            h = ((_e897 ^ _e900) * 16777619u);
            let _e907 = h;
            let _e908 = i_2;
            let _e910 = node_a0_[_e908];
            h = ((_e907 ^ _e910) * 16777619u);
            let _e917 = h;
            let _e918 = i_2;
            let _e920 = node_tc[_e918];
            h = ((_e917 ^ _e920) * 16777619u);
            let _e927 = h;
            let _e928 = i_2;
            let _e930 = node_k[_e928];
            h = ((_e927 ^ _e930) * 16777619u);
        }
        continuing {
            let _e935 = i_2;
            i_2 = (_e935 + 1u);
        }
    }
    let _e940 = count;
    lengths[gid_1] = _e940;
    gene_ok[gid_1] = 1u;
    let _e949 = node_depth[0];
    gene_depth[gid_1] = _e949;
    let _e953 = h;
    gene_hash[gid_1] = _e953;
    return;
}
