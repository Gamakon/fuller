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
var<private> node_a1_: array<u32, 64>;
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
    var a1_: u32;
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
            let _e186 = count;
            let _e189 = fallback_op[demand];
            node_op[_e186] = _e189;
            let _e193 = count;
            node_a0_[_e193] = 0u;
            let _e198 = count;
            node_a1_[_e198] = 0u;
            let _e203 = count;
            let _e206 = fallback_k[demand];
            node_k[_e203] = _e206;
            let _e210 = count;
            node_nk[_e210] = 0u;
            let _e214 = count;
            count = (_e214 + 1u);
            let _e218 = masked;
            masked = (_e218 + 1u);
            continue;
        }
        let k_3 = kind[sid_1];
        op = 0u;
        a0_ = 0u;
        a1_ = 0u;
        kv = 0u;
        nk = 0u;
        if (k_3 == KIND_COMPOUND) {
            let c_3 = code[sid_1];
            let len = compound_len[c_3];
            l = left;
            let _e244 = l;
            if (_e244 == 0u) {
                l = len;
            }
            let _e253 = l;
            let _e257 = compound_ops[((c_3 * 3u) + (_e253 - 1u))];
            op = _e257;
            let _e260 = l;
            if (_e260 == 1u) {
                let _e265 = tail_i;
                let _e268 = kid_first[pos];
                q_pos[_e265] = _e268;
                let _e272 = tail_i;
                q_left[_e272] = 0u;
                let _e276 = tail_i;
                q_demand[_e276] = demand;
                let _e280 = tail_i;
                tail_i = (_e280 + 1u);
                let _e286 = tail_i;
                let _e289 = kid_first[pos];
                q_pos[_e286] = (_e289 + 1u);
                let _e294 = tail_i;
                q_left[_e294] = 0u;
                let _e298 = tail_i;
                q_demand[_e298] = demand;
                let _e302 = tail_i;
                tail_i = (_e302 + 1u);
                let _e306 = next;
                next = (_e306 + 2u);
                let _e311 = next;
                a0_ = (_e311 - 2u);
                let _e316 = next;
                a1_ = (_e316 - 1u);
                nk = 2u;
            } else {
                let _e322 = tail_i;
                q_pos[_e322] = pos;
                let _e328 = tail_i;
                let _e330 = l;
                q_left[_e328] = (_e330 - 1u);
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
                let _e414 = a0_;
                a1_ = select(0u, (_e414 + 1u), (a_1 >= 2u));
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e426 = code[sid_1];
                    a0_ = _e426;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e434 = konst[sid_1];
                        kv = bitcast<u32>(_e434);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e445 = i_1;
                                if (_e445 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e451 = i_1;
                                    let _e454 = genome[(base + _e451)];
                                    let _e456 = kind[_e454];
                                    if (_e456 == KIND_RNC) {
                                        let _e460 = idx;
                                        idx = (_e460 + 1u);
                                    }
                                }
                                continuing {
                                    let _e464 = i_1;
                                    i_1 = (_e464 + 1u);
                                }
                            }
                            let _e469 = idx;
                            let dc = genome[((base + ht) + _e469)];
                            let _e475 = params.n_rnc;
                            if (dc >= _e475) {
                                fail(gid_1);
                                return;
                            }
                            let _e481 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e481);
                            let _e485 = n_rnc_seen;
                            n_rnc_seen = (_e485 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e494 = code[sid_1];
                                a0_ = _e494;
                            } else {
                                op = OP_NUM;
                                let _e501 = code[sid_1];
                                a0_ = (_e501 + 1u);
                                let _e506 = konst[sid_1];
                                kv = bitcast<u32>(_e506);
                            }
                        }
                    }
                }
            }
        }
        let _e511 = count;
        let _e513 = op;
        node_op[_e511] = _e513;
        let _e517 = count;
        let _e519 = a0_;
        node_a0_[_e517] = _e519;
        let _e523 = count;
        let _e525 = a1_;
        node_a1_[_e523] = _e525;
        let _e529 = count;
        let _e531 = kv;
        node_k[_e529] = _e531;
        let _e535 = count;
        let _e537 = nk;
        node_nk[_e535] = _e537;
        let _e540 = count;
        count = (_e540 + 1u);
    }
    let _e544 = count;
    k2_ = _e544;
    loop {
        let _e547 = k2_;
        if (_e547 == 0u) {
            break;
        }
        let _e551 = k2_;
        k2_ = (_e551 - 1u);
        let _e555 = k2_;
        let op_5 = node_op[_e555];
        let _e560 = k2_;
        let nk_1 = node_nk[_e560];
        ty = TY_F;
        d = 0u;
        if (nk_1 == 0u) {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e576 = c;
                if (_e576 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e581 = k2_;
                    let _e583 = node_a0_[_e581];
                    let _e584 = c;
                    let kid = (_e583 + _e584);
                    let _e588 = m;
                    let _e590 = node_ty[kid];
                    m = max(_e588, _e590);
                    let _e594 = d;
                    let _e596 = node_depth[kid];
                    d = max(_e594, _e596);
                }
                continuing {
                    let _e600 = c;
                    c = (_e600 + 1u);
                }
            }
            let _e603 = m;
            let _e604 = apply(op_5, _e603);
            ty = _e604;
            let _e608 = ty;
            if (_e608 == TY_NONE) {
                ok = false;
            }
            let _e612 = is_raiser_op(op_5);
            if _e612 {
                let _e615 = d;
                d = (_e615 + 1u);
            }
        }
        let _e620 = k2_;
        let _e622 = ty;
        node_ty[_e620] = _e622;
        let _e626 = k2_;
        let _e628 = d;
        node_depth[_e626] = _e628;
        if (nk_1 == 0u) {
            let _e634 = k2_;
            node_best[_e634] = 1u;
            let _e639 = k2_;
            node_ties[_e639] = 0u;
            let _e644 = k2_;
            node_order[_e644] = 1u;
        } else {
            let _e646 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e654 = c_1;
                if (_e654 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e659 = k2_;
                    let _e661 = node_a0_[_e659];
                    let _e662 = c_1;
                    let kid_1 = (_e661 + _e662);
                    let _e666 = node_op[kid_1];
                    let _e667 = family(_e666);
                    if ((_e646 != 0u) && (_e667 == _e646)) {
                        let _e675 = node_best[kid_1];
                        o = _e675;
                        let _e679 = node_ties[kid_1];
                        t = _e679;
                    } else {
                        let _e683 = node_order[kid_1];
                        o = _e683;
                        t = 1u;
                    }
                    let _e688 = o;
                    let _e689 = best;
                    if (_e688 > _e689) {
                        let _e693 = o;
                        best = _e693;
                        let _e696 = t;
                        ties = _e696;
                    } else {
                        let _e699 = o;
                        let _e700 = best;
                        if (_e699 == _e700) {
                            let _e704 = ties;
                            let _e705 = t;
                            ties = (_e704 + _e705);
                        }
                    }
                }
                continuing {
                    let _e709 = c_1;
                    c_1 = (_e709 + 1u);
                }
            }
            let _e714 = k2_;
            let _e716 = best;
            node_best[_e714] = _e716;
            let _e720 = k2_;
            let _e722 = ties;
            node_ties[_e720] = _e722;
            let _e730 = k2_;
            let _e732 = best;
            let _e733 = ties;
            node_order[_e730] = (_e732 + select(0u, 1u, (_e733 >= 2u)));
        }
    }
    let _e741 = params.typed;
    let _e743 = ok;
    if ((_e741 == 1u) && !(_e743)) {
        fail(gid_1);
        return;
    }
    let _e749 = params.order_ceiling;
    if (_e749 > 0u) {
        let _e755 = node_order[0];
        let _e757 = params.order_ceiling;
        if (_e755 > _e757) {
            fail(gid_1);
            return;
        }
        loop {
            let _e761 = k;
            let _e762 = count;
            if (_e761 < _e762) {
            } else {
                break;
            }
            {
                let _e766 = k;
                let op_6 = node_op[_e766];
                let _e772 = k;
                let _e774 = node_nk[_e772];
                if (_e774 == 0u) {
                    continue;
                }
                let _e777 = k;
                let _e778 = is_unary(_e777);
                let _e779 = is_transparent(op_6);
                if (_e778 && !(_e779)) {
                    let _e784 = k;
                    let _e786 = node_a0_[_e784];
                    let _e787 = through_transparent(_e786);
                    let _e788 = is_unary(_e787);
                    let _e794 = op_stack[op_6];
                    let _e796 = node_op[_e787];
                    if (_e788 && ((_e794 & (1u << _e796)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e804 = op_order[op_6];
                let lo = (_e804 & 255u);
                let _e809 = op_order[op_6];
                let hi = (_e809 >> 8u);
                c_2 = 0u;
                loop {
                    let _e816 = c_2;
                    let _e817 = k;
                    let _e819 = node_nk[_e817];
                    if (_e816 < _e819) {
                    } else {
                        break;
                    }
                    {
                        let _e825 = c_2;
                        if ((op_6 == OP_POW) && (_e825 == 1u)) {
                            continue;
                        }
                        let _e831 = k;
                        let _e833 = node_a0_[_e831];
                        let _e834 = c_2;
                        let kid_2 = (_e833 + _e834);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e844 = c_2;
                        c_2 = (_e844 + 1u);
                    }
                }
            }
            continuing {
                let _e848 = k;
                k = (_e848 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e857 = count;
    h = (2166136261u ^ _e857);
    loop {
        let _e861 = i_2;
        let _e862 = count;
        if (_e861 < _e862) {
        } else {
            break;
        }
        {
            let _e868 = i_2;
            let _e872 = i_2;
            let _e874 = node_op[_e872];
            nodes[(out + (_e868 * 4u))] = _e874;
            let _e880 = i_2;
            let _e885 = i_2;
            let _e887 = node_a0_[_e885];
            nodes[((out + (_e880 * 4u)) + 1u)] = _e887;
            let _e893 = i_2;
            let _e898 = i_2;
            let _e900 = node_a1_[_e898];
            nodes[((out + (_e893 * 4u)) + 2u)] = _e900;
            let _e906 = i_2;
            let _e911 = i_2;
            let _e913 = node_k[_e911];
            nodes[((out + (_e906 * 4u)) + 3u)] = _e913;
            let _e918 = h;
            let _e919 = i_2;
            let _e921 = node_op[_e919];
            h = ((_e918 ^ _e921) * 16777619u);
            let _e928 = h;
            let _e929 = i_2;
            let _e931 = node_a0_[_e929];
            h = ((_e928 ^ _e931) * 16777619u);
            let _e938 = h;
            let _e939 = i_2;
            let _e941 = node_a1_[_e939];
            h = ((_e938 ^ _e941) * 16777619u);
            let _e948 = h;
            let _e949 = i_2;
            let _e951 = node_k[_e949];
            h = ((_e948 ^ _e951) * 16777619u);
        }
        continuing {
            let _e956 = i_2;
            i_2 = (_e956 + 1u);
        }
    }
    let _e961 = count;
    lengths[gid_1] = _e961;
    gene_ok[gid_1] = 1u;
    let _e970 = node_depth[0];
    gene_depth[gid_1] = _e970;
    let _e974 = h;
    gene_hash[gid_1] = _e974;
    return;
}
