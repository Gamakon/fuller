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
        let _e180 = konst[sid_1];
        let def_c = bitcast<u32>(_e180);
        let _e187 = params.typed_kingdom;
        let _e192 = out_ty[sid_1];
        if (((_e187 == 1u) && (left == 0u)) && (_e192 != demand)) {
            let _e198 = fallback_op[demand];
            if (_e198 == 4294967295u) {
                fail(gid_1);
                return;
            }
            let _e203 = count;
            let _e206 = fallback_op[demand];
            node_op[_e203] = _e206;
            let _e210 = count;
            node_a0_[_e210] = 0u;
            let _e214 = count;
            node_tc[_e214] = demand;
            let _e219 = count;
            let _e222 = fallback_k[demand];
            node_k[_e219] = _e222;
            let _e226 = count;
            node_nk[_e226] = 0u;
            count = def_d;
            let _e231 = masked;
            masked = (_e231 + 1u);
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
            let _e258 = l;
            if (_e258 == 0u) {
                l = len;
            }
            let _e263 = l;
            let def_f = (_e263 - 1u);
            let _e271 = compound_ops[((c_3 * 3u) + def_f)];
            op = _e271;
            let _e274 = l;
            if (_e274 == 1u) {
                let _e279 = tail_i;
                let _e282 = kid_first[pos];
                q_pos[_e279] = _e282;
                let _e286 = tail_i;
                q_left[_e286] = 0u;
                let _e290 = tail_i;
                q_demand[_e290] = demand;
                tail_i = def_e;
                let _e297 = tail_i;
                let _e300 = kid_first[pos];
                q_pos[_e297] = (_e300 + 1u);
                let _e305 = tail_i;
                q_left[_e305] = 0u;
                let _e309 = tail_i;
                q_demand[_e309] = demand;
                let _e313 = tail_i;
                tail_i = (_e313 + 1u);
                let _e317 = next;
                next = (_e317 + 2u);
                let _e322 = next;
                a0_ = (_e322 - 2u);
                nk = 2u;
            } else {
                let _e328 = tail_i;
                q_pos[_e328] = pos;
                let _e332 = tail_i;
                q_left[_e332] = def_f;
                let _e336 = tail_i;
                q_demand[_e336] = demand;
                tail_i = def_e;
                let _e341 = next;
                next = (_e341 + 1u);
                let _e346 = next;
                a0_ = (_e346 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e355 = code[sid_1];
                op = _e355;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e362 = j;
                    if (_e362 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e368 = tail_i;
                        let _e371 = kid_first[pos];
                        let _e372 = j;
                        q_pos[_e368] = (_e371 + _e372);
                        let _e377 = tail_i;
                        q_left[_e377] = 0u;
                        let _e384 = tail_i;
                        let _e387 = params.k_in;
                        let _e389 = j;
                        let _e392 = in_ty[((sid_1 * _e387) + _e389)];
                        q_demand[_e384] = _e392;
                        let _e395 = tail_i;
                        tail_i = (_e395 + 1u);
                    }
                    continuing {
                        let _e399 = j;
                        j = (_e399 + 1u);
                    }
                }
                let _e402 = next;
                next = (_e402 + a_1);
                let _e406 = next;
                a0_ = (_e406 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e416 = code[sid_1];
                    a0_ = _e416;
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
                                kv = def_c;
                            }
                        }
                    }
                }
            }
        }
        let _e493 = count;
        let _e495 = op;
        node_op[_e493] = _e495;
        let _e499 = count;
        let _e501 = a0_;
        node_a0_[_e499] = _e501;
        let _e504 = count;
        node_tc[_e504] = ty_1;
        let _e509 = count;
        let _e511 = kv;
        node_k[_e509] = _e511;
        let _e515 = count;
        let _e517 = nk;
        node_nk[_e515] = _e517;
        count = def_d;
    }
    let _e521 = count;
    k2_ = _e521;
    loop {
        let _e524 = k2_;
        if (_e524 == 0u) {
            break;
        }
        let _e528 = k2_;
        k2_ = (_e528 - 1u);
        let _e532 = k2_;
        let op_5 = node_op[_e532];
        let _e537 = k2_;
        let nk_1 = node_nk[_e537];
        ty = TY_F;
        d = 0u;
        if (nk_1 == 0u) {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e553 = c;
                if (_e553 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e558 = k2_;
                    let _e560 = node_a0_[_e558];
                    let _e561 = c;
                    let kid = (_e560 + _e561);
                    let _e565 = m;
                    let _e567 = node_ty[kid];
                    m = max(_e565, _e567);
                    let _e571 = d;
                    let _e573 = node_depth[kid];
                    d = max(_e571, _e573);
                }
                continuing {
                    let _e577 = c;
                    c = (_e577 + 1u);
                }
            }
            let _e580 = m;
            let _e581 = apply(op_5, _e580);
            ty = _e581;
            let _e585 = ty;
            if (_e585 == TY_NONE) {
                ok = false;
            }
            let _e589 = is_raiser_op(op_5);
            if _e589 {
                let _e592 = d;
                d = (_e592 + 1u);
            }
        }
        let _e597 = k2_;
        let _e599 = ty;
        node_ty[_e597] = _e599;
        let _e603 = k2_;
        let _e605 = d;
        node_depth[_e603] = _e605;
        if (nk_1 == 0u) {
            let _e611 = k2_;
            node_best[_e611] = 1u;
            let _e616 = k2_;
            node_ties[_e616] = 0u;
            let _e621 = k2_;
            node_order[_e621] = 1u;
        } else {
            let _e623 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e631 = c_1;
                if (_e631 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e636 = k2_;
                    let _e638 = node_a0_[_e636];
                    let _e639 = c_1;
                    let kid_1 = (_e638 + _e639);
                    let _e643 = node_op[kid_1];
                    let _e644 = family(_e643);
                    if ((_e623 != 0u) && (_e644 == _e623)) {
                        let _e652 = node_best[kid_1];
                        o = _e652;
                        let _e656 = node_ties[kid_1];
                        t = _e656;
                    } else {
                        let _e660 = node_order[kid_1];
                        o = _e660;
                        t = 1u;
                    }
                    let _e665 = o;
                    let _e666 = best;
                    if (_e665 > _e666) {
                        let _e670 = o;
                        best = _e670;
                        let _e673 = t;
                        ties = _e673;
                    } else {
                        let _e676 = o;
                        let _e677 = best;
                        if (_e676 == _e677) {
                            let _e681 = ties;
                            let _e682 = t;
                            ties = (_e681 + _e682);
                        }
                    }
                }
                continuing {
                    let _e686 = c_1;
                    c_1 = (_e686 + 1u);
                }
            }
            let _e691 = k2_;
            let _e693 = best;
            node_best[_e691] = _e693;
            let _e697 = k2_;
            let _e699 = ties;
            node_ties[_e697] = _e699;
            let _e707 = k2_;
            let _e709 = best;
            let _e710 = ties;
            node_order[_e707] = (_e709 + select(0u, 1u, (_e710 >= 2u)));
        }
    }
    let _e718 = params.typed;
    let _e720 = ok;
    if ((_e718 == 1u) && !(_e720)) {
        fail(gid_1);
        return;
    }
    let _e726 = params.order_ceiling;
    if (_e726 > 0u) {
        let _e732 = node_order[0];
        let _e734 = params.order_ceiling;
        if (_e732 > _e734) {
            fail(gid_1);
            return;
        }
        loop {
            let _e738 = k;
            let _e739 = count;
            if (_e738 < _e739) {
            } else {
                break;
            }
            {
                let _e743 = k;
                let op_6 = node_op[_e743];
                let _e749 = k;
                let _e751 = node_nk[_e749];
                if (_e751 == 0u) {
                    continue;
                }
                let _e754 = k;
                let _e755 = is_unary(_e754);
                let _e756 = is_transparent(op_6);
                if (_e755 && !(_e756)) {
                    let _e761 = k;
                    let _e763 = node_a0_[_e761];
                    let _e764 = through_transparent(_e763);
                    let _e765 = is_unary(_e764);
                    let _e771 = op_stack[op_6];
                    let _e773 = node_op[_e764];
                    if (_e765 && ((_e771 & (1u << _e773)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e781 = op_order[op_6];
                let lo = (_e781 & 255u);
                let _e786 = op_order[op_6];
                let hi = (_e786 >> 8u);
                c_2 = 0u;
                loop {
                    let _e793 = c_2;
                    let _e794 = k;
                    let _e796 = node_nk[_e794];
                    if (_e793 < _e796) {
                    } else {
                        break;
                    }
                    {
                        let _e802 = c_2;
                        if ((op_6 == OP_POW) && (_e802 == 1u)) {
                            continue;
                        }
                        let _e808 = k;
                        let _e810 = node_a0_[_e808];
                        let _e811 = c_2;
                        let kid_2 = (_e810 + _e811);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e821 = c_2;
                        c_2 = (_e821 + 1u);
                    }
                }
            }
            continuing {
                let _e825 = k;
                k = (_e825 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e834 = count;
    h = (2166136261u ^ _e834);
    loop {
        let _e838 = i_2;
        let _e839 = count;
        if (_e838 < _e839) {
        } else {
            break;
        }
        {
            let _e845 = i_2;
            let _e849 = i_2;
            let _e851 = node_op[_e849];
            nodes[(out + (_e845 * 4u))] = _e851;
            let _e857 = i_2;
            let _e862 = i_2;
            let _e864 = node_a0_[_e862];
            nodes[((out + (_e857 * 4u)) + 1u)] = _e864;
            let _e870 = i_2;
            let _e875 = i_2;
            let _e877 = node_tc[_e875];
            nodes[((out + (_e870 * 4u)) + 2u)] = _e877;
            let _e883 = i_2;
            let _e888 = i_2;
            let _e890 = node_k[_e888];
            nodes[((out + (_e883 * 4u)) + 3u)] = _e890;
            let _e895 = h;
            let _e896 = i_2;
            let _e898 = node_op[_e896];
            h = ((_e895 ^ _e898) * 16777619u);
            let _e905 = h;
            let _e906 = i_2;
            let _e908 = node_a0_[_e906];
            h = ((_e905 ^ _e908) * 16777619u);
            let _e915 = h;
            let _e916 = i_2;
            let _e918 = node_tc[_e916];
            h = ((_e915 ^ _e918) * 16777619u);
            let _e925 = h;
            let _e926 = i_2;
            let _e928 = node_k[_e926];
            h = ((_e925 ^ _e928) * 16777619u);
        }
        continuing {
            let _e933 = i_2;
            i_2 = (_e933 + 1u);
        }
    }
    let _e938 = count;
    lengths[gid_1] = _e938;
    gene_ok[gid_1] = 1u;
    let _e947 = node_depth[0];
    gene_depth[gid_1] = _e947;
    let _e951 = h;
    gene_hash[gid_1] = _e951;
    return;
}
