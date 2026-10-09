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
                let _e332 = tail_i;
                let _e334 = l;
                q_left[_e332] = (_e334 - 1u);
                let _e338 = tail_i;
                q_demand[_e338] = demand;
                let _e342 = tail_i;
                tail_i = (_e342 + 1u);
                let _e346 = next;
                next = (_e346 + 1u);
                let _e351 = next;
                a0_ = (_e351 - 1u);
                nk = 1u;
            }
        } else {
            if (k_3 == KIND_FUNCTION) {
                let _e360 = code[sid_1];
                op = _e360;
                let a_1 = arity[sid_1];
                j = 0u;
                loop {
                    let _e367 = j;
                    if (_e367 < a_1) {
                    } else {
                        break;
                    }
                    {
                        let _e373 = tail_i;
                        let _e376 = kid_first[pos];
                        let _e377 = j;
                        q_pos[_e373] = (_e376 + _e377);
                        let _e382 = tail_i;
                        q_left[_e382] = 0u;
                        let _e389 = tail_i;
                        let _e392 = params.k_in;
                        let _e394 = j;
                        let _e397 = in_ty[((sid_1 * _e392) + _e394)];
                        q_demand[_e389] = _e397;
                        let _e400 = tail_i;
                        tail_i = (_e400 + 1u);
                    }
                    continuing {
                        let _e404 = j;
                        j = (_e404 + 1u);
                    }
                }
                let _e407 = next;
                next = (_e407 + a_1);
                let _e411 = next;
                a0_ = (_e411 - a_1);
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e421 = code[sid_1];
                    a0_ = _e421;
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
                                let _e436 = i_1;
                                if (_e436 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e442 = i_1;
                                    let _e445 = genome[(base + _e442)];
                                    let _e447 = kind[_e445];
                                    if (_e447 == KIND_RNC) {
                                        let _e451 = idx;
                                        idx = (_e451 + 1u);
                                    }
                                }
                                continuing {
                                    let _e455 = i_1;
                                    i_1 = (_e455 + 1u);
                                }
                            }
                            let _e460 = idx;
                            let dc = genome[((base + ht) + _e460)];
                            let _e466 = params.n_rnc;
                            if (dc >= _e466) {
                                fail(gid_1);
                                return;
                            }
                            let _e472 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e472);
                            let _e476 = n_rnc_seen;
                            n_rnc_seen = (_e476 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e485 = code[sid_1];
                                a0_ = _e485;
                            } else {
                                op = OP_NUM;
                                let _e492 = code[sid_1];
                                a0_ = (_e492 + 1u);
                                kv = def_c;
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
        let def_b = (nk_1 == 0u);
        ty = TY_F;
        d = 0u;
        if def_b {
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
        if def_b {
            let _e617 = k2_;
            node_best[_e617] = 1u;
            let _e622 = k2_;
            node_ties[_e622] = 0u;
            let _e627 = k2_;
            node_order[_e627] = 1u;
        } else {
            let _e629 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e637 = c_1;
                if (_e637 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e642 = k2_;
                    let _e644 = node_a0_[_e642];
                    let _e645 = c_1;
                    let kid_1 = (_e644 + _e645);
                    let _e649 = node_op[kid_1];
                    let _e650 = family(_e649);
                    if ((_e629 != 0u) && (_e650 == _e629)) {
                        let _e658 = node_best[kid_1];
                        o = _e658;
                        let _e662 = node_ties[kid_1];
                        t = _e662;
                    } else {
                        let _e666 = node_order[kid_1];
                        o = _e666;
                        t = 1u;
                    }
                    let _e671 = o;
                    let _e672 = best;
                    if (_e671 > _e672) {
                        let _e676 = o;
                        best = _e676;
                        let _e679 = t;
                        ties = _e679;
                    } else {
                        let _e682 = o;
                        let _e683 = best;
                        if (_e682 == _e683) {
                            let _e687 = ties;
                            let _e688 = t;
                            ties = (_e687 + _e688);
                        }
                    }
                }
                continuing {
                    let _e692 = c_1;
                    c_1 = (_e692 + 1u);
                }
            }
            let _e697 = k2_;
            let _e699 = best;
            node_best[_e697] = _e699;
            let _e703 = k2_;
            let _e705 = ties;
            node_ties[_e703] = _e705;
            let _e713 = k2_;
            let _e715 = best;
            let _e716 = ties;
            node_order[_e713] = (_e715 + select(0u, 1u, (_e716 >= 2u)));
        }
    }
    let _e724 = params.typed;
    let _e726 = ok;
    if ((_e724 == 1u) && !(_e726)) {
        fail(gid_1);
        return;
    }
    let _e732 = params.order_ceiling;
    if (_e732 > 0u) {
        let _e738 = node_order[0];
        let _e740 = params.order_ceiling;
        if (_e738 > _e740) {
            fail(gid_1);
            return;
        }
        loop {
            let _e744 = k;
            let _e745 = count;
            if (_e744 < _e745) {
            } else {
                break;
            }
            {
                let _e749 = k;
                let op_6 = node_op[_e749];
                let _e755 = k;
                let _e757 = node_nk[_e755];
                if (_e757 == 0u) {
                    continue;
                }
                let _e760 = k;
                let _e761 = is_unary(_e760);
                let _e762 = is_transparent(op_6);
                if (_e761 && !(_e762)) {
                    let _e767 = k;
                    let _e769 = node_a0_[_e767];
                    let _e770 = through_transparent(_e769);
                    let _e771 = is_unary(_e770);
                    let _e777 = op_stack[op_6];
                    let _e779 = node_op[_e770];
                    if (_e771 && ((_e777 & (1u << _e779)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e787 = op_order[op_6];
                let lo = (_e787 & 255u);
                let _e792 = op_order[op_6];
                let hi = (_e792 >> 8u);
                c_2 = 0u;
                loop {
                    let _e799 = c_2;
                    let _e800 = k;
                    let _e802 = node_nk[_e800];
                    if (_e799 < _e802) {
                    } else {
                        break;
                    }
                    {
                        let _e808 = c_2;
                        if ((op_6 == OP_POW) && (_e808 == 1u)) {
                            continue;
                        }
                        let _e814 = k;
                        let _e816 = node_a0_[_e814];
                        let _e817 = c_2;
                        let kid_2 = (_e816 + _e817);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e827 = c_2;
                        c_2 = (_e827 + 1u);
                    }
                }
            }
            continuing {
                let _e831 = k;
                k = (_e831 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e840 = count;
    h = (2166136261u ^ _e840);
    loop {
        let _e844 = i_2;
        let _e845 = count;
        if (_e844 < _e845) {
        } else {
            break;
        }
        {
            let _e851 = i_2;
            let _e855 = i_2;
            let _e857 = node_op[_e855];
            nodes[(out + (_e851 * 4u))] = _e857;
            let _e863 = i_2;
            let _e868 = i_2;
            let _e870 = node_a0_[_e868];
            nodes[((out + (_e863 * 4u)) + 1u)] = _e870;
            let _e876 = i_2;
            let _e881 = i_2;
            let _e883 = node_tc[_e881];
            nodes[((out + (_e876 * 4u)) + 2u)] = _e883;
            let _e889 = i_2;
            let _e894 = i_2;
            let _e896 = node_k[_e894];
            nodes[((out + (_e889 * 4u)) + 3u)] = _e896;
            let _e901 = h;
            let _e902 = i_2;
            let _e904 = node_op[_e902];
            h = ((_e901 ^ _e904) * 16777619u);
            let _e911 = h;
            let _e912 = i_2;
            let _e914 = node_a0_[_e912];
            h = ((_e911 ^ _e914) * 16777619u);
            let _e921 = h;
            let _e922 = i_2;
            let _e924 = node_tc[_e922];
            h = ((_e921 ^ _e924) * 16777619u);
            let _e931 = h;
            let _e932 = i_2;
            let _e934 = node_k[_e932];
            h = ((_e931 ^ _e934) * 16777619u);
        }
        continuing {
            let _e939 = i_2;
            i_2 = (_e939 + 1u);
        }
    }
    let _e944 = count;
    lengths[gid_1] = _e944;
    gene_ok[gid_1] = 1u;
    let _e953 = node_depth[0];
    gene_depth[gid_1] = _e953;
    let _e957 = h;
    gene_hash[gid_1] = _e957;
    return;
}
