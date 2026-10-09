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
            let _e258 = l;
            let _e262 = compound_ops[((c_3 * 3u) + (_e258 - 1u))];
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
                nk = a_1;
            } else {
                if (k_3 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e417 = code[sid_1];
                    a0_ = _e417;
                } else {
                    if (k_3 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e425 = konst[sid_1];
                        kv = bitcast<u32>(_e425);
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
                                let _e497 = konst[sid_1];
                                kv = bitcast<u32>(_e497);
                            }
                        }
                    }
                }
            }
        }
        let _e502 = count;
        let _e504 = op;
        node_op[_e502] = _e504;
        let _e508 = count;
        let _e510 = a0_;
        node_a0_[_e508] = _e510;
        let _e513 = count;
        node_tc[_e513] = ty_1;
        let _e518 = count;
        let _e520 = kv;
        node_k[_e518] = _e520;
        let _e524 = count;
        let _e526 = nk;
        node_nk[_e524] = _e526;
        let _e529 = count;
        count = (_e529 + 1u);
    }
    let _e533 = count;
    k2_ = _e533;
    loop {
        let _e536 = k2_;
        if (_e536 == 0u) {
            break;
        }
        let _e540 = k2_;
        k2_ = (_e540 - 1u);
        let _e544 = k2_;
        let op_5 = node_op[_e544];
        let _e549 = k2_;
        let nk_1 = node_nk[_e549];
        ty = TY_F;
        d = 0u;
        if (nk_1 == 0u) {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e565 = c;
                if (_e565 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e570 = k2_;
                    let _e572 = node_a0_[_e570];
                    let _e573 = c;
                    let kid = (_e572 + _e573);
                    let _e577 = m;
                    let _e579 = node_ty[kid];
                    m = max(_e577, _e579);
                    let _e583 = d;
                    let _e585 = node_depth[kid];
                    d = max(_e583, _e585);
                }
                continuing {
                    let _e589 = c;
                    c = (_e589 + 1u);
                }
            }
            let _e592 = m;
            let _e593 = apply(op_5, _e592);
            ty = _e593;
            let _e597 = ty;
            if (_e597 == TY_NONE) {
                ok = false;
            }
            let _e601 = is_raiser_op(op_5);
            if _e601 {
                let _e604 = d;
                d = (_e604 + 1u);
            }
        }
        let _e609 = k2_;
        let _e611 = ty;
        node_ty[_e609] = _e611;
        let _e615 = k2_;
        let _e617 = d;
        node_depth[_e615] = _e617;
        if (nk_1 == 0u) {
            let _e623 = k2_;
            node_best[_e623] = 1u;
            let _e628 = k2_;
            node_ties[_e628] = 0u;
            let _e633 = k2_;
            node_order[_e633] = 1u;
        } else {
            let _e635 = family(op_5);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e643 = c_1;
                if (_e643 < nk_1) {
                } else {
                    break;
                }
                {
                    let _e648 = k2_;
                    let _e650 = node_a0_[_e648];
                    let _e651 = c_1;
                    let kid_1 = (_e650 + _e651);
                    let _e655 = node_op[kid_1];
                    let _e656 = family(_e655);
                    if ((_e635 != 0u) && (_e656 == _e635)) {
                        let _e664 = node_best[kid_1];
                        o = _e664;
                        let _e668 = node_ties[kid_1];
                        t = _e668;
                    } else {
                        let _e672 = node_order[kid_1];
                        o = _e672;
                        t = 1u;
                    }
                    let _e677 = o;
                    let _e678 = best;
                    if (_e677 > _e678) {
                        let _e682 = o;
                        best = _e682;
                        let _e685 = t;
                        ties = _e685;
                    } else {
                        let _e688 = o;
                        let _e689 = best;
                        if (_e688 == _e689) {
                            let _e693 = ties;
                            let _e694 = t;
                            ties = (_e693 + _e694);
                        }
                    }
                }
                continuing {
                    let _e698 = c_1;
                    c_1 = (_e698 + 1u);
                }
            }
            let _e703 = k2_;
            let _e705 = best;
            node_best[_e703] = _e705;
            let _e709 = k2_;
            let _e711 = ties;
            node_ties[_e709] = _e711;
            let _e719 = k2_;
            let _e721 = best;
            let _e722 = ties;
            node_order[_e719] = (_e721 + select(0u, 1u, (_e722 >= 2u)));
        }
    }
    let _e730 = params.typed;
    let _e732 = ok;
    if ((_e730 == 1u) && !(_e732)) {
        fail(gid_1);
        return;
    }
    let _e738 = params.order_ceiling;
    if (_e738 > 0u) {
        let _e744 = node_order[0];
        let _e746 = params.order_ceiling;
        if (_e744 > _e746) {
            fail(gid_1);
            return;
        }
        loop {
            let _e750 = k;
            let _e751 = count;
            if (_e750 < _e751) {
            } else {
                break;
            }
            {
                let _e755 = k;
                let op_6 = node_op[_e755];
                let _e761 = k;
                let _e763 = node_nk[_e761];
                if (_e763 == 0u) {
                    continue;
                }
                let _e766 = k;
                let _e767 = is_unary(_e766);
                let _e768 = is_transparent(op_6);
                if (_e767 && !(_e768)) {
                    let _e773 = k;
                    let _e775 = node_a0_[_e773];
                    let _e776 = through_transparent(_e775);
                    let _e777 = is_unary(_e776);
                    let _e783 = op_stack[op_6];
                    let _e785 = node_op[_e776];
                    if (_e777 && ((_e783 & (1u << _e785)) == 0u)) {
                        fail(gid_1);
                        return;
                    }
                }
                let _e793 = op_order[op_6];
                let lo = (_e793 & 255u);
                let _e798 = op_order[op_6];
                let hi = (_e798 >> 8u);
                c_2 = 0u;
                loop {
                    let _e805 = c_2;
                    let _e806 = k;
                    let _e808 = node_nk[_e806];
                    if (_e805 < _e808) {
                    } else {
                        break;
                    }
                    {
                        let _e814 = c_2;
                        if ((op_6 == OP_POW) && (_e814 == 1u)) {
                            continue;
                        }
                        let _e820 = k;
                        let _e822 = node_a0_[_e820];
                        let _e823 = c_2;
                        let kid_2 = (_e822 + _e823);
                        let o_1 = node_order[kid_2];
                        if ((o_1 < lo) || (o_1 > hi)) {
                            fail(gid_1);
                            return;
                        }
                    }
                    continuing {
                        let _e833 = c_2;
                        c_2 = (_e833 + 1u);
                    }
                }
            }
            continuing {
                let _e837 = k;
                k = (_e837 + 1u);
            }
        }
    }
    let out = ((gid_1 * MAX_NODES) * 4u);
    let _e846 = count;
    h = (2166136261u ^ _e846);
    loop {
        let _e850 = i_2;
        let def_a = (out + (_e850 * 4u));
        let _e855 = i_2;
        let _e856 = count;
        if (_e855 < _e856) {
        } else {
            break;
        }
        {
            let _e862 = i_2;
            let _e864 = node_op[_e862];
            nodes[def_a] = _e864;
            let _e871 = i_2;
            let _e873 = node_a0_[_e871];
            nodes[(def_a + 1u)] = _e873;
            let _e880 = i_2;
            let _e882 = node_tc[_e880];
            nodes[(def_a + 2u)] = _e882;
            let _e889 = i_2;
            let _e891 = node_k[_e889];
            nodes[(def_a + 3u)] = _e891;
            let _e896 = h;
            let _e897 = i_2;
            let _e899 = node_op[_e897];
            h = ((_e896 ^ _e899) * 16777619u);
            let _e906 = h;
            let _e907 = i_2;
            let _e909 = node_a0_[_e907];
            h = ((_e906 ^ _e909) * 16777619u);
            let _e916 = h;
            let _e917 = i_2;
            let _e919 = node_tc[_e917];
            h = ((_e916 ^ _e919) * 16777619u);
            let _e926 = h;
            let _e927 = i_2;
            let _e929 = node_k[_e927];
            h = ((_e926 ^ _e929) * 16777619u);
        }
        continuing {
            let _e934 = i_2;
            i_2 = (_e934 + 1u);
        }
    }
    let _e939 = count;
    lengths[gid_1] = _e939;
    gene_ok[gid_1] = 1u;
    let _e948 = node_depth[0];
    gene_depth[gid_1] = _e948;
    let _e952 = h;
    gene_hash[gid_1] = _e952;
    return;
}
