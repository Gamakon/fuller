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
            let _e263 = l;
            let _e267 = compound_ops[((c_3 * 3u) + (_e263 - 1u))];
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
                let _e330 = tail_i;
                let _e332 = l;
                q_left[_e330] = (_e332 - 1u);
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
                        let _e424 = konst[sid_1];
                        kv = bitcast<u32>(_e424);
                    } else {
                        if (k_3 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e435 = i_1;
                                if (_e435 < pos) {
                                } else {
                                    break;
                                }
                                {
                                    let _e441 = i_1;
                                    let _e444 = genome[(base + _e441)];
                                    let _e446 = kind[_e444];
                                    if (_e446 == KIND_RNC) {
                                        let _e450 = idx;
                                        idx = (_e450 + 1u);
                                    }
                                }
                                continuing {
                                    let _e454 = i_1;
                                    i_1 = (_e454 + 1u);
                                }
                            }
                            let _e459 = idx;
                            let dc = genome[((base + ht) + _e459)];
                            let _e465 = params.n_rnc;
                            if (dc >= _e465) {
                                fail(gid_1);
                                return;
                            }
                            let _e471 = rnc[(rbase + dc)];
                            kv = bitcast<u32>(_e471);
                            let _e475 = n_rnc_seen;
                            n_rnc_seen = (_e475 + 1u);
                        } else {
                            if (k_3 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e484 = code[sid_1];
                                a0_ = _e484;
                            } else {
                                op = OP_NUM;
                                let _e491 = code[sid_1];
                                a0_ = (_e491 + 1u);
                                let _e496 = konst[sid_1];
                                kv = bitcast<u32>(_e496);
                            }
                        }
                    }
                }
            }
        }
        let _e501 = count;
        let _e503 = op;
        node_op[_e501] = _e503;
        let _e507 = count;
        let _e509 = a0_;
        node_a0_[_e507] = _e509;
        let _e512 = count;
        node_tc[_e512] = ty_1;
        let _e517 = count;
        let _e519 = kv;
        node_k[_e517] = _e519;
        let _e523 = count;
        let _e525 = nk;
        node_nk[_e523] = _e525;
        count = def_d;
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
        let def_a = (out + (_e844 * 4u));
        let _e849 = i_2;
        let _e850 = count;
        if (_e849 < _e850) {
        } else {
            break;
        }
        {
            let _e856 = i_2;
            let _e858 = node_op[_e856];
            nodes[def_a] = _e858;
            let _e865 = i_2;
            let _e867 = node_a0_[_e865];
            nodes[(def_a + 1u)] = _e867;
            let _e874 = i_2;
            let _e876 = node_tc[_e874];
            nodes[(def_a + 2u)] = _e876;
            let _e883 = i_2;
            let _e885 = node_k[_e883];
            nodes[(def_a + 3u)] = _e885;
            let _e890 = h;
            let _e891 = i_2;
            let _e893 = node_op[_e891];
            h = ((_e890 ^ _e893) * 16777619u);
            let _e900 = h;
            let _e901 = i_2;
            let _e903 = node_a0_[_e901];
            h = ((_e900 ^ _e903) * 16777619u);
            let _e910 = h;
            let _e911 = i_2;
            let _e913 = node_tc[_e911];
            h = ((_e910 ^ _e913) * 16777619u);
            let _e920 = h;
            let _e921 = i_2;
            let _e923 = node_k[_e921];
            h = ((_e920 ^ _e923) * 16777619u);
        }
        continuing {
            let _e928 = i_2;
            i_2 = (_e928 + 1u);
        }
    }
    let _e933 = count;
    lengths[gid_1] = _e933;
    gene_ok[gid_1] = 1u;
    let _e942 = node_depth[0];
    gene_depth[gid_1] = _e942;
    let _e946 = h;
    gene_hash[gid_1] = _e946;
    return;
}
