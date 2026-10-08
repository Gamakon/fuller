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

    let _e18 = params.pop;
    let _e20 = params.n_genes;
    if (id.x >= (_e18 * _e20)) {
        return;
    }
    offsets[id.x] = (id.x * MAX_NODES);
    loop {
        let _e34 = need;
        let _e36 = n;
        let _e38 = params.head;
        let _e40 = params.tail;
        if ((_e34 <= 0i) || (_e36 >= (_e38 + _e40))) {
            break;
        }
        let _e52 = need;
        let _e55 = params.head;
        let _e57 = params.tail;
        let _e61 = n;
        let _e64 = genome[((id.x * (_e55 + (2u * _e57))) + _e61)];
        let _e66 = arity[_e64];
        need = ((_e52 + i32(_e66)) - 1i);
        let _e72 = n;
        n = (_e72 + 1u);
    }
    let _e78 = need;
    let _e81 = params.head;
    let _e83 = params.tail;
    if ((_e78 > 0i) || ((_e81 + _e83) > MAX_HT)) {
        fail(id.x);
        return;
    }
    loop {
        let _e91 = i;
        let _e92 = n;
        if (_e91 < _e92) {
        } else {
            break;
        }
        {
            let _e104 = i;
            let _e106 = child;
            let _e109 = params.head;
            let _e111 = params.tail;
            let _e115 = i;
            let _e118 = genome[((id.x * (_e109 + (2u * _e111))) + _e115)];
            let _e120 = arity[_e118];
            kid_first[_e104] = select(0u, _e106, (_e120 >= 1u));
            let _e130 = child;
            let _e133 = params.head;
            let _e135 = params.tail;
            let _e139 = i;
            let _e142 = genome[((id.x * (_e133 + (2u * _e135))) + _e139)];
            let _e144 = arity[_e142];
            child = (_e130 + _e144);
            let _e155 = params.head;
            let _e157 = params.tail;
            let _e161 = i;
            let _e164 = genome[((id.x * (_e155 + (2u * _e157))) + _e161)];
            let _e166 = kind[_e164];
            if (_e166 == KIND_RNC) {
                let _e170 = n_q;
                n_q = (_e170 + 1u);
            }
        }
        continuing {
            let _e174 = i;
            i = (_e174 + 1u);
        }
    }
    let _e178 = n_q;
    let _e180 = params.n_rnc;
    if (_e178 > _e180) {
        fail(id.x);
        return;
    }
    q_pos[0] = 0u;
    q_left[0] = 0u;
    let _e195 = params.root_ty;
    q_demand[0] = _e195;
    loop {
        let _e198 = head_i;
        let _e199 = tail_i;
        if (_e198 >= _e199) {
            break;
        }
        let _e203 = head_i;
        head_i = (_e203 + 1u);
        let _e207 = count;
        if (_e207 >= MAX_NODES) {
            fail(id.x);
            return;
        }
        let _e223 = params.typed_kingdom;
        let _e225 = head_i;
        let _e227 = q_left[_e225];
        let _e232 = params.head;
        let _e234 = params.tail;
        let _e238 = head_i;
        let _e240 = q_pos[_e238];
        let _e243 = genome[((id.x * (_e232 + (2u * _e234))) + _e240)];
        let _e245 = out_ty[_e243];
        let _e246 = head_i;
        let _e248 = q_demand[_e246];
        if (((_e223 == 1u) && (_e227 == 0u)) && (_e245 != _e248)) {
            let _e256 = count;
            let _e258 = head_i;
            let _e260 = q_demand[_e258];
            let _e262 = fallback_op[_e260];
            node_op[_e256] = _e262;
            let _e266 = count;
            node_a0_[_e266] = 0u;
            let _e271 = count;
            node_a1_[_e271] = 0u;
            let _e278 = count;
            let _e280 = head_i;
            let _e282 = q_demand[_e280];
            let _e284 = fallback_k[_e282];
            node_k[_e278] = _e284;
            let _e288 = count;
            node_nk[_e288] = 0u;
            let _e292 = count;
            count = (_e292 + 1u);
            let _e296 = masked;
            masked = (_e296 + 1u);
            continue;
        }
        op = 0u;
        a0_ = 0u;
        a1_ = 0u;
        kv = 0u;
        nk = 0u;
        let _e318 = params.head;
        let _e320 = params.tail;
        let _e324 = head_i;
        let _e326 = q_pos[_e324];
        let _e329 = genome[((id.x * (_e318 + (2u * _e320))) + _e326)];
        let _e331 = kind[_e329];
        if (_e331 == KIND_COMPOUND) {
            let _e336 = head_i;
            let _e338 = q_left[_e336];
            l = _e338;
            let _e341 = l;
            if (_e341 == 0u) {
                let _e354 = params.head;
                let _e356 = params.tail;
                let _e360 = head_i;
                let _e362 = q_pos[_e360];
                let _e365 = genome[((id.x * (_e354 + (2u * _e356))) + _e362)];
                let _e367 = code[_e365];
                let _e369 = compound_len[_e367];
                l = _e369;
            }
            let _e384 = params.head;
            let _e386 = params.tail;
            let _e390 = head_i;
            let _e392 = q_pos[_e390];
            let _e395 = genome[((id.x * (_e384 + (2u * _e386))) + _e392)];
            let _e397 = code[_e395];
            let _e399 = l;
            let _e403 = compound_ops[((_e397 * 3u) + (_e399 - 1u))];
            op = _e403;
            let _e406 = l;
            if (_e406 == 1u) {
                let _e412 = tail_i;
                let _e414 = head_i;
                let _e416 = q_pos[_e414];
                let _e418 = kid_first[_e416];
                q_pos[_e412] = _e418;
                let _e422 = tail_i;
                q_left[_e422] = 0u;
                let _e427 = tail_i;
                let _e429 = head_i;
                let _e431 = q_demand[_e429];
                q_demand[_e427] = _e431;
                let _e434 = tail_i;
                tail_i = (_e434 + 1u);
                let _e441 = tail_i;
                let _e443 = head_i;
                let _e445 = q_pos[_e443];
                let _e447 = kid_first[_e445];
                q_pos[_e441] = (_e447 + 1u);
                let _e452 = tail_i;
                q_left[_e452] = 0u;
                let _e457 = tail_i;
                let _e459 = head_i;
                let _e461 = q_demand[_e459];
                q_demand[_e457] = _e461;
                let _e464 = tail_i;
                tail_i = (_e464 + 1u);
                let _e468 = next;
                next = (_e468 + 2u);
                let _e473 = next;
                a0_ = (_e473 - 2u);
                let _e478 = next;
                a1_ = (_e478 - 1u);
                nk = 2u;
            } else {
                let _e485 = tail_i;
                let _e487 = head_i;
                let _e489 = q_pos[_e487];
                q_pos[_e485] = _e489;
                let _e494 = tail_i;
                let _e496 = l;
                q_left[_e494] = (_e496 - 1u);
                let _e501 = tail_i;
                let _e503 = head_i;
                let _e505 = q_demand[_e503];
                q_demand[_e501] = _e505;
                let _e508 = tail_i;
                tail_i = (_e508 + 1u);
                let _e512 = next;
                next = (_e512 + 1u);
                let _e517 = next;
                a0_ = (_e517 - 1u);
                nk = 1u;
            }
        } else {
            let _e531 = params.head;
            let _e533 = params.tail;
            let _e537 = head_i;
            let _e539 = q_pos[_e537];
            let _e542 = genome[((id.x * (_e531 + (2u * _e533))) + _e539)];
            let _e544 = kind[_e542];
            if (_e544 == KIND_FUNCTION) {
                let _e556 = params.head;
                let _e558 = params.tail;
                let _e562 = head_i;
                let _e564 = q_pos[_e562];
                let _e567 = genome[((id.x * (_e556 + (2u * _e558))) + _e564)];
                let _e569 = code[_e567];
                op = _e569;
                j = 0u;
                loop {
                    let _e580 = j;
                    let _e583 = params.head;
                    let _e585 = params.tail;
                    let _e589 = head_i;
                    let _e591 = q_pos[_e589];
                    let _e594 = genome[((id.x * (_e583 + (2u * _e585))) + _e591)];
                    let _e596 = arity[_e594];
                    if (_e580 < _e596) {
                    } else {
                        break;
                    }
                    {
                        let _e603 = tail_i;
                        let _e605 = head_i;
                        let _e607 = q_pos[_e605];
                        let _e609 = kid_first[_e607];
                        let _e610 = j;
                        q_pos[_e603] = (_e609 + _e610);
                        let _e615 = tail_i;
                        q_left[_e615] = 0u;
                        let _e627 = tail_i;
                        let _e631 = params.head;
                        let _e633 = params.tail;
                        let _e637 = head_i;
                        let _e639 = q_pos[_e637];
                        let _e642 = genome[((id.x * (_e631 + (2u * _e633))) + _e639)];
                        let _e644 = params.k_in;
                        let _e646 = j;
                        let _e649 = in_ty[((_e642 * _e644) + _e646)];
                        q_demand[_e627] = _e649;
                        let _e652 = tail_i;
                        tail_i = (_e652 + 1u);
                    }
                    continuing {
                        let _e656 = j;
                        j = (_e656 + 1u);
                    }
                }
                let _e666 = next;
                let _e669 = params.head;
                let _e671 = params.tail;
                let _e675 = head_i;
                let _e677 = q_pos[_e675];
                let _e680 = genome[((id.x * (_e669 + (2u * _e671))) + _e677)];
                let _e682 = arity[_e680];
                next = (_e666 + _e682);
                let _e693 = next;
                let _e696 = params.head;
                let _e698 = params.tail;
                let _e702 = head_i;
                let _e704 = q_pos[_e702];
                let _e707 = genome[((id.x * (_e696 + (2u * _e698))) + _e704)];
                let _e709 = arity[_e707];
                a0_ = (_e693 - _e709);
                let _e722 = a0_;
                let _e726 = params.head;
                let _e728 = params.tail;
                let _e732 = head_i;
                let _e734 = q_pos[_e732];
                let _e737 = genome[((id.x * (_e726 + (2u * _e728))) + _e734)];
                let _e739 = arity[_e737];
                a1_ = select(0u, (_e722 + 1u), (_e739 >= 2u));
                let _e752 = params.head;
                let _e754 = params.tail;
                let _e758 = head_i;
                let _e760 = q_pos[_e758];
                let _e763 = genome[((id.x * (_e752 + (2u * _e754))) + _e760)];
                let _e765 = arity[_e763];
                nk = _e765;
            } else {
                let _e776 = params.head;
                let _e778 = params.tail;
                let _e782 = head_i;
                let _e784 = q_pos[_e782];
                let _e787 = genome[((id.x * (_e776 + (2u * _e778))) + _e784)];
                let _e789 = kind[_e787];
                if (_e789 == KIND_INPUT) {
                    op = OP_VAR;
                    let _e803 = params.head;
                    let _e805 = params.tail;
                    let _e809 = head_i;
                    let _e811 = q_pos[_e809];
                    let _e814 = genome[((id.x * (_e803 + (2u * _e805))) + _e811)];
                    let _e816 = code[_e814];
                    a0_ = _e816;
                } else {
                    let _e827 = params.head;
                    let _e829 = params.tail;
                    let _e833 = head_i;
                    let _e835 = q_pos[_e833];
                    let _e838 = genome[((id.x * (_e827 + (2u * _e829))) + _e835)];
                    let _e840 = kind[_e838];
                    if (_e840 == KIND_CONSTANT) {
                        op = OP_NUM;
                        let _e854 = params.head;
                        let _e856 = params.tail;
                        let _e860 = head_i;
                        let _e862 = q_pos[_e860];
                        let _e865 = genome[((id.x * (_e854 + (2u * _e856))) + _e862)];
                        let _e867 = konst[_e865];
                        kv = bitcast<u32>(_e867);
                    } else {
                        let _e879 = params.head;
                        let _e881 = params.tail;
                        let _e885 = head_i;
                        let _e887 = q_pos[_e885];
                        let _e890 = genome[((id.x * (_e879 + (2u * _e881))) + _e887)];
                        let _e892 = kind[_e890];
                        if (_e892 == KIND_RNC) {
                            op = OP_NUM;
                            idx = 0u;
                            i_1 = 0u;
                            loop {
                                let _e903 = i_1;
                                let _e904 = head_i;
                                let _e906 = q_pos[_e904];
                                if (_e903 < _e906) {
                                } else {
                                    break;
                                }
                                {
                                    let _e917 = params.head;
                                    let _e919 = params.tail;
                                    let _e923 = i_1;
                                    let _e926 = genome[((id.x * (_e917 + (2u * _e919))) + _e923)];
                                    let _e928 = kind[_e926];
                                    if (_e928 == KIND_RNC) {
                                        let _e932 = idx;
                                        idx = (_e932 + 1u);
                                    }
                                }
                                continuing {
                                    let _e936 = i_1;
                                    i_1 = (_e936 + 1u);
                                }
                            }
                            let _e945 = params.head;
                            let _e947 = params.tail;
                            let _e952 = params.head;
                            let _e954 = params.tail;
                            let _e957 = idx;
                            let _e960 = genome[(((id.x * (_e945 + (2u * _e947))) + (_e952 + _e954)) + _e957)];
                            let _e962 = params.n_rnc;
                            if (_e960 >= _e962) {
                                fail(id.x);
                                return;
                            }
                            let _e975 = params.n_rnc;
                            let _e979 = params.head;
                            let _e981 = params.tail;
                            let _e986 = params.head;
                            let _e988 = params.tail;
                            let _e991 = idx;
                            let _e994 = genome[(((id.x * (_e979 + (2u * _e981))) + (_e986 + _e988)) + _e991)];
                            let _e997 = rnc[((id.x * _e975) + _e994)];
                            kv = bitcast<u32>(_e997);
                            let _e1001 = n_rnc_seen;
                            n_rnc_seen = (_e1001 + 1u);
                        } else {
                            let _e1013 = params.head;
                            let _e1015 = params.tail;
                            let _e1019 = head_i;
                            let _e1021 = q_pos[_e1019];
                            let _e1024 = genome[((id.x * (_e1013 + (2u * _e1015))) + _e1021)];
                            let _e1026 = kind[_e1024];
                            if (_e1026 == KIND_GENEREF) {
                                op = OP_GENEREF;
                                let _e1040 = params.head;
                                let _e1042 = params.tail;
                                let _e1046 = head_i;
                                let _e1048 = q_pos[_e1046];
                                let _e1051 = genome[((id.x * (_e1040 + (2u * _e1042))) + _e1048)];
                                let _e1053 = code[_e1051];
                                a0_ = _e1053;
                            } else {
                                op = OP_NUM;
                                let _e1067 = params.head;
                                let _e1069 = params.tail;
                                let _e1073 = head_i;
                                let _e1075 = q_pos[_e1073];
                                let _e1078 = genome[((id.x * (_e1067 + (2u * _e1069))) + _e1075)];
                                let _e1080 = code[_e1078];
                                a0_ = (_e1080 + 1u);
                                let _e1092 = params.head;
                                let _e1094 = params.tail;
                                let _e1098 = head_i;
                                let _e1100 = q_pos[_e1098];
                                let _e1103 = genome[((id.x * (_e1092 + (2u * _e1094))) + _e1100)];
                                let _e1105 = konst[_e1103];
                                kv = bitcast<u32>(_e1105);
                            }
                        }
                    }
                }
            }
        }
        let _e1110 = count;
        let _e1112 = op;
        node_op[_e1110] = _e1112;
        let _e1116 = count;
        let _e1118 = a0_;
        node_a0_[_e1116] = _e1118;
        let _e1122 = count;
        let _e1124 = a1_;
        node_a1_[_e1122] = _e1124;
        let _e1128 = count;
        let _e1130 = kv;
        node_k[_e1128] = _e1130;
        let _e1134 = count;
        let _e1136 = nk;
        node_nk[_e1134] = _e1136;
        let _e1139 = count;
        count = (_e1139 + 1u);
    }
    let _e1143 = count;
    k2_ = _e1143;
    loop {
        let _e1146 = k2_;
        if (_e1146 == 0u) {
            break;
        }
        let _e1150 = k2_;
        k2_ = (_e1150 - 1u);
        ty = TY_F;
        d = 0u;
        let _e1159 = k2_;
        let _e1161 = node_nk[_e1159];
        if (_e1161 == 0u) {
            ty = TY_F;
        } else {
            m = TY_F;
            c = 0u;
            loop {
                let _e1172 = c;
                let _e1173 = k2_;
                let _e1175 = node_nk[_e1173];
                if (_e1172 < _e1175) {
                } else {
                    break;
                }
                {
                    let _e1182 = m;
                    let _e1183 = k2_;
                    let _e1185 = node_a0_[_e1183];
                    let _e1186 = c;
                    let _e1189 = node_ty[(_e1185 + _e1186)];
                    m = max(_e1182, _e1189);
                    let _e1196 = d;
                    let _e1197 = k2_;
                    let _e1199 = node_a0_[_e1197];
                    let _e1200 = c;
                    let _e1203 = node_depth[(_e1199 + _e1200)];
                    d = max(_e1196, _e1203);
                }
                continuing {
                    let _e1207 = c;
                    c = (_e1207 + 1u);
                }
            }
            let _e1211 = k2_;
            let _e1213 = node_op[_e1211];
            let _e1215 = m;
            let _e1216 = apply(_e1213, _e1215);
            ty = _e1216;
            let _e1220 = ty;
            if (_e1220 == TY_NONE) {
                ok = false;
            }
            let _e1226 = k2_;
            let _e1228 = node_op[_e1226];
            let _e1229 = is_raiser_op(_e1228);
            if _e1229 {
                let _e1232 = d;
                d = (_e1232 + 1u);
            }
        }
        let _e1237 = k2_;
        let _e1239 = ty;
        node_ty[_e1237] = _e1239;
        let _e1243 = k2_;
        let _e1245 = d;
        node_depth[_e1243] = _e1245;
        let _e1249 = k2_;
        let _e1251 = node_nk[_e1249];
        if (_e1251 == 0u) {
            let _e1256 = k2_;
            node_best[_e1256] = 1u;
            let _e1261 = k2_;
            node_ties[_e1261] = 0u;
            let _e1266 = k2_;
            node_order[_e1266] = 1u;
        } else {
            let _e1270 = k2_;
            let _e1272 = node_op[_e1270];
            let _e1273 = family(_e1272);
            best = 0u;
            ties = 0u;
            c_1 = 0u;
            loop {
                let _e1283 = c_1;
                let _e1284 = k2_;
                let _e1286 = node_nk[_e1284];
                if (_e1283 < _e1286) {
                } else {
                    break;
                }
                {
                    let _e1292 = k2_;
                    let _e1294 = node_a0_[_e1292];
                    let _e1295 = c_1;
                    let _e1298 = node_op[(_e1294 + _e1295)];
                    let _e1299 = family(_e1298);
                    if ((_e1273 != 0u) && (_e1299 == _e1273)) {
                        let _e1309 = k2_;
                        let _e1311 = node_a0_[_e1309];
                        let _e1312 = c_1;
                        let _e1315 = node_best[(_e1311 + _e1312)];
                        o = _e1315;
                        let _e1321 = k2_;
                        let _e1323 = node_a0_[_e1321];
                        let _e1324 = c_1;
                        let _e1327 = node_ties[(_e1323 + _e1324)];
                        t = _e1327;
                    } else {
                        let _e1333 = k2_;
                        let _e1335 = node_a0_[_e1333];
                        let _e1336 = c_1;
                        let _e1339 = node_order[(_e1335 + _e1336)];
                        o = _e1339;
                        t = 1u;
                    }
                    let _e1344 = o;
                    let _e1345 = best;
                    if (_e1344 > _e1345) {
                        let _e1349 = o;
                        best = _e1349;
                        let _e1352 = t;
                        ties = _e1352;
                    } else {
                        let _e1355 = o;
                        let _e1356 = best;
                        if (_e1355 == _e1356) {
                            let _e1360 = ties;
                            let _e1361 = t;
                            ties = (_e1360 + _e1361);
                        }
                    }
                }
                continuing {
                    let _e1365 = c_1;
                    c_1 = (_e1365 + 1u);
                }
            }
            let _e1370 = k2_;
            let _e1372 = best;
            node_best[_e1370] = _e1372;
            let _e1376 = k2_;
            let _e1378 = ties;
            node_ties[_e1376] = _e1378;
            let _e1386 = k2_;
            let _e1388 = best;
            let _e1389 = ties;
            node_order[_e1386] = (_e1388 + select(0u, 1u, (_e1389 >= 2u)));
        }
    }
    let _e1397 = params.typed;
    let _e1399 = ok;
    if ((_e1397 == 1u) && !(_e1399)) {
        fail(id.x);
        return;
    }
    let _e1407 = params.order_ceiling;
    if (_e1407 > 0u) {
        let _e1413 = node_order[0];
        let _e1415 = params.order_ceiling;
        if (_e1413 > _e1415) {
            fail(id.x);
            return;
        }
        loop {
            let _e1421 = k;
            let _e1422 = count;
            if (_e1421 < _e1422) {
            } else {
                break;
            }
            {
                let _e1427 = k;
                let _e1429 = node_nk[_e1427];
                if (_e1429 == 0u) {
                    continue;
                }
                let _e1432 = k;
                let _e1433 = is_unary(_e1432);
                let _e1436 = k;
                let _e1438 = node_op[_e1436];
                let _e1439 = is_transparent(_e1438);
                if (_e1433 && !(_e1439)) {
                    let _e1444 = k;
                    let _e1446 = node_a0_[_e1444];
                    let _e1447 = through_transparent(_e1446);
                    let _e1448 = is_unary(_e1447);
                    let _e1454 = k;
                    let _e1456 = node_op[_e1454];
                    let _e1458 = op_stack[_e1456];
                    let _e1460 = node_op[_e1447];
                    if (_e1448 && ((_e1458 & (1u << _e1460)) == 0u)) {
                        fail(id.x);
                        return;
                    }
                }
                c_2 = 0u;
                loop {
                    let _e1472 = c_2;
                    let _e1473 = k;
                    let _e1475 = node_nk[_e1473];
                    if (_e1472 < _e1475) {
                    } else {
                        break;
                    }
                    {
                        let _e1482 = k;
                        let _e1484 = node_op[_e1482];
                        let _e1486 = c_2;
                        if ((_e1484 == OP_POW) && (_e1486 == 1u)) {
                            continue;
                        }
                        let _e1497 = k;
                        let _e1499 = node_a0_[_e1497];
                        let _e1500 = c_2;
                        let _e1503 = node_order[(_e1499 + _e1500)];
                        let _e1504 = k;
                        let _e1506 = node_op[_e1504];
                        let _e1508 = op_order[_e1506];
                        let _e1511 = k;
                        let _e1513 = node_a0_[_e1511];
                        let _e1514 = c_2;
                        let _e1517 = node_order[(_e1513 + _e1514)];
                        let _e1518 = k;
                        let _e1520 = node_op[_e1518];
                        let _e1522 = op_order[_e1520];
                        if ((_e1503 < (_e1508 & 255u)) || (_e1517 > (_e1522 >> 8u))) {
                            fail(id.x);
                            return;
                        }
                    }
                    continuing {
                        let _e1530 = c_2;
                        c_2 = (_e1530 + 1u);
                    }
                }
            }
            continuing {
                let _e1534 = k;
                k = (_e1534 + 1u);
            }
        }
    }
    let _e1539 = count;
    h = (2166136261u ^ _e1539);
    loop {
        let _e1543 = i_2;
        let _e1544 = count;
        if (_e1543 < _e1544) {
        } else {
            break;
        }
        {
            let _e1555 = i_2;
            let _e1559 = i_2;
            let _e1561 = node_op[_e1559];
            nodes[(((id.x * MAX_NODES) * 4u) + (_e1555 * 4u))] = _e1561;
            let _e1572 = i_2;
            let _e1577 = i_2;
            let _e1579 = node_a0_[_e1577];
            nodes[((((id.x * MAX_NODES) * 4u) + (_e1572 * 4u)) + 1u)] = _e1579;
            let _e1590 = i_2;
            let _e1595 = i_2;
            let _e1597 = node_a1_[_e1595];
            nodes[((((id.x * MAX_NODES) * 4u) + (_e1590 * 4u)) + 2u)] = _e1597;
            let _e1608 = i_2;
            let _e1613 = i_2;
            let _e1615 = node_k[_e1613];
            nodes[((((id.x * MAX_NODES) * 4u) + (_e1608 * 4u)) + 3u)] = _e1615;
            let _e1620 = h;
            let _e1621 = i_2;
            let _e1623 = node_op[_e1621];
            h = ((_e1620 ^ _e1623) * 16777619u);
            let _e1630 = h;
            let _e1631 = i_2;
            let _e1633 = node_a0_[_e1631];
            h = ((_e1630 ^ _e1633) * 16777619u);
            let _e1640 = h;
            let _e1641 = i_2;
            let _e1643 = node_a1_[_e1641];
            h = ((_e1640 ^ _e1643) * 16777619u);
            let _e1650 = h;
            let _e1651 = i_2;
            let _e1653 = node_k[_e1651];
            h = ((_e1650 ^ _e1653) * 16777619u);
        }
        continuing {
            let _e1658 = i_2;
            i_2 = (_e1658 + 1u);
        }
    }
    let _e1665 = count;
    lengths[id.x] = _e1665;
    gene_ok[id.x] = 1u;
    let _e1678 = node_depth[0];
    gene_depth[id.x] = _e1678;
    let _e1684 = h;
    gene_hash[id.x] = _e1684;
    return;
}
