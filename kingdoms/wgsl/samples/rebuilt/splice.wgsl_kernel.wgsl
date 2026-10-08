struct Node {
    op: u32,
    arg0_: u32,
    arg1_: u32,
    konst: f32,
}

struct Cfg {
    n_expr: u32,
    n_guards: u32,
    admit: u32,
    rounds: u32,
    stride: u32,
    pad0_: u32,
    pad1_: u32,
    pad2_: u32,
}

const SLOT: u32 = 64u;
const WORK: u32 = 80u;
const NONE: u32 = 4294967295u;
const OP_VAR: u32 = 0u;
const OP_NUM: u32 = 1u;
const HEAP: u32 = 15u;
const RULE_STRIDE: u32 = 126u;
const GUARD_STRIDE: u32 = 8u;
const KIND_UNUSED: u32 = 0u;
const KIND_OP: u32 = 1u;
const KIND_MV: u32 = 2u;
const KIND_NUM: u32 = 3u;
const N_OPS: u32 = 28u;

var<private> cur: array<Node, 80>;
var<private> caux: array<u32, 80>;
var<private> nxt: array<Node, 80>;
var<private> naux: array<u32, 80>;
var<private> qa: array<u32, 80>;
var<private> n_cur: u32;
@group(0) @binding(0) 
var<storage> nodes_in: array<Node>;
@group(0) @binding(1) 
var<storage> aux_in: array<u32>;
@group(0) @binding(2) 
var<storage> len_in: array<u32>;
@group(0) @binding(3) 
var<storage> rules: array<u32>;
@group(0) @binding(4) 
var<storage> bucket: array<u32>;
@group(0) @binding(5) 
var<storage> guards: array<u32>;
@group(0) @binding(6) 
var<storage> pool: array<u32>;
@group(0) @binding(7) 
var<storage> var_facts: array<u32>;
@group(0) @binding(8) 
var<storage, read_write> nodes_out: array<Node>;
@group(0) @binding(9) 
var<storage, read_write> info_out: array<u32>;
@group(0) @binding(10) 
var<uniform> cfg: Cfg;
var<private> strict: array<u32, 80>;
var<private> loose: array<u32, 80>;
var<private> qb: array<u32, 80>;
var<private> at: array<u32, 15>;
var<private> home: array<u32, 15>;
var<private> bind: array<u32, 8>;

fn arity(op: u32) -> u32 {
    switch op {
        case 0u, 1u: {
            return 0u;
        }
        case 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 16u, 17u, 18u, 20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u: {
            return 1u;
        }
        default: {
            return 2u;
        }
    }
}

fn splice(pos_1: u32, new_root: u32) {
    var root: u32 = 0u;
    var i_2: u32 = 0u;
    var head: u32 = 0u;
    var tail: u32 = 1u;
    var a0_: u32;
    var a1_: u32;
    var i_3: u32 = 0u;

    if (pos_1 == 0u) {
        root = new_root;
    } else {
        loop {
            let _e12 = i_2;
            let _e13 = n_cur;
            if (_e12 < _e13) {
            } else {
                break;
            }
            {
                let _e17 = i_2;
                let _e20 = cur[_e17].op;
                let _e21 = arity(_e20);
                let _e27 = i_2;
                let _e30 = cur[_e27].arg0_;
                if ((_e21 >= 1u) && (_e30 == pos_1)) {
                    let _e36 = i_2;
                    cur[_e36].arg0_ = new_root;
                }
                let _e44 = i_2;
                let _e47 = cur[_e44].arg1_;
                if ((_e21 >= 2u) && (_e47 == pos_1)) {
                    let _e53 = i_2;
                    cur[_e53].arg1_ = new_root;
                }
            }
            continuing {
                let _e58 = i_2;
                i_2 = (_e58 + 1u);
            }
        }
    }
    let _e64 = root;
    qa[0] = _e64;
    loop {
        let _e67 = head;
        let _e68 = tail;
        if (_e67 >= _e68) {
            break;
        }
        let _e73 = head;
        let _e75 = qa[_e73];
        let old = cur[_e75];
        let _e81 = head;
        let _e83 = qa[_e81];
        let oaux = caux[_e83];
        let _e87 = arity(old.op);
        let first = tail;
        if (_e87 >= 1u) {
            let _e94 = tail;
            qa[_e94] = old.arg0_;
            let _e99 = tail;
            tail = (_e99 + 1u);
        }
        if (_e87 >= 2u) {
            let _e105 = tail;
            qa[_e105] = old.arg1_;
            let _e110 = tail;
            tail = (_e110 + 1u);
        }
        a0_ = old.arg0_;
        a1_ = old.arg1_;
        if (_e87 >= 1u) {
            a0_ = first;
            a1_ = 0u;
        }
        if (_e87 >= 2u) {
            a1_ = (first + 1u);
        }
        let _e130 = head;
        let _e133 = a0_;
        let _e134 = a1_;
        nxt[_e130] = Node(old.op, _e133, _e134, old.konst);
        let _e139 = head;
        naux[_e139] = oaux;
        let _e143 = head;
        head = (_e143 + 1u);
    }
    let _e147 = tail;
    n_cur = _e147;
    loop {
        let _e150 = i_3;
        let _e151 = n_cur;
        if (_e150 < _e151) {
        } else {
            break;
        }
        {
            let _e156 = i_3;
            let _e158 = i_3;
            let _e160 = nxt[_e158];
            cur[_e156] = _e160;
            let _e164 = i_3;
            let _e166 = i_3;
            let _e168 = naux[_e166];
            caux[_e164] = _e168;
        }
        continuing {
            let _e171 = i_3;
            i_3 = (_e171 + 1u);
        }
    }
    return;
}

fn facts_pass() {
    var k: u32;
    var s: u32;
    var l: u32;
    var it: u32;
    var g: u32;
    var shape_s: bool;
    var shape_l: bool;

    let _e2 = n_cur;
    k = _e2;
    loop {
        let _e5 = k;
        if (_e5 == 0u) {
            break;
        }
        let _e9 = k;
        k = (_e9 - 1u);
        let _e13 = k;
        let nd = cur[_e13];
        s = 0u;
        l = 0u;
        if (nd.op == OP_VAR) {
            let _e27 = var_facts[nd.arg0_];
            s = _e27;
            let _e30 = s;
            l = _e30;
        }
        it = 0u;
        loop {
            let _e35 = it;
            if (_e35 < 3u) {
            } else {
                break;
            }
            {
                g = 0u;
                loop {
                    let _e41 = g;
                    let _e43 = cfg.n_guards;
                    if (_e41 < _e43) {
                    } else {
                        break;
                    }
                    {
                        let _e47 = g;
                        let b_1 = (_e47 * GUARD_STRIDE);
                        let g_op = guards[b_1];
                        let req0_ = guards[(b_1 + 1u)];
                        let req1_ = guards[(b_1 + 2u)];
                        let self_req = guards[(b_1 + 3u)];
                        let gives = guards[(b_1 + 4u)];
                        let range_only = guards[(b_1 + 5u)];
                        let pred_req = guards[(b_1 + 6u)];
                        shape_s = true;
                        shape_l = true;
                        if (g_op != NONE) {
                            if (nd.op != g_op) {
                                shape_s = false;
                                shape_l = false;
                            } else {
                                let _e95 = arity(nd.op);
                                if (_e95 >= 1u) {
                                    let _e100 = shape_s;
                                    let _e103 = strict[nd.arg0_];
                                    shape_s = (_e100 && ((_e103 & req0_) == req0_));
                                    let _e109 = shape_l;
                                    let _e112 = loose[nd.arg0_];
                                    shape_l = (_e109 && ((_e112 & req0_) == req0_));
                                }
                                if (_e95 >= 2u) {
                                    let _e120 = shape_s;
                                    let _e123 = strict[nd.arg1_];
                                    shape_s = (_e120 && ((_e123 & req1_) == req1_));
                                    let _e129 = shape_l;
                                    let _e132 = loose[nd.arg1_];
                                    shape_l = (_e129 && ((_e132 & req1_) == req1_));
                                }
                                if (nd.op == OP_NUM) {
                                    let _e142 = k;
                                    let _e144 = caux[_e142];
                                    let ok = (((_e144 >> 8u) & pred_req) == pred_req);
                                    let _e149 = shape_s;
                                    shape_s = (_e149 && ok);
                                    let _e152 = shape_l;
                                    shape_l = (_e152 && ok);
                                }
                            }
                        }
                        let _e156 = shape_l;
                        let _e157 = l;
                        if (_e156 && ((_e157 & self_req) == self_req)) {
                            let _e162 = l;
                            l = (_e162 | gives);
                        }
                        let _e168 = shape_s;
                        let _e170 = s;
                        if (((range_only == 0u) && _e168) && ((_e170 & self_req) == self_req)) {
                            let _e175 = s;
                            s = (_e175 | gives);
                        }
                    }
                    continuing {
                        let _e179 = g;
                        g = (_e179 + 1u);
                    }
                }
            }
            continuing {
                let _e183 = it;
                it = (_e183 + 1u);
            }
        }
        let _e188 = k;
        let _e190 = s;
        strict[_e188] = _e190;
        let _e194 = k;
        let _e196 = l;
        loose[_e194] = _e196;
    }
    return;
}

fn same_subtree(a: u32, b: u32) -> bool {
    var head_1: u32 = 0u;
    var tail_1: u32 = 1u;

    qa[0] = a;
    qb[0] = b;
    loop {
        let _e12 = head_1;
        let _e13 = tail_1;
        if (_e12 >= _e13) {
            break;
        }
        let _e17 = head_1;
        let x = qa[_e17];
        let _e22 = head_1;
        let y = qb[_e22];
        let _e27 = head_1;
        head_1 = (_e27 + 1u);
        if (x == y) {
            continue;
        }
        let nx = cur[x];
        let ny = cur[y];
        if (nx.op != ny.op) {
            return false;
        }
        let _e41 = arity(nx.op);
        if (_e41 == 0u) {
            if (nx.op == OP_NUM) {
                if (nx.arg1_ != ny.arg1_) {
                    return false;
                }
            } else {
                if (nx.arg0_ != ny.arg0_) {
                    return false;
                }
            }
        } else {
            let _e57 = tail_1;
            if ((_e57 + _e41) > WORK) {
                return false;
            }
            let _e63 = tail_1;
            qa[_e63] = nx.arg0_;
            let _e68 = tail_1;
            qb[_e68] = ny.arg0_;
            let _e73 = tail_1;
            tail_1 = (_e73 + 1u);
            if (_e41 == 2u) {
                let _e79 = tail_1;
                qa[_e79] = nx.arg1_;
                let _e84 = tail_1;
                qb[_e84] = ny.arg1_;
                let _e89 = tail_1;
                tail_1 = (_e89 + 1u);
            }
        }
    }
    return true;
}

fn match_at(r_1: u32, pos_2: u32, use_loose: bool) -> u32 {
    var i_4: u32 = 0u;
    var i_5: u32 = 0u;
    var needed_loose: bool = false;
    var slot: u32 = 0u;

    loop {
        let _e6 = i_4;
        if (_e6 < HEAP) {
        } else {
            break;
        }
        {
            let _e11 = i_4;
            at[_e11] = NONE;
        }
        continuing {
            let _e15 = i_4;
            i_4 = (_e15 + 1u);
        }
    }
    loop {
        let _e19 = i_5;
        if (_e19 < 8u) {
        } else {
            break;
        }
        {
            let _e24 = i_5;
            bind[_e24] = NONE;
        }
        continuing {
            let _e28 = i_5;
            i_5 = (_e28 + 1u);
        }
    }
    at[0] = pos_2;
    let pat = ((r_1 * RULE_STRIDE) + 6u);
    loop {
        let _e41 = slot;
        if (_e41 < HEAP) {
        } else {
            break;
        }
        {
            let _e46 = slot;
            let kind = rules[(pat + (_e46 * 4u))];
            let _e56 = slot;
            let _e58 = at[_e56];
            if ((kind == KIND_UNUSED) || (_e58 == NONE)) {
                continue;
            }
            let _e65 = slot;
            let id = rules[((pat + (_e65 * 4u)) + 1u)];
            let _e75 = slot;
            let req = rules[((pat + (_e75 * 4u)) + 2u)];
            let _e85 = slot;
            let lit = rules[((pat + (_e85 * 4u)) + 3u)];
            let _e93 = slot;
            let idx = at[_e93];
            let nd_1 = cur[idx];
            if (kind == KIND_OP) {
                if (nd_1.op != id) {
                    return 0u;
                }
                let _e105 = arity(nd_1.op);
                if (_e105 >= 1u) {
                    let _e112 = slot;
                    at[((2u * _e112) + 1u)] = nd_1.arg0_;
                }
                if (_e105 >= 2u) {
                    let _e122 = slot;
                    at[((2u * _e122) + 2u)] = nd_1.arg1_;
                }
            } else {
                if (kind == KIND_MV) {
                    let _e132 = bind[id];
                    if (_e132 == NONE) {
                        bind[id] = idx;
                        let _e138 = strict[idx];
                        if ((_e138 & req) != req) {
                            let _e144 = loose[idx];
                            if (use_loose && ((_e144 & req) == req)) {
                                needed_loose = true;
                            } else {
                                return 0u;
                            }
                        }
                    } else {
                        let _e153 = bind[id];
                        let _e154 = same_subtree(_e153, idx);
                        if !(_e154) {
                            return 0u;
                        }
                    }
                } else {
                    if (nd_1.op != OP_NUM) {
                        return 0u;
                    }
                    if (lit != 0u) {
                        let _e166 = caux[idx];
                        if ((_e166 & 255u) != lit) {
                            return 0u;
                        }
                    } else {
                        let _e173 = caux[idx];
                        if (((_e173 >> 8u) & req) != req) {
                            return 0u;
                        }
                    }
                }
            }
        }
        continuing {
            let _e180 = slot;
            slot = (_e180 + 1u);
        }
    }
    let _e183 = needed_loose;
    if _e183 {
        return 2u;
    }
    return 1u;
}

fn apply(r_2: u32, pos_3: u32) {
    var n: u32;
    var slot_1: u32 = 0u;
    var slot_2: u32 = 0u;
    var a0_1: u32;
    var a1_1: u32;

    let tm = (((r_2 * RULE_STRIDE) + 6u) + 60u);
    let _e11 = n_cur;
    n = _e11;
    loop {
        let _e14 = slot_1;
        if (_e14 < HEAP) {
        } else {
            break;
        }
        {
            let _e19 = slot_1;
            let kind_1 = rules[(tm + (_e19 * 4u))];
            let _e27 = slot_1;
            home[_e27] = NONE;
            if ((kind_1 == KIND_OP) || (kind_1 == KIND_NUM)) {
                let _e37 = slot_1;
                let _e39 = n;
                home[_e37] = _e39;
                let _e42 = n;
                n = (_e42 + 1u);
            } else {
                if (kind_1 == KIND_MV) {
                    let _e52 = slot_1;
                    let _e54 = slot_1;
                    let _e59 = rules[((tm + (_e54 * 4u)) + 1u)];
                    let _e61 = bind[_e59];
                    home[_e52] = _e61;
                }
            }
        }
        continuing {
            let _e64 = slot_1;
            slot_1 = (_e64 + 1u);
        }
    }
    loop {
        let _e68 = slot_2;
        if (_e68 < HEAP) {
        } else {
            break;
        }
        {
            let _e73 = slot_2;
            let kind_2 = rules[(tm + (_e73 * 4u))];
            let _e82 = slot_2;
            let id_1 = rules[((tm + (_e82 * 4u)) + 1u)];
            if (kind_2 == KIND_OP) {
                let _e90 = arity(id_1);
                a0_1 = 0u;
                a1_1 = 0u;
                if (_e90 >= 1u) {
                    let _e102 = slot_2;
                    let _e106 = home[((2u * _e102) + 1u)];
                    a0_1 = _e106;
                }
                if (_e90 >= 2u) {
                    let _e113 = slot_2;
                    let _e117 = home[((2u * _e113) + 2u)];
                    a1_1 = _e117;
                }
                let _e124 = slot_2;
                let _e126 = home[_e124];
                let _e128 = a0_1;
                let _e129 = a1_1;
                cur[_e126] = Node(id_1, _e128, _e129, 0f);
                let _e135 = slot_2;
                let _e137 = home[_e135];
                caux[_e137] = 0u;
            } else {
                if (kind_2 == KIND_NUM) {
                    let _e149 = slot_2;
                    let _e151 = home[_e149];
                    let _e156 = pool[((id_1 * 3u) + 1u)];
                    let _e159 = pool[(id_1 * 3u)];
                    cur[_e151] = Node(OP_NUM, 0u, _e156, bitcast<f32>(_e159));
                    let _e168 = slot_2;
                    let _e170 = home[_e168];
                    let _e175 = pool[((id_1 * 3u) + 2u)];
                    caux[_e170] = _e175;
                }
            }
        }
        continuing {
            let _e178 = slot_2;
            slot_2 = (_e178 + 1u);
        }
    }
    let _e184 = home[0];
    splice(pos_3, _e184);
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn lint_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i: u32 = 0u;
    var steps: u32 = 0u;
    var level: u32 = 0u;
    var round: u32 = 0u;
    var hit_rule: u32;
    var hit_pos: u32;
    var hit_kind: u32;
    var order: u32;
    var pos: u32;
    var r: u32;
    var used: u32;
    var i_1: u32 = 0u;

    let _e9 = cfg.stride;
    let e = ((gid.y * _e9) + gid.x);
    let _e15 = cfg.n_expr;
    if (e >= _e15) {
        return;
    }
    let base = (e * SLOT);
    let _e22 = len_in[e];
    n_cur = _e22;
    let _e26 = n_cur;
    let _e28 = n_cur;
    if ((_e26 == 0u) || (_e28 > SLOT)) {
        info_out[(e * 4u)] = 0u;
        info_out[((e * 4u) + 1u)] = 0u;
        info_out[((e * 4u) + 2u)] = 0u;
        info_out[((e * 4u) + 3u)] = 1u;
        return;
    }
    loop {
        let _e59 = i;
        let _e60 = n_cur;
        if (_e59 < _e60) {
        } else {
            break;
        }
        {
            let _e65 = i;
            let _e67 = i;
            let _e70 = nodes_in[(base + _e67)];
            cur[_e65] = _e70;
            let _e74 = i;
            let _e76 = i;
            let _e79 = aux_in[(base + _e76)];
            caux[_e74] = _e79;
        }
        continuing {
            let _e82 = i;
            i = (_e82 + 1u);
        }
    }
    let _e87 = cfg.admit;
    let use_loose_1 = (_e87 == 2u);
    loop {
        let _e91 = round;
        let _e93 = cfg.rounds;
        if (_e91 < _e93) {
        } else {
            break;
        }
        {
            facts_pass();
            hit_rule = NONE;
            hit_pos = 0u;
            hit_kind = 0u;
            order = 0u;
            loop {
                let _e107 = order;
                let _e109 = hit_rule;
                if ((_e107 < 2u) && (_e109 == NONE)) {
                } else {
                    break;
                }
                {
                    pos = 0u;
                    loop {
                        let _e118 = pos;
                        let _e119 = n_cur;
                        let _e121 = hit_rule;
                        if ((_e118 < _e119) && (_e121 == NONE)) {
                        } else {
                            break;
                        }
                        {
                            let _e126 = pos;
                            let op_1 = cur[_e126].op;
                            let _e133 = bucket[op_1];
                            r = _e133;
                            loop {
                                let _e137 = r;
                                let _e140 = bucket[(op_1 + 1u)];
                                if (_e137 < _e140) {
                                } else {
                                    break;
                                }
                                {
                                    let _e144 = r;
                                    let h = (_e144 * RULE_STRIDE);
                                    let _e153 = rules[(h + 4u)];
                                    let _e154 = order;
                                    let _e158 = rules[(h + 5u)];
                                    let _e160 = cfg.admit;
                                    if ((_e153 != _e154) || (_e158 > _e160)) {
                                        continue;
                                    }
                                    let _e164 = r;
                                    let _e166 = pos;
                                    let _e167 = match_at(_e164, _e166, use_loose_1);
                                    if (_e167 != 0u) {
                                        let _e172 = r;
                                        hit_rule = _e172;
                                        let _e175 = pos;
                                        hit_pos = _e175;
                                        hit_kind = _e167;
                                        break;
                                    }
                                }
                                continuing {
                                    let _e179 = r;
                                    r = (_e179 + 1u);
                                }
                            }
                        }
                        continuing {
                            let _e183 = pos;
                            pos = (_e183 + 1u);
                        }
                    }
                }
                continuing {
                    let _e187 = order;
                    order = (_e187 + 1u);
                }
            }
            let _e191 = hit_rule;
            if (_e191 == NONE) {
                break;
            }
            let _e198 = hit_rule;
            let _e202 = rules[((_e198 * RULE_STRIDE) + 5u)];
            used = _e202;
            let _e205 = hit_kind;
            if (_e205 == 2u) {
                used = 2u;
            }
            let _e211 = level;
            let _e212 = used;
            level = max(_e211, _e212);
            let _e215 = hit_rule;
            let _e217 = hit_pos;
            apply(_e215, _e217);
            let _e220 = steps;
            steps = (_e220 + 1u);
        }
        continuing {
            let _e224 = round;
            round = (_e224 + 1u);
        }
    }
    loop {
        let _e228 = i_1;
        let _e229 = n_cur;
        if (_e228 < _e229) {
        } else {
            break;
        }
        {
            let _e234 = i_1;
            let _e237 = i_1;
            let _e239 = cur[_e237];
            nodes_out[(base + _e234)] = _e239;
        }
        continuing {
            let _e242 = i_1;
            i_1 = (_e242 + 1u);
        }
    }
    let _e249 = n_cur;
    info_out[(e * 4u)] = _e249;
    let _e257 = steps;
    info_out[((e * 4u) + 1u)] = _e257;
    let _e265 = level;
    info_out[((e * 4u) + 2u)] = _e265;
    info_out[((e * 4u) + 3u)] = 0u;
    return;
}
