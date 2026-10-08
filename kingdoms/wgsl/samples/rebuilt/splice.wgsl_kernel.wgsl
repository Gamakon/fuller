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
        let _e77 = cur[_e75];
        let _e79 = arity(_e77.op);
        if (_e79 >= 1u) {
            let _e86 = tail;
            let _e88 = head;
            let _e90 = qa[_e88];
            let _e92 = cur[_e90];
            qa[_e86] = _e92.arg0_;
            let _e96 = tail;
            tail = (_e96 + 1u);
        }
        if (_e79 >= 2u) {
            let _e104 = tail;
            let _e106 = head;
            let _e108 = qa[_e106];
            let _e110 = cur[_e108];
            qa[_e104] = _e110.arg1_;
            let _e114 = tail;
            tail = (_e114 + 1u);
        }
        let _e120 = head;
        let _e122 = qa[_e120];
        let _e124 = cur[_e122];
        a0_ = _e124.arg0_;
        let _e130 = head;
        let _e132 = qa[_e130];
        let _e134 = cur[_e132];
        a1_ = _e134.arg1_;
        if (_e79 >= 1u) {
            let _e140 = tail;
            a0_ = _e140;
            a1_ = 0u;
        }
        if (_e79 >= 2u) {
            let _e148 = tail;
            a1_ = (_e148 + 1u);
        }
        let _e156 = head;
        let _e158 = head;
        let _e160 = qa[_e158];
        let _e162 = cur[_e160];
        let _e164 = a0_;
        let _e165 = a1_;
        let _e166 = head;
        let _e168 = qa[_e166];
        let _e170 = cur[_e168];
        nxt[_e156] = Node(_e162.op, _e164, _e165, _e170.konst);
        let _e177 = head;
        let _e179 = head;
        let _e181 = qa[_e179];
        let _e183 = caux[_e181];
        naux[_e177] = _e183;
        let _e186 = head;
        head = (_e186 + 1u);
    }
    let _e190 = tail;
    n_cur = _e190;
    loop {
        let _e193 = i_3;
        let _e194 = n_cur;
        if (_e193 < _e194) {
        } else {
            break;
        }
        {
            let _e199 = i_3;
            let _e201 = i_3;
            let _e203 = nxt[_e201];
            cur[_e199] = _e203;
            let _e207 = i_3;
            let _e209 = i_3;
            let _e211 = naux[_e209];
            caux[_e207] = _e211;
        }
        continuing {
            let _e214 = i_3;
            i_3 = (_e214 + 1u);
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
        s = 0u;
        l = 0u;
        let _e18 = k;
        let _e20 = cur[_e18];
        if (_e20.op == OP_VAR) {
            let _e27 = k;
            let _e29 = cur[_e27];
            let _e32 = var_facts[_e29.arg0_];
            s = _e32;
            let _e35 = s;
            l = _e35;
        }
        it = 0u;
        loop {
            let _e40 = it;
            if (_e40 < 3u) {
            } else {
                break;
            }
            {
                g = 0u;
                loop {
                    let _e46 = g;
                    let _e48 = cfg.n_guards;
                    if (_e46 < _e48) {
                    } else {
                        break;
                    }
                    {
                        shape_s = true;
                        shape_l = true;
                        let _e58 = g;
                        let _e61 = guards[(_e58 * GUARD_STRIDE)];
                        if (_e61 != NONE) {
                            let _e68 = k;
                            let _e70 = cur[_e68];
                            let _e72 = g;
                            let _e75 = guards[(_e72 * GUARD_STRIDE)];
                            if (_e70.op != _e75) {
                                shape_s = false;
                                shape_l = false;
                            } else {
                                let _e83 = k;
                                let _e85 = cur[_e83];
                                let _e87 = arity(_e85.op);
                                if (_e87 >= 1u) {
                                    let _e98 = shape_s;
                                    let _e99 = k;
                                    let _e101 = cur[_e99];
                                    let _e104 = strict[_e101.arg0_];
                                    let _e105 = g;
                                    let _e109 = guards[((_e105 * GUARD_STRIDE) + 1u)];
                                    let _e111 = g;
                                    let _e115 = guards[((_e111 * GUARD_STRIDE) + 1u)];
                                    shape_s = (_e98 && ((_e104 & _e109) == _e115));
                                    let _e126 = shape_l;
                                    let _e127 = k;
                                    let _e129 = cur[_e127];
                                    let _e132 = loose[_e129.arg0_];
                                    let _e133 = g;
                                    let _e137 = guards[((_e133 * GUARD_STRIDE) + 1u)];
                                    let _e139 = g;
                                    let _e143 = guards[((_e139 * GUARD_STRIDE) + 1u)];
                                    shape_l = (_e126 && ((_e132 & _e137) == _e143));
                                }
                                if (_e87 >= 2u) {
                                    let _e156 = shape_s;
                                    let _e157 = k;
                                    let _e159 = cur[_e157];
                                    let _e162 = strict[_e159.arg1_];
                                    let _e163 = g;
                                    let _e167 = guards[((_e163 * GUARD_STRIDE) + 2u)];
                                    let _e169 = g;
                                    let _e173 = guards[((_e169 * GUARD_STRIDE) + 2u)];
                                    shape_s = (_e156 && ((_e162 & _e167) == _e173));
                                    let _e184 = shape_l;
                                    let _e185 = k;
                                    let _e187 = cur[_e185];
                                    let _e190 = loose[_e187.arg1_];
                                    let _e191 = g;
                                    let _e195 = guards[((_e191 * GUARD_STRIDE) + 2u)];
                                    let _e197 = g;
                                    let _e201 = guards[((_e197 * GUARD_STRIDE) + 2u)];
                                    shape_l = (_e184 && ((_e190 & _e195) == _e201));
                                }
                                let _e207 = k;
                                let _e209 = cur[_e207];
                                if (_e209.op == OP_NUM) {
                                    let _e220 = shape_s;
                                    let _e221 = k;
                                    let _e223 = caux[_e221];
                                    let _e225 = g;
                                    let _e229 = guards[((_e225 * GUARD_STRIDE) + 6u)];
                                    let _e231 = g;
                                    let _e235 = guards[((_e231 * GUARD_STRIDE) + 6u)];
                                    shape_s = (_e220 && (((_e223 >> 8u) & _e229) == _e235));
                                    let _e246 = shape_l;
                                    let _e247 = k;
                                    let _e249 = caux[_e247];
                                    let _e251 = g;
                                    let _e255 = guards[((_e251 * GUARD_STRIDE) + 6u)];
                                    let _e257 = g;
                                    let _e261 = guards[((_e257 * GUARD_STRIDE) + 6u)];
                                    shape_l = (_e246 && (((_e249 >> 8u) & _e255) == _e261));
                                }
                            }
                        }
                        let _e270 = shape_l;
                        let _e271 = l;
                        let _e272 = g;
                        let _e276 = guards[((_e272 * GUARD_STRIDE) + 3u)];
                        let _e278 = g;
                        let _e282 = guards[((_e278 * GUARD_STRIDE) + 3u)];
                        if (_e270 && ((_e271 & _e276) == _e282)) {
                            let _e290 = l;
                            let _e291 = g;
                            let _e295 = guards[((_e291 * GUARD_STRIDE) + 4u)];
                            l = (_e290 | _e295);
                        }
                        let _e305 = g;
                        let _e309 = guards[((_e305 * GUARD_STRIDE) + 5u)];
                        let _e311 = shape_s;
                        let _e313 = s;
                        let _e314 = g;
                        let _e318 = guards[((_e314 * GUARD_STRIDE) + 3u)];
                        let _e320 = g;
                        let _e324 = guards[((_e320 * GUARD_STRIDE) + 3u)];
                        if (((_e309 == 0u) && _e311) && ((_e313 & _e318) == _e324)) {
                            let _e332 = s;
                            let _e333 = g;
                            let _e337 = guards[((_e333 * GUARD_STRIDE) + 4u)];
                            s = (_e332 | _e337);
                        }
                    }
                    continuing {
                        let _e341 = g;
                        g = (_e341 + 1u);
                    }
                }
            }
            continuing {
                let _e345 = it;
                it = (_e345 + 1u);
            }
        }
        let _e350 = k;
        let _e352 = s;
        strict[_e350] = _e352;
        let _e356 = k;
        let _e358 = l;
        loose[_e356] = _e358;
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
        head_1 = (_e17 + 1u);
        let _e22 = head_1;
        let _e24 = qa[_e22];
        let _e25 = head_1;
        let _e27 = qb[_e25];
        if (_e24 == _e27) {
            continue;
        }
        let _e33 = head_1;
        let _e35 = qa[_e33];
        let _e37 = cur[_e35];
        let _e39 = head_1;
        let _e41 = qb[_e39];
        let _e43 = cur[_e41];
        if (_e37.op != _e43.op) {
            return false;
        }
        let _e50 = head_1;
        let _e52 = qa[_e50];
        let _e54 = cur[_e52];
        let _e56 = arity(_e54.op);
        if (_e56 == 0u) {
            let _e63 = head_1;
            let _e65 = qa[_e63];
            let _e67 = cur[_e65];
            if (_e67.op == OP_NUM) {
                let _e74 = head_1;
                let _e76 = qa[_e74];
                let _e78 = cur[_e76];
                let _e80 = head_1;
                let _e82 = qb[_e80];
                let _e84 = cur[_e82];
                if (_e78.arg1_ != _e84.arg1_) {
                    return false;
                }
            } else {
                let _e92 = head_1;
                let _e94 = qa[_e92];
                let _e96 = cur[_e94];
                let _e98 = head_1;
                let _e100 = qb[_e98];
                let _e102 = cur[_e100];
                if (_e96.arg0_ != _e102.arg0_) {
                    return false;
                }
            }
        } else {
            let _e108 = tail_1;
            if ((_e108 + _e56) > WORK) {
                return false;
            }
            let _e116 = tail_1;
            let _e118 = head_1;
            let _e120 = qa[_e118];
            let _e122 = cur[_e120];
            qa[_e116] = _e122.arg0_;
            let _e128 = tail_1;
            let _e130 = head_1;
            let _e132 = qb[_e130];
            let _e134 = cur[_e132];
            qb[_e128] = _e134.arg0_;
            let _e138 = tail_1;
            tail_1 = (_e138 + 1u);
            if (_e56 == 2u) {
                let _e146 = tail_1;
                let _e148 = head_1;
                let _e150 = qa[_e148];
                let _e152 = cur[_e150];
                qa[_e146] = _e152.arg1_;
                let _e158 = tail_1;
                let _e160 = head_1;
                let _e162 = qb[_e160];
                let _e164 = cur[_e162];
                qb[_e158] = _e164.arg1_;
                let _e168 = tail_1;
                tail_1 = (_e168 + 1u);
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
    loop {
        let _e36 = slot;
        if (_e36 < HEAP) {
        } else {
            break;
        }
        {
            let _e49 = slot;
            let _e53 = rules[(((r_1 * RULE_STRIDE) + 6u) + (_e49 * 4u))];
            let _e55 = slot;
            let _e57 = at[_e55];
            if ((_e53 == KIND_UNUSED) || (_e57 == NONE)) {
                continue;
            }
            let _e69 = slot;
            let _e73 = rules[(((r_1 * RULE_STRIDE) + 6u) + (_e69 * 4u))];
            if (_e73 == KIND_OP) {
                let _e84 = slot;
                let _e86 = at[_e84];
                let _e88 = cur[_e86];
                let _e92 = slot;
                let _e97 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e92 * 4u)) + 1u)];
                if (_e88.op != _e97) {
                    return 0u;
                }
                let _e103 = slot;
                let _e105 = at[_e103];
                let _e107 = cur[_e105];
                let _e109 = arity(_e107.op);
                if (_e109 >= 1u) {
                    let _e117 = slot;
                    let _e121 = slot;
                    let _e123 = at[_e121];
                    let _e125 = cur[_e123];
                    at[((2u * _e117) + 1u)] = _e125.arg0_;
                }
                if (_e109 >= 2u) {
                    let _e133 = slot;
                    let _e137 = slot;
                    let _e139 = at[_e137];
                    let _e141 = cur[_e139];
                    at[((2u * _e133) + 2u)] = _e141.arg1_;
                }
            } else {
                let _e152 = slot;
                let _e156 = rules[(((r_1 * RULE_STRIDE) + 6u) + (_e152 * 4u))];
                if (_e156 == KIND_MV) {
                    let _e169 = slot;
                    let _e174 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e169 * 4u)) + 1u)];
                    let _e176 = bind[_e174];
                    if (_e176 == NONE) {
                        let _e189 = slot;
                        let _e194 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e189 * 4u)) + 1u)];
                        let _e196 = slot;
                        let _e198 = at[_e196];
                        bind[_e194] = _e198;
                        let _e208 = slot;
                        let _e210 = at[_e208];
                        let _e212 = strict[_e210];
                        let _e215 = slot;
                        let _e220 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e215 * 4u)) + 2u)];
                        let _e224 = slot;
                        let _e229 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e224 * 4u)) + 2u)];
                        if ((_e212 & _e220) != _e229) {
                            let _e241 = slot;
                            let _e243 = at[_e241];
                            let _e245 = loose[_e243];
                            let _e248 = slot;
                            let _e253 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e248 * 4u)) + 2u)];
                            let _e257 = slot;
                            let _e262 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e257 * 4u)) + 2u)];
                            if (use_loose && ((_e245 & _e253) == _e262)) {
                                needed_loose = true;
                            } else {
                                return 0u;
                            }
                        }
                    } else {
                        let _e278 = slot;
                        let _e283 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e278 * 4u)) + 1u)];
                        let _e285 = bind[_e283];
                        let _e288 = slot;
                        let _e290 = at[_e288];
                        let _e291 = same_subtree(_e285, _e290);
                        if !(_e291) {
                            return 0u;
                        }
                    }
                } else {
                    let _e298 = slot;
                    let _e300 = at[_e298];
                    let _e302 = cur[_e300];
                    if (_e302.op != OP_NUM) {
                        return 0u;
                    }
                    let _e316 = slot;
                    let _e321 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e316 * 4u)) + 3u)];
                    if (_e321 != 0u) {
                        let _e333 = slot;
                        let _e335 = at[_e333];
                        let _e337 = caux[_e335];
                        let _e341 = slot;
                        let _e346 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e341 * 4u)) + 3u)];
                        if ((_e337 & 255u) != _e346) {
                            return 0u;
                        }
                    } else {
                        let _e359 = slot;
                        let _e361 = at[_e359];
                        let _e363 = caux[_e361];
                        let _e367 = slot;
                        let _e372 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e367 * 4u)) + 2u)];
                        let _e376 = slot;
                        let _e381 = rules[((((r_1 * RULE_STRIDE) + 6u) + (_e376 * 4u)) + 2u)];
                        if (((_e363 >> 8u) & _e372) != _e381) {
                            return 0u;
                        }
                    }
                }
            }
        }
        continuing {
            let _e386 = slot;
            slot = (_e386 + 1u);
        }
    }
    let _e389 = needed_loose;
    if _e389 {
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

    let _e4 = n_cur;
    n = _e4;
    loop {
        let _e7 = slot_1;
        if (_e7 < HEAP) {
        } else {
            break;
        }
        {
            let _e12 = slot_1;
            home[_e12] = NONE;
            let _e26 = slot_1;
            let _e30 = rules[((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e26 * 4u))];
            let _e35 = slot_1;
            let _e39 = rules[((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e35 * 4u))];
            if ((_e30 == KIND_OP) || (_e39 == KIND_NUM)) {
                let _e45 = slot_1;
                let _e47 = n;
                home[_e45] = _e47;
                let _e50 = n;
                n = (_e50 + 1u);
            } else {
                let _e63 = slot_1;
                let _e67 = rules[((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e63 * 4u))];
                if (_e67 == KIND_MV) {
                    let _e79 = slot_1;
                    let _e84 = slot_1;
                    let _e89 = rules[(((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e84 * 4u)) + 1u)];
                    let _e91 = bind[_e89];
                    home[_e79] = _e91;
                }
            }
        }
        continuing {
            let _e94 = slot_1;
            slot_1 = (_e94 + 1u);
        }
    }
    loop {
        let _e98 = slot_2;
        if (_e98 < HEAP) {
        } else {
            break;
        }
        {
            let _e111 = slot_2;
            let _e115 = rules[((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e111 * 4u))];
            if (_e115 == KIND_OP) {
                let _e128 = slot_2;
                let _e133 = rules[(((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e128 * 4u)) + 1u)];
                let _e134 = arity(_e133);
                a0_1 = 0u;
                a1_1 = 0u;
                if (_e134 >= 1u) {
                    let _e146 = slot_2;
                    let _e150 = home[((2u * _e146) + 1u)];
                    a0_1 = _e150;
                }
                if (_e134 >= 2u) {
                    let _e157 = slot_2;
                    let _e161 = home[((2u * _e157) + 2u)];
                    a1_1 = _e161;
                }
                let _e175 = slot_2;
                let _e177 = home[_e175];
                let _e182 = slot_2;
                let _e187 = rules[(((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e182 * 4u)) + 1u)];
                let _e188 = a0_1;
                let _e189 = a1_1;
                cur[_e177] = Node(_e187, _e188, _e189, 0f);
                let _e195 = slot_2;
                let _e197 = home[_e195];
                caux[_e197] = 0u;
            } else {
                let _e210 = slot_2;
                let _e214 = rules[((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e210 * 4u))];
                if (_e214 == KIND_NUM) {
                    let _e230 = slot_2;
                    let _e232 = home[_e230];
                    let _e237 = slot_2;
                    let _e242 = rules[(((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e237 * 4u)) + 1u)];
                    let _e246 = pool[((_e242 * 3u) + 1u)];
                    let _e250 = slot_2;
                    let _e255 = rules[(((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e250 * 4u)) + 1u)];
                    let _e258 = pool[(_e255 * 3u)];
                    cur[_e232] = Node(OP_NUM, 0u, _e246, bitcast<f32>(_e258));
                    let _e274 = slot_2;
                    let _e276 = home[_e274];
                    let _e281 = slot_2;
                    let _e286 = rules[(((((r_2 * RULE_STRIDE) + 6u) + 60u) + (_e281 * 4u)) + 1u)];
                    let _e290 = pool[((_e286 * 3u) + 2u)];
                    caux[_e276] = _e290;
                }
            }
        }
        continuing {
            let _e293 = slot_2;
            slot_2 = (_e293 + 1u);
        }
    }
    let _e299 = home[0];
    splice(pos_3, _e299);
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
    let _e14 = cfg.n_expr;
    if (((gid.y * _e9) + gid.x) >= _e14) {
        return;
    }
    let _e22 = cfg.stride;
    let _e27 = len_in[((gid.y * _e22) + gid.x)];
    n_cur = _e27;
    let _e31 = n_cur;
    let _e33 = n_cur;
    if ((_e31 == 0u) || (_e33 > SLOT)) {
        let _e43 = cfg.stride;
        info_out[(((gid.y * _e43) + gid.x) * 4u)] = 0u;
        let _e57 = cfg.stride;
        info_out[((((gid.y * _e57) + gid.x) * 4u) + 1u)] = 0u;
        let _e72 = cfg.stride;
        info_out[((((gid.y * _e72) + gid.x) * 4u) + 2u)] = 0u;
        let _e87 = cfg.stride;
        info_out[((((gid.y * _e87) + gid.x) * 4u) + 3u)] = 1u;
        return;
    }
    loop {
        let _e96 = i;
        let _e97 = n_cur;
        if (_e96 < _e97) {
        } else {
            break;
        }
        {
            let _e105 = i;
            let _e109 = cfg.stride;
            let _e114 = i;
            let _e117 = nodes_in[((((gid.y * _e109) + gid.x) * SLOT) + _e114)];
            cur[_e105] = _e117;
            let _e124 = i;
            let _e128 = cfg.stride;
            let _e133 = i;
            let _e136 = aux_in[((((gid.y * _e128) + gid.x) * SLOT) + _e133)];
            caux[_e124] = _e136;
        }
        continuing {
            let _e139 = i;
            i = (_e139 + 1u);
        }
    }
    loop {
        let _e143 = round;
        let _e145 = cfg.rounds;
        if (_e143 < _e145) {
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
                let _e159 = order;
                let _e161 = hit_rule;
                if ((_e159 < 2u) && (_e161 == NONE)) {
                } else {
                    break;
                }
                {
                    pos = 0u;
                    loop {
                        let _e170 = pos;
                        let _e171 = n_cur;
                        let _e173 = hit_rule;
                        if ((_e170 < _e171) && (_e173 == NONE)) {
                        } else {
                            break;
                        }
                        {
                            let _e180 = pos;
                            let _e183 = cur[_e180].op;
                            let _e185 = bucket[_e183];
                            r = _e185;
                            loop {
                                let _e191 = r;
                                let _e192 = pos;
                                let _e195 = cur[_e192].op;
                                let _e198 = bucket[(_e195 + 1u)];
                                if (_e191 < _e198) {
                                } else {
                                    break;
                                }
                                {
                                    let _e207 = r;
                                    let _e211 = rules[((_e207 * RULE_STRIDE) + 4u)];
                                    let _e212 = order;
                                    let _e214 = r;
                                    let _e218 = rules[((_e214 * RULE_STRIDE) + 5u)];
                                    let _e220 = cfg.admit;
                                    if ((_e211 != _e212) || (_e218 > _e220)) {
                                        continue;
                                    }
                                    let _e224 = r;
                                    let _e226 = pos;
                                    let _e230 = cfg.admit;
                                    let _e232 = match_at(_e224, _e226, (_e230 == 2u));
                                    if (_e232 != 0u) {
                                        let _e237 = r;
                                        hit_rule = _e237;
                                        let _e240 = pos;
                                        hit_pos = _e240;
                                        hit_kind = _e232;
                                        break;
                                    }
                                }
                                continuing {
                                    let _e244 = r;
                                    r = (_e244 + 1u);
                                }
                            }
                        }
                        continuing {
                            let _e248 = pos;
                            pos = (_e248 + 1u);
                        }
                    }
                }
                continuing {
                    let _e252 = order;
                    order = (_e252 + 1u);
                }
            }
            let _e256 = hit_rule;
            if (_e256 == NONE) {
                break;
            }
            let _e263 = hit_rule;
            let _e267 = rules[((_e263 * RULE_STRIDE) + 5u)];
            used = _e267;
            let _e270 = hit_kind;
            if (_e270 == 2u) {
                used = 2u;
            }
            let _e276 = level;
            let _e277 = used;
            level = max(_e276, _e277);
            let _e280 = hit_rule;
            let _e282 = hit_pos;
            apply(_e280, _e282);
            let _e285 = steps;
            steps = (_e285 + 1u);
        }
        continuing {
            let _e289 = round;
            round = (_e289 + 1u);
        }
    }
    loop {
        let _e293 = i_1;
        let _e294 = n_cur;
        if (_e293 < _e294) {
        } else {
            break;
        }
        {
            let _e304 = cfg.stride;
            let _e309 = i_1;
            let _e312 = i_1;
            let _e314 = cur[_e312];
            nodes_out[((((gid.y * _e304) + gid.x) * SLOT) + _e309)] = _e314;
        }
        continuing {
            let _e317 = i_1;
            i_1 = (_e317 + 1u);
        }
    }
    let _e326 = cfg.stride;
    let _e332 = n_cur;
    info_out[(((gid.y * _e326) + gid.x) * 4u)] = _e332;
    let _e341 = cfg.stride;
    let _e348 = steps;
    info_out[((((gid.y * _e341) + gid.x) * 4u) + 1u)] = _e348;
    let _e357 = cfg.stride;
    let _e364 = level;
    info_out[((((gid.y * _e357) + gid.x) * 4u) + 2u)] = _e364;
    let _e373 = cfg.stride;
    info_out[((((gid.y * _e373) + gid.x) * 4u) + 3u)] = 0u;
    return;
}
