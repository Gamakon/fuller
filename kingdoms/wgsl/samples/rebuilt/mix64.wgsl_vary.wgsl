struct Gen {
    pop: u32,
    n_genes: u32,
    head: u32,
    tail: u32,
    n_rnc: u32,
    n_functions: u32,
    n_terminals: u32,
    n_islands: u32,
    seed: u32,
    generation: u32,
    rnc_lo: i32,
    rnc_span: u32,
    cohort_merge: u32,
    rnc_id: u32,
    vhead: u32,
    typed_depth: u32,
    n_flat: u32,
}

struct Island {
    lo: u32,
    hi: u32,
    elites: u32,
    tournsize: u32,
    arrivals: u32,
    arrival_children: u32,
    open_fight: u32,
    cohort_merge: u32,
    mut_point: u32,
    invert: u32,
    is_transpose: u32,
    ris_transpose: u32,
    gene_transpose: u32,
    dc_point: u32,
    invert_dc: u32,
    transpose_dc: u32,
    rnc_point: u32,
    cx_one_point: u32,
    cx_two_point: u32,
    cx_gene: u32,
    cleanse: u32,
    cleanse_collapse: u32,
}

const MAX_HT: u32 = 512u;
const STREAM_SELECT_1_: u32 = 6u;
const STREAM_SELECT_2_: u32 = 7u;
const STREAM_MUT_HIT: u32 = 8u;
const STREAM_MUT_KIND: u32 = 9u;
const STREAM_MUT_SYMBOL: u32 = 10u;
const STREAM_OPERATOR: u32 = 11u;
const STREAM_INVERT: u32 = 12u;
const STREAM_IS: u32 = 13u;
const STREAM_RIS: u32 = 14u;
const STREAM_GENE_T: u32 = 15u;
const STREAM_DC_HIT: u32 = 16u;
const STREAM_DC_VALUE: u32 = 17u;
const STREAM_INVERT_DC: u32 = 18u;
const STREAM_TRANSPOSE_DC: u32 = 19u;
const STREAM_RNC_HIT: u32 = 20u;
const STREAM_RNC_VALUE: u32 = 21u;
const STREAM_CX_1P: u32 = 22u;
const STREAM_CX_2P: u32 = 23u;
const STREAM_CX_GENE: u32 = 24u;
const STREAM_CLEANSE: u32 = 25u;
const STREAM_ARRIVAL_MATE: u32 = 26u;
const OP_INVERT: u32 = 0u;
const OP_IS: u32 = 1u;
const OP_RIS: u32 = 2u;
const OP_GENE_T: u32 = 3u;
const OP_INVERT_DC: u32 = 4u;
const OP_TRANSPOSE_DC: u32 = 5u;
const OP_CX_1P: u32 = 6u;
const OP_CX_2P: u32 = 7u;
const OP_CX_GENE: u32 = 8u;
const OP_CLEANSE: u32 = 9u;
const NONE: u32 = 4294967295u;
const F32_MAX: f32 = 340282350000000000000000000000000000000f;
const NAN_BITS: u32 = 2143289344u;
const DOMAIN_LO: u32 = 2042731284u;
const DOMAIN_HI: u32 = 1804002330u;
const ELDERS: u32 = 4294967295u;
const MAX_CLEANSE: u32 = 512u;
const LEAF: u32 = 4294967294u;

var<private> mut_budget: array<u32, 512>;
@group(0) @binding(0) 
var<uniform> gp: Gen;
@group(0) @binding(1) 
var<storage> sample_functions: array<u32>;
@group(0) @binding(2) 
var<storage> sample_terminals: array<u32>;
@group(0) @binding(3) 
var<storage> arity: array<u32>;
@group(0) @binding(4) 
var<storage> islands: array<Island>;
@group(0) @binding(5) 
var<storage> genome_now: array<u32>;
@group(0) @binding(6) 
var<storage> rnc_now: array<f32>;
@group(0) @binding(7) 
var<storage> fitness_now: array<f32>;
@group(0) @binding(8) 
var<storage> wrapper_now: array<u32>;
@group(0) @binding(9) 
var<storage, read_write> parent: array<u32>;
@group(0) @binding(10) 
var<storage, read_write> genome: array<u32>;
@group(0) @binding(11) 
var<storage, read_write> rnc: array<f32>;
@group(0) @binding(12) 
var<storage, read_write> fitness: array<f32>;
@group(0) @binding(13) 
var<storage, read_write> wrapper_id: array<u32>;
@group(0) @binding(14) 
var<storage, read_write> stage1_: array<u32>;
@group(0) @binding(15) 
var<storage> cohort_now: array<u32>;
@group(0) @binding(16) 
var<storage, read_write> cohort: array<u32>;
@group(0) @binding(17) 
var<storage> sample_flat: array<u32>;
@group(0) @binding(18) 
var<storage> depth_cost: array<u32>;
@group(0) @binding(19) 
var<storage> elite_src: array<u32>;
@group(0) @binding(20) 
var<storage> lane_seed: array<u32>;
@group(0) @binding(21) 
var<storage> lane_base: array<u32>;
var<private> segment: array<u32, 512>;
var<private> before: array<u32, 512>;
var<private> cl_child: array<u32, 512>;
var<private> cl_ordinal: array<u32, 512>;
var<private> cl_queue: array<u32, 512>;
var<private> cl_tok: array<u32, 512>;
var<private> cl_dc: array<u32, 512>;

fn mul32x32_64_(a: u32, b: u32) -> vec2<u32> {
    let a0_ = (a & 65535u);
    let a1_ = (a >> 16u);
    let b0_ = (b & 65535u);
    let b1_ = (b >> 16u);
    let p00_ = (a0_ * b0_);
    let p01_ = (a0_ * b1_);
    let p10_ = (a1_ * b0_);
    let p11_ = (a1_ * b1_);
    let mid = (p01_ + p10_);
    let mid_carry = select(0u, 1u, (mid < p01_));
    let mid_lo = (mid << 16u);
    let mid_hi = ((mid >> 16u) | (mid_carry << 16u));
    let lo_1 = (p00_ + mid_lo);
    let lo_carry = select(0u, 1u, (lo_1 < p00_));
    let hi_1 = ((p11_ + mid_hi) + lo_carry);
    return vec2<u32>(lo_1, hi_1);
}

fn mul64_lo(a_1: vec2<u32>, b_1: vec2<u32>) -> vec2<u32> {
    let _e4 = mul32x32_64_(a_1.x, b_1.x);
    return vec2<u32>(_e4.x, ((_e4.y + (a_1.x * b_1.y)) + (a_1.y * b_1.x)));
}

fn shr64_(v: vec2<u32>, s: u32) -> vec2<u32> {
    return vec2<u32>(((v.x >> s) | (v.y << (32u - s))), (v.y >> s));
}

fn mix64p(x_in: vec2<u32>) -> vec2<u32> {
    var x: vec2<u32>;

    x = x_in;
    let _e3 = x;
    let _e5 = shr64_(_e3, 30u);
    let _e7 = x;
    x = (_e7 ^ _e5);
    let _e10 = x;
    let _e14 = mul64_lo(_e10, vec2<u32>(484763065u, 3210233709u));
    x = _e14;
    let _e17 = x;
    let _e19 = shr64_(_e17, 27u);
    let _e21 = x;
    x = (_e21 ^ _e19);
    let _e24 = x;
    let _e28 = mul64_lo(_e24, vec2<u32>(321982955u, 2496678331u));
    x = _e28;
    let _e31 = x;
    let _e33 = shr64_(_e31, 31u);
    let _e35 = x;
    x = (_e35 ^ _e33);
    let _e38 = x;
    return _e38;
}

fn mix64_3p(domain: vec2<u32>, a_2: vec2<u32>, b_2: vec2<u32>, c: vec2<u32>) -> vec2<u32> {
    let _e3 = mix64p((domain ^ a_2));
    let _e6 = mix64p((_e3 ^ b_2));
    let _e9 = mix64p((_e6 ^ c));
    return _e9;
}

fn mix64_modp(domain_1: vec2<u32>, a_3: vec2<u32>, b_3: vec2<u32>, c_1: vec2<u32>, n: u32) -> u32 {
    let _e4 = mix64_3p(domain_1, a_3, b_3, c_1);
    let _e7 = mul32x32_64_(_e4.x, n);
    let _e10 = mul32x32_64_(_e4.y, n);
    let mid_1 = (_e7.y + _e10.x);
    let carry = select(0u, 1u, (mid_1 < _e7.y));
    return (_e10.y + carry);
}

fn hash(row: u32, slot: u32, stream: u32) -> vec2<u32> {
    let _e6 = lane_seed[row];
    let _e13 = gp.generation;
    let _e19 = lane_base[row];
    let _e25 = mix64_3p(vec2<u32>((DOMAIN_LO ^ _e6), (DOMAIN_HI ^ stream)), vec2<u32>(_e13, 0u), vec2<u32>((row - _e19), 0u), vec2<u32>(slot, 0u));
    return _e25;
}

fn below(row_1: u32, slot_1: u32, stream_1: u32, n_1: u32) -> u32 {
    let _e6 = lane_seed[row_1];
    let _e13 = gp.generation;
    let _e19 = lane_base[row_1];
    let _e26 = mix64_modp(vec2<u32>((DOMAIN_LO ^ _e6), (DOMAIN_HI ^ stream_1)), vec2<u32>(_e13, 0u), vec2<u32>((row_1 - _e19), 0u), vec2<u32>(slot_1, 0u), n_1);
    return _e26;
}

fn coin(row_2: u32, slot_2: u32, stream_2: u32) -> bool {
    let _e3 = hash(row_2, slot_2, stream_2);
    return ((_e3.y >> 31u) == 1u);
}

fn chance(row_3: u32, slot_3: u32, stream_3: u32, thr: u32) -> bool {
    let _e3 = hash(row_3, slot_3, stream_3);
    return ((thr == NONE) || (_e3.y < thr));
}

fn band_of(row_4: u32, merge: u32) -> u32 {
    var age: u32 = 0u;

    let label = cohort_now[row_4];
    let _e7 = gp.generation;
    if (_e7 > label) {
        let _e12 = gp.generation;
        age = (_e12 - label);
    }
    let _e16 = age;
    if (_e16 >= merge) {
        return ELDERS;
    }
    return label;
}

fn same_cohort(a_4: u32, b_4: u32, merge_1: u32) -> bool {
    if (merge_1 == 0u) {
        return true;
    }
    let _e6 = band_of(a_4, merge_1);
    let _e9 = band_of(b_4, merge_1);
    return (_e6 == _e9);
}

fn key(row_5: u32) -> f32 {
    let f = fitness_now[row_5];
    if ((bitcast<u32>(f) & 2147483647u) > 2139095040u) {
        return F32_MAX;
    }
    return min(f, F32_MAX);
}

fn better_mate(me: u32, c_2: u32, w_2: u32, open_fight: bool, merge_2: u32) -> bool {
    if !(open_fight) {
        let _e5 = same_cohort(me, c_2, merge_2);
        let _e9 = same_cohort(me, w_2, merge_2);
        if (_e5 != _e9) {
            return _e5;
        }
    }
    let _e12 = key(c_2);
    let _e14 = key(w_2);
    return (_e12 < _e14);
}

fn island_of(row_6: u32) -> Island {
    var i_13: u32 = 0u;

    loop {
        let _e3 = i_13;
        let _e5 = gp.n_islands;
        if (_e3 < _e5) {
        } else {
            break;
        }
        {
            let _e10 = i_13;
            let _e13 = islands[_e10].hi;
            if (row_6 < _e13) {
                let _e17 = i_13;
                let _e19 = islands[_e17];
                return _e19;
            }
        }
        continuing {
            let _e22 = i_13;
            i_13 = (_e22 + 1u);
        }
    }
    let _e28 = gp.n_islands;
    let _e31 = islands[(_e28 - 1u)];
    return _e31;
}

fn at(base: u32, g_2: u32, pos_9: u32) -> u32 {
    let _e6 = gp.head;
    let _e8 = gp.tail;
    return ((base + (g_2 * (_e6 + (2u * _e8)))) + pos_9);
}

fn reverse(lo: u32, hi: u32) {
    var a_5: u32;
    var b_5: u32;

    a_5 = lo;
    b_5 = hi;
    loop {
        let _e7 = a_5;
        let _e9 = b_5;
        if ((_e7 + 1u) >= _e9) {
            break;
        }
        let _e13 = b_5;
        b_5 = (_e13 - 1u);
        let _e17 = a_5;
        let held = genome[_e17];
        let _e23 = a_5;
        let _e25 = b_5;
        let _e27 = genome[_e25];
        genome[_e23] = _e27;
        let _e30 = b_5;
        genome[_e30] = held;
        let _e34 = a_5;
        a_5 = (_e34 + 1u);
    }
    return;
}

fn cleanse_gene(row_7: u32, gene: u32, collapse_thr: u32) {
    var need: i32 = 1i;
    var n_2: u32 = 0u;
    var next_child_1: u32 = 1u;
    var n_rnc: u32 = 0u;
    var n_fn_1: u32 = 0u;
    var i_14: u32 = 0u;
    var p: u32 = 0u;
    var seen_1: u32 = 0u;
    var i_15: u32 = 0u;
    var q_1: u32 = LEAF;
    var head: u32 = 0u;
    var tail: u32 = 1u;
    var m: u32 = 0u;
    var k_3: u32 = 0u;
    var j: u32;
    var x_1: u32;
    var x_2: u32 = 0u;
    var x_3: u32 = 0u;

    let h = gp.vhead;
    let _e21 = gp.head;
    let _e23 = gp.tail;
    let ht = (_e21 + _e23);
    if (ht > MAX_CLEANSE) {
        return;
    }
    loop {
        let _e30 = need;
        let _e32 = n_2;
        if ((_e30 <= 0i) || (_e32 >= ht)) {
            break;
        }
        let _e41 = need;
        let _e42 = n_2;
        let _e45 = genome[(gene + _e42)];
        let _e47 = arity[_e45];
        need = ((_e41 + i32(_e47)) - 1i);
        let _e53 = n_2;
        n_2 = (_e53 + 1u);
    }
    let _e57 = need;
    if (_e57 > 0i) {
        return;
    }
    loop {
        let _e61 = i_14;
        let _e62 = n_2;
        if (_e61 < _e62) {
        } else {
            break;
        }
        {
            let _e67 = i_14;
            let tok = genome[(gene + _e67)];
            let _e74 = i_14;
            let _e76 = next_child_1;
            cl_child[_e74] = _e76;
            let _e79 = next_child_1;
            let _e81 = arity[tok];
            next_child_1 = (_e79 + _e81);
            let _e86 = i_14;
            cl_ordinal[_e86] = NONE;
            let _e91 = gp.rnc_id;
            let _e94 = gp.rnc_id;
            if ((_e91 != NONE) && (tok == _e94)) {
                let _e100 = i_14;
                let _e102 = n_rnc;
                cl_ordinal[_e100] = _e102;
                let _e105 = n_rnc;
                n_rnc = (_e105 + 1u);
            }
            let _e110 = arity[tok];
            if (_e110 > 0u) {
                let _e114 = n_fn_1;
                n_fn_1 = (_e114 + 1u);
            }
        }
        continuing {
            let _e118 = i_14;
            i_14 = (_e118 + 1u);
        }
    }
    let _e122 = n_fn_1;
    if (_e122 == 0u) {
        return;
    }
    let _e128 = n_fn_1;
    let _e129 = below(row_7, 1u, STREAM_CLEANSE, _e128);
    loop {
        let _e132 = i_15;
        let _e133 = n_2;
        if (_e132 < _e133) {
        } else {
            break;
        }
        {
            let _e140 = i_15;
            let _e143 = genome[(gene + _e140)];
            let _e145 = arity[_e143];
            if (_e145 > 0u) {
                let _e148 = seen_1;
                if (_e148 == _e129) {
                    let _e152 = i_15;
                    p = _e152;
                }
                let _e155 = seen_1;
                seen_1 = (_e155 + 1u);
            }
        }
        continuing {
            let _e159 = i_15;
            i_15 = (_e159 + 1u);
        }
    }
    let _e165 = chance(row_7, 2u, STREAM_CLEANSE, collapse_thr);
    let _e169 = gp.rnc_id;
    let collapse = ((_e169 != NONE) && _e165);
    if !(collapse) {
        let _e180 = p;
        let _e183 = genome[(gene + _e180)];
        let _e185 = arity[_e183];
        let _e186 = below(row_7, 3u, STREAM_CLEANSE, _e185);
        let _e190 = p;
        let _e192 = cl_child[_e190];
        q_1 = (_e192 + _e186);
    }
    let _e199 = gp.n_rnc;
    let _e200 = below(row_7, 4u, STREAM_CLEANSE, _e199);
    let _e206 = q_1;
    let _e207 = p;
    cl_queue[0] = select(0u, _e206, (_e207 == 0u));
    loop {
        let _e212 = head;
        let _e213 = tail;
        if (_e212 >= _e213) {
            break;
        }
        let _e217 = head;
        let i_16 = cl_queue[_e217];
        let _e222 = head;
        head = (_e222 + 1u);
        if (i_16 == LEAF) {
            let _e229 = m;
            let _e232 = gp.rnc_id;
            cl_tok[_e229] = _e232;
            let _e235 = k_3;
            cl_dc[_e235] = _e200;
            let _e239 = k_3;
            k_3 = (_e239 + 1u);
            let _e243 = m;
            m = (_e243 + 1u);
            continue;
        }
        let tok_1 = genome[(gene + i_16)];
        let _e252 = m;
        cl_tok[_e252] = tok_1;
        let _e257 = cl_ordinal[i_16];
        if (_e257 != NONE) {
            let _e266 = k_3;
            let _e270 = cl_ordinal[i_16];
            let _e273 = genome[((gene + ht) + _e270)];
            let _e275 = cl_ordinal[i_16];
            let _e277 = gp.tail;
            cl_dc[_e266] = select(0u, _e273, (_e275 < _e277));
            let _e282 = k_3;
            k_3 = (_e282 + 1u);
        }
        let _e286 = m;
        m = (_e286 + 1u);
        j = 0u;
        loop {
            let _e292 = j;
            let _e294 = arity[tok_1];
            if (_e292 < _e294) {
            } else {
                break;
            }
            {
                let _e299 = cl_child[i_16];
                let _e300 = j;
                let c_3 = (_e299 + _e300);
                let _e306 = tail;
                let _e308 = q_1;
                let _e309 = p;
                cl_queue[_e306] = select(c_3, _e308, (c_3 == _e309));
                let _e314 = tail;
                tail = (_e314 + 1u);
            }
            continuing {
                let _e318 = j;
                j = (_e318 + 1u);
            }
        }
    }
    let _e322 = k_3;
    let _e324 = gp.tail;
    if (_e322 > _e324) {
        return;
    }
    x_1 = h;
    loop {
        let _e329 = x_1;
        let _e330 = m;
        if (_e329 < _e330) {
        } else {
            break;
        }
        {
            let _e336 = x_1;
            let _e338 = cl_tok[_e336];
            let _e340 = arity[_e338];
            if (_e340 > 0u) {
                return;
            }
        }
        continuing {
            let _e344 = x_1;
            x_1 = (_e344 + 1u);
        }
    }
    loop {
        let _e348 = x_2;
        let _e349 = m;
        if (_e348 < _e349) {
        } else {
            break;
        }
        {
            let _e355 = x_2;
            let _e358 = x_2;
            let _e360 = cl_tok[_e358];
            genome[(gene + _e355)] = _e360;
        }
        continuing {
            let _e363 = x_2;
            x_2 = (_e363 + 1u);
        }
    }
    loop {
        let _e367 = x_3;
        let _e368 = k_3;
        if (_e367 < _e368) {
        } else {
            break;
        }
        {
            let _e375 = x_3;
            let _e378 = x_3;
            let _e380 = cl_dc[_e378];
            genome[((gene + ht) + _e375)] = _e380;
        }
        continuing {
            let _e383 = x_3;
            x_3 = (_e383 + 1u);
        }
    }
    return;
}

fn swap_tokens(a_6: u32, b_6: u32, g_3: u32, from_pos: u32, to_pos: u32) {
    var pos_10: u32;

    let _e3 = gp.head;
    let _e5 = gp.tail;
    let width = (_e3 + (2u * _e5));
    let _e10 = gp.n_genes;
    let row_w = (_e10 * width);
    pos_10 = from_pos;
    loop {
        let _e16 = pos_10;
        if (_e16 < to_pos) {
        } else {
            break;
        }
        {
            let _e24 = pos_10;
            let ia = (((a_6 * row_w) + (g_3 * width)) + _e24);
            let _e32 = pos_10;
            let ib = (((b_6 * row_w) + (g_3 * width)) + _e32);
            let held_1 = genome[ia];
            let _e40 = genome[ib];
            genome[ia] = _e40;
            genome[ib] = held_1;
        }
        continuing {
            let _e45 = pos_10;
            pos_10 = (_e45 + 1u);
        }
    }
    return;
}

fn swap_consts(a_7: u32, ga: u32, b_7: u32, gb: u32) {
    var k_4: u32 = 0u;

    let _e3 = gp.n_genes;
    let _e5 = gp.n_rnc;
    let rnc_w = (_e3 * _e5);
    loop {
        let _e9 = k_4;
        let _e11 = gp.n_rnc;
        if (_e9 < _e11) {
        } else {
            break;
        }
        {
            let _e19 = gp.n_rnc;
            let _e22 = k_4;
            let ia_1 = (((a_7 * rnc_w) + (ga * _e19)) + _e22);
            let _e30 = gp.n_rnc;
            let _e33 = k_4;
            let ib_1 = (((b_7 * rnc_w) + (gb * _e30)) + _e33);
            let held_2 = rnc[ia_1];
            let _e41 = rnc[ib_1];
            rnc[ia_1] = _e41;
            rnc[ib_1] = held_2;
        }
        continuing {
            let _e46 = k_4;
            k_4 = (_e46 + 1u);
        }
    }
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn first_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var w: u32 = NONE;
    var i: u32 = 0u;

    let row_8 = gid.x;
    let _e6 = gp.pop;
    if (row_8 >= _e6) {
        return;
    }
    let _e8 = island_of(row_8);
    let open_fight_1 = (_e8.open_fight == 1u);
    let n_3 = (_e8.hi - _e8.lo);
    loop {
        let _e16 = i;
        if (_e16 < _e8.tournsize) {
        } else {
            break;
        }
        {
            let _e20 = i;
            let _e22 = below(row_8, _e20, STREAM_SELECT_1_, n_3);
            let c_4 = (_e8.lo + _e22);
            let _e26 = w;
            let _e28 = better_mate(row_8, c_4, _e26, open_fight_1, _e8.cohort_merge);
            let _e31 = w;
            if ((_e31 == NONE) || _e28) {
                w = c_4;
            }
        }
        continuing {
            let _e37 = i;
            i = (_e37 + 1u);
        }
    }
    let _e42 = w;
    stage1_[row_8] = _e42;
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn select_main(@builtin(global_invocation_id) gid_1: vec3<u32>) {
    var taken: array<u32, 64>;
    var best: u32 = NONE;
    var e: u32 = 0u;
    var r: u32;
    var used: bool;
    var q: u32;
    var w_1: u32 = NONE;
    var t: u32 = 0u;

    let row_9 = gid_1.x;
    let _e8 = gp.pop;
    if (row_9 >= _e8) {
        return;
    }
    let _e10 = island_of(row_9);
    let open_fight_2 = (_e10.open_fight == 1u);
    let n_4 = (_e10.hi - _e10.lo);
    if (row_9 < (_e10.lo + _e10.elites)) {
        let src = elite_src[row_9];
        if (src != NONE) {
            parent[row_9] = src;
            return;
        }
        let j_1 = (row_9 - _e10.lo);
        loop {
            let _e31 = e;
            if (_e31 <= j_1) {
            } else {
                break;
            }
            {
                best = NONE;
                r = _e10.lo;
                loop {
                    let _e38 = r;
                    if (_e38 < _e10.hi) {
                    } else {
                        break;
                    }
                    {
                        used = false;
                        q = 0u;
                        loop {
                            let _e47 = q;
                            let _e48 = e;
                            if (_e47 < _e48) {
                            } else {
                                break;
                            }
                            {
                                let _e53 = q;
                                let _e55 = taken[_e53];
                                let _e56 = r;
                                if (_e55 == _e56) {
                                    used = true;
                                }
                            }
                            continuing {
                                let _e62 = q;
                                q = (_e62 + 1u);
                            }
                        }
                        let _e65 = r;
                        let _e66 = key(_e65);
                        let _e68 = best;
                        let _e69 = key(_e68);
                        let _e73 = used;
                        let _e75 = best;
                        if (!(_e73) && ((_e75 == NONE) || (_e66 < _e69))) {
                            let _e82 = r;
                            best = _e82;
                        }
                    }
                    continuing {
                        let _e85 = r;
                        r = (_e85 + 1u);
                    }
                }
                let _e90 = e;
                let _e92 = best;
                taken[_e90] = _e92;
            }
            continuing {
                let _e95 = e;
                e = (_e95 + 1u);
            }
        }
        let _e100 = best;
        parent[row_9] = _e100;
        return;
    }
    let stride = (_e10.arrival_children + 1u);
    let band = (_e10.lo + _e10.elites);
    if ((((_e10.arrivals > 0u) && (_e10.arrival_children > 0u)) && (row_9 >= band)) && (row_9 < (band + (_e10.arrivals * stride)))) {
        if (((row_9 - band) % stride) == 0u) {
            parent[row_9] = row_9;
        } else {
            let _e129 = below(row_9, ((row_9 - band) % stride), STREAM_ARRIVAL_MATE, n_4);
            parent[row_9] = (_e10.lo + _e129);
        }
        return;
    }
    loop {
        let _e135 = t;
        if (_e135 < _e10.tournsize) {
        } else {
            break;
        }
        {
            let _e139 = t;
            let _e141 = below(row_9, _e139, STREAM_SELECT_2_, n_4);
            let c_5 = stage1_[(_e10.lo + _e141)];
            let _e148 = w_1;
            let _e150 = better_mate(row_9, c_5, _e148, open_fight_2, _e10.cohort_merge);
            let _e153 = w_1;
            if ((_e153 == NONE) || _e150) {
                w_1 = c_5;
            }
        }
        continuing {
            let _e159 = t;
            t = (_e159 + 1u);
        }
    }
    let _e164 = w_1;
    parent[row_9] = _e164;
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn mutate_main(@builtin(global_invocation_id) gid_2: vec3<u32>) {
    var i_1: u32 = 0u;
    var i_2: u32 = 0u;
    var g: u32 = 0u;
    var i_3: u32;
    var next_child: u32;
    var pos: u32;
    var k: u32;
    var i_4: u32 = 0u;
    var i_5: u32 = 0u;
    var pos_1: u32;
    var i_6: u32 = 0u;
    var trial: u32 = 0u;
    var n_fn: u32;
    var pos_2: u32;
    var start: u32;
    var seen: u32;
    var pos_3: u32;
    var i_7: u32;
    var i_8: u32;
    var pos_4: u32;
    var i_9: u32;
    var pos_5: u32 = 0u;
    var k_1: u32 = 0u;
    var g_1: u32 = 0u;
    var pos_6: u32;
    var i_10: u32 = 0u;
    var i_11: u32 = 0u;
    var pos_7: u32;
    var i_12: u32 = 0u;
    var k_2: u32 = 0u;

    let row_10 = gid_2.x;
    let _e18 = gp.pop;
    if (row_10 >= _e18) {
        return;
    }
    let _e20 = island_of(row_10);
    let h_1 = gp.head;
    let vh = gp.vhead;
    let t_1 = gp.tail;
    let ht_1 = (h_1 + t_1);
    let width_1 = (h_1 + (2u * t_1));
    let _e36 = gp.n_genes;
    let row_w_1 = (_e36 * width_1);
    let _e40 = gp.n_genes;
    let _e42 = gp.n_rnc;
    let rnc_w_1 = (_e40 * _e42);
    let src_1 = parent[row_10];
    let base_1 = (row_10 * row_w_1);
    let rbase = (row_10 * rnc_w_1);
    loop {
        let _e50 = i_1;
        if (_e50 < row_w_1) {
        } else {
            break;
        }
        {
            let _e55 = i_1;
            let _e59 = i_1;
            let _e62 = genome_now[((src_1 * row_w_1) + _e59)];
            genome[(base_1 + _e55)] = _e62;
        }
        continuing {
            let _e65 = i_1;
            i_1 = (_e65 + 1u);
        }
    }
    loop {
        let _e68 = i_2;
        if (_e68 < rnc_w_1) {
        } else {
            break;
        }
        {
            let _e73 = i_2;
            let _e77 = i_2;
            let _e80 = rnc_now[((src_1 * rnc_w_1) + _e77)];
            rnc[(rbase + _e73)] = _e80;
        }
        continuing {
            let _e83 = i_2;
            i_2 = (_e83 + 1u);
        }
    }
    let _e89 = wrapper_now[src_1];
    wrapper_id[row_10] = _e89;
    let _e94 = cohort_now[src_1];
    cohort[row_10] = _e94;
    if (row_10 < (_e20.lo + _e20.elites)) {
        let _e103 = fitness_now[src_1];
        fitness[row_10] = _e103;
        return;
    }
    fitness[row_10] = bitcast<f32>(NAN_BITS);
    let _e112 = gp.typed_depth;
    let typed = ((_e112 > 0u) && (ht_1 <= MAX_HT));
    loop {
        let _e118 = g;
        let _e120 = gp.n_genes;
        if (_e118 < _e120) {
        } else {
            break;
        }
        {
            if typed {
                i_3 = 0u;
                loop {
                    let _e125 = i_3;
                    if (_e125 < ht_1) {
                    } else {
                        break;
                    }
                    {
                        let _e130 = i_3;
                        let _e133 = gp.typed_depth;
                        mut_budget[_e130] = _e133;
                    }
                    continuing {
                        let _e136 = i_3;
                        i_3 = (_e136 + 1u);
                    }
                }
            }
            next_child = 1u;
            pos = 0u;
            loop {
                let _e143 = pos;
                if (_e143 < ht_1) {
                } else {
                    break;
                }
                {
                    let _e147 = g;
                    let _e149 = pos;
                    let slot_4 = ((_e147 * width_1) + _e149);
                    let _e153 = chance(row_10, slot_4, STREAM_MUT_HIT, _e20.mut_point);
                    if _e153 {
                        let _e155 = coin(row_10, slot_4, STREAM_MUT_KIND);
                        let _e157 = pos;
                        if ((_e157 < vh) && _e155) {
                            let _e163 = pos;
                            let _e165 = mut_budget[_e163];
                            if (typed && (_e165 == 0u)) {
                                let _e171 = gp.n_flat;
                                if (_e171 == 0u) {
                                    let _e174 = g;
                                    let _e176 = pos;
                                    let _e177 = at(base_1, _e174, _e176);
                                    let _e181 = gp.n_terminals;
                                    let _e182 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e181);
                                    let _e187 = sample_terminals[_e182];
                                    genome[_e177] = _e187;
                                } else {
                                    let _e189 = g;
                                    let _e191 = pos;
                                    let _e192 = at(base_1, _e189, _e191);
                                    let _e196 = gp.n_flat;
                                    let _e197 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e196);
                                    let _e202 = sample_flat[_e197];
                                    genome[_e192] = _e202;
                                }
                            } else {
                                let _e204 = g;
                                let _e206 = pos;
                                let _e207 = at(base_1, _e204, _e206);
                                let _e211 = gp.n_functions;
                                let _e212 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e211);
                                let _e217 = sample_functions[_e212];
                                genome[_e207] = _e217;
                            }
                        } else {
                            let _e219 = g;
                            let _e221 = pos;
                            let _e222 = at(base_1, _e219, _e221);
                            let _e226 = gp.n_terminals;
                            let _e227 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e226);
                            let _e232 = sample_terminals[_e227];
                            genome[_e222] = _e232;
                        }
                    }
                    let _e235 = pos;
                    let _e236 = next_child;
                    if (typed && (_e235 < _e236)) {
                        let _e240 = g;
                        let _e242 = pos;
                        let _e243 = at(base_1, _e240, _e242);
                        let id = genome[_e243];
                        let a_8 = arity[id];
                        let _e253 = pos;
                        let _e255 = mut_budget[_e253];
                        let _e256 = pos;
                        let _e258 = mut_budget[_e256];
                        let _e260 = depth_cost[id];
                        let child = (_e255 - min(_e258, _e260));
                        k = 0u;
                        loop {
                            let _e266 = k;
                            if (_e266 < a_8) {
                            } else {
                                break;
                            }
                            {
                                let _e269 = next_child;
                                if (_e269 < ht_1) {
                                    let _e273 = next_child;
                                    mut_budget[_e273] = child;
                                }
                                let _e277 = next_child;
                                next_child = (_e277 + 1u);
                            }
                            continuing {
                                let _e281 = k;
                                k = (_e281 + 1u);
                            }
                        }
                    }
                }
                continuing {
                    let _e285 = pos;
                    pos = (_e285 + 1u);
                }
            }
        }
        continuing {
            let _e289 = g;
            g = (_e289 + 1u);
        }
    }
    let _e294 = chance(row_10, OP_INVERT, STREAM_OPERATOR, _e20.invert);
    if _e294 {
        let _e299 = gp.n_genes;
        let _e300 = below(row_10, 0u, STREAM_INVERT, _e299);
        let _e305 = below(row_10, 1u, STREAM_INVERT, (vh - 1u));
        let len = (2u + _e305);
        let _e313 = below(row_10, 2u, STREAM_INVERT, ((vh - len) + 1u));
        let _e314 = at(base_1, _e300, _e313);
        let _e316 = at(base_1, _e300, (_e313 + len));
        reverse(_e314, _e316);
    }
    let _e320 = chance(row_10, OP_IS, STREAM_OPERATOR, _e20.is_transpose);
    if _e320 {
        let _e325 = gp.n_genes;
        let _e326 = below(row_10, 0u, STREAM_IS, _e325);
        let _e331 = gp.n_genes;
        let _e332 = below(row_10, 1u, STREAM_IS, _e331);
        let _e337 = below(row_10, 2u, STREAM_IS, (vh - 1u));
        let len_1 = (1u + _e337);
        let _e345 = below(row_10, 3u, STREAM_IS, ((ht_1 - len_1) + 1u));
        let _e349 = below(row_10, 4u, STREAM_IS, (vh - len_1));
        let ins = (1u + _e349);
        loop {
            let _e353 = i_4;
            if (_e353 < len_1) {
            } else {
                break;
            }
            {
                let _e356 = i_4;
                let _e358 = at(base_1, _e326, (_e345 + _e356));
                let _e362 = i_4;
                let _e365 = genome[_e358];
                segment[_e362] = _e365;
            }
            continuing {
                let _e368 = i_4;
                i_4 = (_e368 + 1u);
            }
        }
        loop {
            let _e371 = i_5;
            if (_e371 < vh) {
            } else {
                break;
            }
            {
                let _e374 = i_5;
                let _e375 = at(base_1, _e332, _e374);
                let _e379 = i_5;
                let _e382 = genome[_e375];
                before[_e379] = _e382;
            }
            continuing {
                let _e385 = i_5;
                i_5 = (_e385 + 1u);
            }
        }
        pos_1 = (ins + len_1);
        loop {
            let _e390 = pos_1;
            if (_e390 < vh) {
            } else {
                break;
            }
            {
                let _e393 = pos_1;
                let _e394 = at(base_1, _e332, _e393);
                let _e399 = pos_1;
                let _e402 = before[(_e399 - len_1)];
                genome[_e394] = _e402;
            }
            continuing {
                let _e405 = pos_1;
                pos_1 = (_e405 + 1u);
            }
        }
        loop {
            let _e408 = i_6;
            if (_e408 < len_1) {
            } else {
                break;
            }
            {
                let _e411 = i_6;
                let _e413 = at(base_1, _e332, (ins + _e411));
                let _e418 = i_6;
                let _e420 = segment[_e418];
                genome[_e413] = _e420;
            }
            continuing {
                let _e423 = i_6;
                i_6 = (_e423 + 1u);
            }
        }
    }
    let _e428 = chance(row_10, OP_RIS, STREAM_OPERATOR, _e20.ris_transpose);
    if _e428 {
        loop {
            let _e433 = trial;
            let _e435 = gp.n_genes;
            if (_e433 < ((2u * _e435) + 1u)) {
            } else {
                break;
            }
            {
                let _e441 = trial;
                let _e446 = gp.n_genes;
                let _e447 = below(row_10, (_e441 * 4u), STREAM_RIS, _e446);
                let _e451 = trial;
                let _e457 = gp.n_genes;
                let _e458 = below(row_10, ((_e451 * 4u) + 1u), STREAM_RIS, _e457);
                n_fn = 0u;
                pos_2 = 0u;
                loop {
                    let _e464 = pos_2;
                    if (_e464 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e467 = pos_2;
                        let _e468 = at(base_1, _e447, _e467);
                        let _e473 = genome[_e468];
                        let _e475 = arity[_e473];
                        if (_e475 > 0u) {
                            let _e479 = n_fn;
                            n_fn = (_e479 + 1u);
                        }
                    }
                    continuing {
                        let _e483 = pos_2;
                        pos_2 = (_e483 + 1u);
                    }
                }
                let _e487 = n_fn;
                if (_e487 == 0u) {
                    continue;
                }
                let _e492 = trial;
                let _e497 = n_fn;
                let _e498 = below(row_10, ((_e492 * 4u) + 2u), STREAM_RIS, _e497);
                start = 0u;
                seen = 0u;
                pos_3 = 0u;
                loop {
                    let _e506 = pos_3;
                    if (_e506 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e509 = pos_3;
                        let _e510 = at(base_1, _e447, _e509);
                        let _e515 = genome[_e510];
                        let _e517 = arity[_e515];
                        if (_e517 > 0u) {
                            let _e520 = seen;
                            if (_e520 == _e498) {
                                let _e524 = pos_3;
                                start = _e524;
                            }
                            let _e527 = seen;
                            seen = (_e527 + 1u);
                        }
                    }
                    continuing {
                        let _e531 = pos_3;
                        pos_3 = (_e531 + 1u);
                    }
                }
                let _e536 = trial;
                let _e542 = start;
                let _e546 = below(row_10, ((_e536 * 4u) + 3u), STREAM_RIS, (min(vh, (ht_1 - _e542)) - 1u));
                let len_2 = (2u + _e546);
                i_7 = 0u;
                loop {
                    let _e552 = i_7;
                    if (_e552 < len_2) {
                    } else {
                        break;
                    }
                    {
                        let _e556 = start;
                        let _e557 = i_7;
                        let _e559 = at(base_1, _e447, (_e556 + _e557));
                        let _e563 = i_7;
                        let _e566 = genome[_e559];
                        segment[_e563] = _e566;
                    }
                    continuing {
                        let _e569 = i_7;
                        i_7 = (_e569 + 1u);
                    }
                }
                i_8 = 0u;
                loop {
                    let _e574 = i_8;
                    if (_e574 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e577 = i_8;
                        let _e578 = at(base_1, _e458, _e577);
                        let _e582 = i_8;
                        let _e585 = genome[_e578];
                        before[_e582] = _e585;
                    }
                    continuing {
                        let _e588 = i_8;
                        i_8 = (_e588 + 1u);
                    }
                }
                pos_4 = len_2;
                loop {
                    let _e592 = pos_4;
                    if (_e592 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e595 = pos_4;
                        let _e596 = at(base_1, _e458, _e595);
                        let _e601 = pos_4;
                        let _e604 = before[(_e601 - len_2)];
                        genome[_e596] = _e604;
                    }
                    continuing {
                        let _e607 = pos_4;
                        pos_4 = (_e607 + 1u);
                    }
                }
                i_9 = 0u;
                loop {
                    let _e612 = i_9;
                    if (_e612 < len_2) {
                    } else {
                        break;
                    }
                    {
                        let _e615 = i_9;
                        let _e616 = at(base_1, _e458, _e615);
                        let _e621 = i_9;
                        let _e623 = segment[_e621];
                        genome[_e616] = _e623;
                    }
                    continuing {
                        let _e626 = i_9;
                        i_9 = (_e626 + 1u);
                    }
                }
                break;
            }
            continuing {
                let _e630 = trial;
                trial = (_e630 + 1u);
            }
        }
    }
    let _e635 = chance(row_10, OP_GENE_T, STREAM_OPERATOR, _e20.gene_transpose);
    let _e639 = gp.n_genes;
    if ((_e639 > 1u) && _e635) {
        let _e647 = gp.n_genes;
        let _e649 = below(row_10, 0u, STREAM_GENE_T, (_e647 - 1u));
        let source = (1u + _e649);
        loop {
            let _e653 = pos_5;
            if (_e653 < width_1) {
            } else {
                break;
            }
            {
                let _e657 = pos_5;
                let _e658 = at(base_1, 0u, _e657);
                let held_3 = genome[_e658];
                let _e664 = pos_5;
                let _e665 = at(base_1, 0u, _e664);
                let _e667 = pos_5;
                let _e668 = at(base_1, source, _e667);
                let _e672 = genome[_e668];
                genome[_e665] = _e672;
                let _e674 = pos_5;
                let _e675 = at(base_1, source, _e674);
                genome[_e675] = held_3;
            }
            continuing {
                let _e680 = pos_5;
                pos_5 = (_e680 + 1u);
            }
        }
        loop {
            let _e684 = k_1;
            let _e686 = gp.n_rnc;
            if (_e684 < _e686) {
            } else {
                break;
            }
            {
                let _e690 = k_1;
                let held_4 = rnc[(rbase + _e690)];
                let _e697 = k_1;
                let _e701 = gp.n_rnc;
                let _e704 = k_1;
                let _e707 = rnc[((rbase + (source * _e701)) + _e704)];
                rnc[(rbase + _e697)] = _e707;
                let _e712 = gp.n_rnc;
                let _e715 = k_1;
                rnc[((rbase + (source * _e712)) + _e715)] = held_4;
            }
            continuing {
                let _e720 = k_1;
                k_1 = (_e720 + 1u);
            }
        }
    }
    loop {
        let _e724 = g_1;
        let _e726 = gp.n_genes;
        if (_e724 < _e726) {
        } else {
            break;
        }
        {
            pos_6 = ht_1;
            loop {
                let _e730 = pos_6;
                if (_e730 < width_1) {
                } else {
                    break;
                }
                {
                    let _e734 = g_1;
                    let _e736 = pos_6;
                    let slot_5 = ((_e734 * width_1) + _e736);
                    let _e740 = chance(row_10, slot_5, STREAM_DC_HIT, _e20.dc_point);
                    if _e740 {
                        let _e742 = g_1;
                        let _e744 = pos_6;
                        let _e745 = at(base_1, _e742, _e744);
                        let _e749 = gp.n_rnc;
                        let _e750 = below(row_10, slot_5, STREAM_DC_VALUE, _e749);
                        genome[_e745] = _e750;
                    }
                }
                continuing {
                    let _e755 = pos_6;
                    pos_6 = (_e755 + 1u);
                }
            }
        }
        continuing {
            let _e759 = g_1;
            g_1 = (_e759 + 1u);
        }
    }
    let _e764 = chance(row_10, OP_INVERT_DC, STREAM_OPERATOR, _e20.invert_dc);
    if ((t_1 >= 2u) && _e764) {
        let _e772 = gp.n_genes;
        let _e773 = below(row_10, 0u, STREAM_INVERT_DC, _e772);
        let _e778 = below(row_10, 1u, STREAM_INVERT_DC, (t_1 - 1u));
        let len_3 = (2u + _e778);
        let _e786 = below(row_10, 2u, STREAM_INVERT_DC, ((t_1 - len_3) + 1u));
        let start_1 = (ht_1 + _e786);
        let _e788 = at(base_1, _e773, start_1);
        let _e792 = at(base_1, _e773, ((start_1 + len_3) - 1u));
        reverse(_e788, _e792);
    }
    let _e796 = chance(row_10, OP_TRANSPOSE_DC, STREAM_OPERATOR, _e20.transpose_dc);
    if _e796 {
        let _e801 = gp.n_genes;
        let _e802 = below(row_10, 0u, STREAM_TRANSPOSE_DC, _e801);
        let _e807 = gp.n_genes;
        let _e808 = below(row_10, 1u, STREAM_TRANSPOSE_DC, _e807);
        let _e811 = below(row_10, 2u, STREAM_TRANSPOSE_DC, t_1);
        let len_4 = (1u + _e811);
        let _e819 = below(row_10, 3u, STREAM_TRANSPOSE_DC, ((t_1 - len_4) + 1u));
        let start_2 = (ht_1 + _e819);
        let _e826 = below(row_10, 4u, STREAM_TRANSPOSE_DC, ((t_1 - len_4) + 1u));
        let ins_1 = (ht_1 + _e826);
        loop {
            let _e829 = i_10;
            if (_e829 < len_4) {
            } else {
                break;
            }
            {
                let _e832 = i_10;
                let _e834 = at(base_1, _e802, (start_2 + _e832));
                let _e838 = i_10;
                let _e841 = genome[_e834];
                segment[_e838] = _e841;
            }
            continuing {
                let _e844 = i_10;
                i_10 = (_e844 + 1u);
            }
        }
        loop {
            let _e847 = i_11;
            if (_e847 < t_1) {
            } else {
                break;
            }
            {
                let _e850 = i_11;
                let _e852 = at(base_1, _e808, (ht_1 + _e850));
                let _e856 = i_11;
                let _e859 = genome[_e852];
                before[_e856] = _e859;
            }
            continuing {
                let _e862 = i_11;
                i_11 = (_e862 + 1u);
            }
        }
        pos_7 = (ins_1 + len_4);
        loop {
            let _e867 = pos_7;
            if (_e867 < width_1) {
            } else {
                break;
            }
            {
                let _e870 = pos_7;
                let _e871 = at(base_1, _e808, _e870);
                let _e876 = pos_7;
                let _e880 = before[((_e876 - len_4) - ht_1)];
                genome[_e871] = _e880;
            }
            continuing {
                let _e883 = pos_7;
                pos_7 = (_e883 + 1u);
            }
        }
        loop {
            let _e886 = i_12;
            if (_e886 < len_4) {
            } else {
                break;
            }
            {
                let _e889 = i_12;
                let _e891 = at(base_1, _e808, (ins_1 + _e889));
                let _e896 = i_12;
                let _e898 = segment[_e896];
                genome[_e891] = _e898;
            }
            continuing {
                let _e901 = i_12;
                i_12 = (_e901 + 1u);
            }
        }
    }
    loop {
        let _e904 = k_2;
        if (_e904 < rnc_w_1) {
        } else {
            break;
        }
        {
            let _e907 = k_2;
            let _e910 = chance(row_10, _e907, STREAM_RNC_HIT, _e20.rnc_point);
            if _e910 {
                let _e912 = k_2;
                let _e916 = gp.rnc_span;
                let _e917 = below(row_10, _e912, STREAM_RNC_VALUE, _e916);
                let _e921 = k_2;
                let _e925 = gp.rnc_lo;
                rnc[(rbase + _e921)] = f32((_e925 + i32(_e917)));
            }
        }
        continuing {
            let _e931 = k_2;
            k_2 = (_e931 + 1u);
        }
    }
    let _e936 = chance(row_10, OP_CLEANSE, STREAM_OPERATOR, _e20.cleanse);
    if _e936 {
        let _e941 = gp.n_genes;
        let _e942 = below(row_10, 0u, STREAM_CLEANSE, _e941);
        cleanse_gene(row_10, (base_1 + (_e942 * width_1)), _e20.cleanse_collapse);
        return;
    } else {
        return;
    }
}

@compute @workgroup_size(64, 1, 1) 
fn crossover_main(@builtin(global_invocation_id) gid_3: vec3<u32>) {
    var whole: u32 = 0u;
    var whole_1: u32;
    var pos_8: u32 = 0u;

    let b_8 = gid_3.x;
    let _e6 = gp.pop;
    if (b_8 >= _e6) {
        return;
    }
    let _e8 = island_of(b_8);
    let first = (_e8.lo + _e8.elites);
    if ((b_8 <= first) || (((b_8 - first) & 1u) == 0u)) {
        return;
    }
    let a_9 = (b_8 - 1u);
    let _e24 = gp.head;
    let _e26 = gp.tail;
    let width_2 = (_e24 + (2u * _e26));
    let _e31 = gp.n_genes;
    let row_w_2 = (_e31 * width_2);
    let arrival_pair = (((_e8.arrivals > 0u) && (_e8.arrival_children > 0u)) && (b_8 < (first + (_e8.arrivals * (_e8.arrival_children + 1u)))));
    let _e50 = chance(b_8, OP_CX_1P, STREAM_OPERATOR, _e8.cx_one_point);
    if (arrival_pair || _e50) {
        let _e56 = gp.n_genes;
        let _e57 = below(b_8, 0u, STREAM_CX_1P, _e56);
        let _e60 = below(b_8, 1u, STREAM_CX_1P, width_2);
        loop {
            let _e62 = whole;
            if (_e62 < _e57) {
            } else {
                break;
            }
            {
                let _e65 = whole;
                swap_tokens(a_9, b_8, _e65, 0u, width_2);
                let _e68 = whole;
                let _e70 = whole;
                swap_consts(a_9, _e68, b_8, _e70);
            }
            continuing {
                let _e73 = whole;
                whole = (_e73 + 1u);
            }
        }
        swap_tokens(a_9, b_8, _e57, 0u, (_e60 + 1u));
    }
    let _e81 = chance(b_8, OP_CX_2P, STREAM_OPERATOR, _e8.cx_two_point);
    if _e81 {
        let _e86 = gp.n_genes;
        let _e87 = below(b_8, 0u, STREAM_CX_2P, _e86);
        let _e92 = gp.n_genes;
        let _e93 = below(b_8, 1u, STREAM_CX_2P, _e92);
        let g1_ = min(_e87, _e93);
        let g2_ = max(_e87, _e93);
        let _e98 = below(b_8, 2u, STREAM_CX_2P, width_2);
        let _e101 = below(b_8, 3u, STREAM_CX_2P, width_2);
        if (g1_ == g2_) {
            swap_tokens(a_9, b_8, g1_, min(_e98, _e101), (max(_e98, _e101) + 1u));
        } else {
            whole_1 = (g1_ + 1u);
            loop {
                let _e111 = whole_1;
                if (_e111 < g2_) {
                } else {
                    break;
                }
                {
                    let _e114 = whole_1;
                    swap_tokens(a_9, b_8, _e114, 0u, width_2);
                    let _e117 = whole_1;
                    let _e119 = whole_1;
                    swap_consts(a_9, _e117, b_8, _e119);
                }
                continuing {
                    let _e122 = whole_1;
                    whole_1 = (_e122 + 1u);
                }
            }
            swap_tokens(a_9, b_8, g1_, _e98, width_2);
            swap_tokens(a_9, b_8, g2_, 0u, (_e101 + 1u));
        }
    }
    let _e130 = chance(b_8, OP_CX_GENE, STREAM_OPERATOR, _e8.cx_gene);
    if _e130 {
        let _e135 = gp.n_genes;
        let _e136 = below(b_8, 0u, STREAM_CX_GENE, _e135);
        let _e141 = gp.n_genes;
        let _e142 = below(b_8, 1u, STREAM_CX_GENE, _e141);
        loop {
            let _e144 = pos_8;
            if (_e144 < width_2) {
            } else {
                break;
            }
            {
                let _e150 = pos_8;
                let ia_2 = (((a_9 * row_w_2) + (_e136 * width_2)) + _e150);
                let _e156 = pos_8;
                let ib_2 = (((b_8 * row_w_2) + (_e142 * width_2)) + _e156);
                let held_5 = genome[ia_2];
                let _e164 = genome[ib_2];
                genome[ia_2] = _e164;
                genome[ib_2] = held_5;
            }
            continuing {
                let _e169 = pos_8;
                pos_8 = (_e169 + 1u);
            }
        }
        swap_consts(a_9, _e136, b_8, _e142);
        return;
    } else {
        return;
    }
}
