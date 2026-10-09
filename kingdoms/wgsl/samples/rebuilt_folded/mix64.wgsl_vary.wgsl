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
        let _e6 = a_5;
        let def_a = (_e6 + 1u);
        let _e9 = b_5;
        if (def_a >= _e9) {
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
        a_5 = def_a;
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
        let _e212 = m;
        let def_b = (_e212 + 1u);
        let _e216 = k_3;
        let def_a_1 = (_e216 + 1u);
        let _e220 = head;
        let _e221 = tail;
        if (_e220 >= _e221) {
            break;
        }
        let _e225 = head;
        let i_16 = cl_queue[_e225];
        let _e230 = head;
        head = (_e230 + 1u);
        if (i_16 == LEAF) {
            let _e237 = m;
            let _e240 = gp.rnc_id;
            cl_tok[_e237] = _e240;
            let _e243 = k_3;
            cl_dc[_e243] = _e200;
            k_3 = def_a_1;
            m = def_b;
            continue;
        }
        let tok_1 = genome[(gene + i_16)];
        let _e254 = m;
        cl_tok[_e254] = tok_1;
        let _e259 = cl_ordinal[i_16];
        if (_e259 != NONE) {
            let _e268 = k_3;
            let _e272 = cl_ordinal[i_16];
            let _e275 = genome[((gene + ht) + _e272)];
            let _e277 = cl_ordinal[i_16];
            let _e279 = gp.tail;
            cl_dc[_e268] = select(0u, _e275, (_e277 < _e279));
            k_3 = def_a_1;
        }
        m = def_b;
        j = 0u;
        loop {
            let _e288 = j;
            let _e290 = arity[tok_1];
            if (_e288 < _e290) {
            } else {
                break;
            }
            {
                let _e295 = cl_child[i_16];
                let _e296 = j;
                let c_3 = (_e295 + _e296);
                let _e302 = tail;
                let _e304 = q_1;
                let _e305 = p;
                cl_queue[_e302] = select(c_3, _e304, (c_3 == _e305));
                let _e310 = tail;
                tail = (_e310 + 1u);
            }
            continuing {
                let _e314 = j;
                j = (_e314 + 1u);
            }
        }
    }
    let _e318 = k_3;
    let _e320 = gp.tail;
    if (_e318 > _e320) {
        return;
    }
    x_1 = h;
    loop {
        let _e325 = x_1;
        let _e326 = m;
        if (_e325 < _e326) {
        } else {
            break;
        }
        {
            let _e332 = x_1;
            let _e334 = cl_tok[_e332];
            let _e336 = arity[_e334];
            if (_e336 > 0u) {
                return;
            }
        }
        continuing {
            let _e340 = x_1;
            x_1 = (_e340 + 1u);
        }
    }
    loop {
        let _e344 = x_2;
        let _e345 = m;
        if (_e344 < _e345) {
        } else {
            break;
        }
        {
            let _e351 = x_2;
            let _e354 = x_2;
            let _e356 = cl_tok[_e354];
            genome[(gene + _e351)] = _e356;
        }
        continuing {
            let _e359 = x_2;
            x_2 = (_e359 + 1u);
        }
    }
    loop {
        let _e363 = x_3;
        let _e364 = k_3;
        if (_e363 < _e364) {
        } else {
            break;
        }
        {
            let _e371 = x_3;
            let _e374 = x_3;
            let _e376 = cl_dc[_e374];
            genome[((gene + ht) + _e371)] = _e376;
        }
        continuing {
            let _e379 = x_3;
            x_3 = (_e379 + 1u);
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
    let band = (_e10.lo + _e10.elites);
    let open_fight_2 = (_e10.open_fight == 1u);
    let n_4 = (_e10.hi - _e10.lo);
    if (row_9 < band) {
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
    let def_b_1 = ((row_9 - band) % stride);
    if ((((_e10.arrivals > 0u) && (_e10.arrival_children > 0u)) && (row_9 >= band)) && (row_9 < (band + (_e10.arrivals * stride)))) {
        if (def_b_1 == 0u) {
            parent[row_9] = row_9;
        } else {
            let _e124 = below(row_9, def_b_1, STREAM_ARRIVAL_MATE, n_4);
            parent[row_9] = (_e10.lo + _e124);
        }
        return;
    }
    loop {
        let _e130 = t;
        if (_e130 < _e10.tournsize) {
        } else {
            break;
        }
        {
            let _e134 = t;
            let _e136 = below(row_9, _e134, STREAM_SELECT_2_, n_4);
            let c_5 = stage1_[(_e10.lo + _e136)];
            let _e143 = w_1;
            let _e145 = better_mate(row_9, c_5, _e143, open_fight_2, _e10.cohort_merge);
            let _e148 = w_1;
            if ((_e148 == NONE) || _e145) {
                w_1 = c_5;
            }
        }
        continuing {
            let _e154 = t;
            t = (_e154 + 1u);
        }
    }
    let _e159 = w_1;
    parent[row_9] = _e159;
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
    let def_d = (vh - 1u);
    let t_1 = gp.tail;
    let ht_1 = (h_1 + t_1);
    let width_1 = (h_1 + (2u * t_1));
    let _e38 = gp.n_genes;
    let row_w_1 = (_e38 * width_1);
    let _e42 = gp.n_genes;
    let _e44 = gp.n_rnc;
    let rnc_w_1 = (_e42 * _e44);
    let src_1 = parent[row_10];
    let base_1 = (row_10 * row_w_1);
    let rbase = (row_10 * rnc_w_1);
    loop {
        let _e52 = i_1;
        if (_e52 < row_w_1) {
        } else {
            break;
        }
        {
            let _e57 = i_1;
            let _e61 = i_1;
            let _e64 = genome_now[((src_1 * row_w_1) + _e61)];
            genome[(base_1 + _e57)] = _e64;
        }
        continuing {
            let _e67 = i_1;
            i_1 = (_e67 + 1u);
        }
    }
    loop {
        let _e70 = i_2;
        if (_e70 < rnc_w_1) {
        } else {
            break;
        }
        {
            let _e75 = i_2;
            let _e79 = i_2;
            let _e82 = rnc_now[((src_1 * rnc_w_1) + _e79)];
            rnc[(rbase + _e75)] = _e82;
        }
        continuing {
            let _e85 = i_2;
            i_2 = (_e85 + 1u);
        }
    }
    let _e91 = wrapper_now[src_1];
    wrapper_id[row_10] = _e91;
    let _e96 = cohort_now[src_1];
    cohort[row_10] = _e96;
    if (row_10 < (_e20.lo + _e20.elites)) {
        let _e105 = fitness_now[src_1];
        fitness[row_10] = _e105;
        return;
    }
    fitness[row_10] = bitcast<f32>(NAN_BITS);
    let _e114 = gp.typed_depth;
    let typed = ((_e114 > 0u) && (ht_1 <= MAX_HT));
    loop {
        let _e120 = g;
        let _e122 = gp.n_genes;
        if (_e120 < _e122) {
        } else {
            break;
        }
        {
            if typed {
                i_3 = 0u;
                loop {
                    let _e127 = i_3;
                    if (_e127 < ht_1) {
                    } else {
                        break;
                    }
                    {
                        let _e132 = i_3;
                        let _e135 = gp.typed_depth;
                        mut_budget[_e132] = _e135;
                    }
                    continuing {
                        let _e138 = i_3;
                        i_3 = (_e138 + 1u);
                    }
                }
            }
            next_child = 1u;
            pos = 0u;
            loop {
                let _e145 = pos;
                if (_e145 < ht_1) {
                } else {
                    break;
                }
                {
                    let _e149 = g;
                    let _e151 = pos;
                    let slot_4 = ((_e149 * width_1) + _e151);
                    let _e155 = chance(row_10, slot_4, STREAM_MUT_HIT, _e20.mut_point);
                    if _e155 {
                        let _e157 = coin(row_10, slot_4, STREAM_MUT_KIND);
                        let _e159 = pos;
                        if ((_e159 < vh) && _e157) {
                            let _e165 = pos;
                            let _e167 = mut_budget[_e165];
                            if (typed && (_e167 == 0u)) {
                                let _e173 = gp.n_flat;
                                if (_e173 == 0u) {
                                    let _e176 = g;
                                    let _e178 = pos;
                                    let _e179 = at(base_1, _e176, _e178);
                                    let _e183 = gp.n_terminals;
                                    let _e184 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e183);
                                    let _e189 = sample_terminals[_e184];
                                    genome[_e179] = _e189;
                                } else {
                                    let _e191 = g;
                                    let _e193 = pos;
                                    let _e194 = at(base_1, _e191, _e193);
                                    let _e198 = gp.n_flat;
                                    let _e199 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e198);
                                    let _e204 = sample_flat[_e199];
                                    genome[_e194] = _e204;
                                }
                            } else {
                                let _e206 = g;
                                let _e208 = pos;
                                let _e209 = at(base_1, _e206, _e208);
                                let _e213 = gp.n_functions;
                                let _e214 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e213);
                                let _e219 = sample_functions[_e214];
                                genome[_e209] = _e219;
                            }
                        } else {
                            let _e221 = g;
                            let _e223 = pos;
                            let _e224 = at(base_1, _e221, _e223);
                            let _e228 = gp.n_terminals;
                            let _e229 = below(row_10, slot_4, STREAM_MUT_SYMBOL, _e228);
                            let _e234 = sample_terminals[_e229];
                            genome[_e224] = _e234;
                        }
                    }
                    let _e237 = pos;
                    let _e238 = next_child;
                    if (typed && (_e237 < _e238)) {
                        let _e242 = g;
                        let _e244 = pos;
                        let _e245 = at(base_1, _e242, _e244);
                        let id = genome[_e245];
                        let a_8 = arity[id];
                        let _e255 = pos;
                        let _e257 = mut_budget[_e255];
                        let _e258 = pos;
                        let _e260 = mut_budget[_e258];
                        let _e262 = depth_cost[id];
                        let child = (_e257 - min(_e260, _e262));
                        k = 0u;
                        loop {
                            let _e268 = k;
                            if (_e268 < a_8) {
                            } else {
                                break;
                            }
                            {
                                let _e271 = next_child;
                                if (_e271 < ht_1) {
                                    let _e275 = next_child;
                                    mut_budget[_e275] = child;
                                }
                                let _e279 = next_child;
                                next_child = (_e279 + 1u);
                            }
                            continuing {
                                let _e283 = k;
                                k = (_e283 + 1u);
                            }
                        }
                    }
                }
                continuing {
                    let _e287 = pos;
                    pos = (_e287 + 1u);
                }
            }
        }
        continuing {
            let _e291 = g;
            g = (_e291 + 1u);
        }
    }
    let _e296 = chance(row_10, OP_INVERT, STREAM_OPERATOR, _e20.invert);
    if _e296 {
        let _e301 = gp.n_genes;
        let _e302 = below(row_10, 0u, STREAM_INVERT, _e301);
        let _e305 = below(row_10, 1u, STREAM_INVERT, def_d);
        let len = (2u + _e305);
        let _e313 = below(row_10, 2u, STREAM_INVERT, ((vh - len) + 1u));
        let _e314 = at(base_1, _e302, _e313);
        let _e316 = at(base_1, _e302, (_e313 + len));
        reverse(_e314, _e316);
    }
    let _e320 = chance(row_10, OP_IS, STREAM_OPERATOR, _e20.is_transpose);
    if _e320 {
        let _e325 = gp.n_genes;
        let _e326 = below(row_10, 0u, STREAM_IS, _e325);
        let _e331 = gp.n_genes;
        let _e332 = below(row_10, 1u, STREAM_IS, _e331);
        let _e335 = below(row_10, 2u, STREAM_IS, def_d);
        let len_1 = (1u + _e335);
        let _e343 = below(row_10, 3u, STREAM_IS, ((ht_1 - len_1) + 1u));
        let _e347 = below(row_10, 4u, STREAM_IS, (vh - len_1));
        let ins = (1u + _e347);
        loop {
            let _e351 = i_4;
            if (_e351 < len_1) {
            } else {
                break;
            }
            {
                let _e354 = i_4;
                let _e356 = at(base_1, _e326, (_e343 + _e354));
                let _e360 = i_4;
                let _e363 = genome[_e356];
                segment[_e360] = _e363;
            }
            continuing {
                let _e366 = i_4;
                i_4 = (_e366 + 1u);
            }
        }
        loop {
            let _e369 = i_5;
            if (_e369 < vh) {
            } else {
                break;
            }
            {
                let _e372 = i_5;
                let _e373 = at(base_1, _e332, _e372);
                let _e377 = i_5;
                let _e380 = genome[_e373];
                before[_e377] = _e380;
            }
            continuing {
                let _e383 = i_5;
                i_5 = (_e383 + 1u);
            }
        }
        pos_1 = (ins + len_1);
        loop {
            let _e388 = pos_1;
            if (_e388 < vh) {
            } else {
                break;
            }
            {
                let _e391 = pos_1;
                let _e392 = at(base_1, _e332, _e391);
                let _e397 = pos_1;
                let _e400 = before[(_e397 - len_1)];
                genome[_e392] = _e400;
            }
            continuing {
                let _e403 = pos_1;
                pos_1 = (_e403 + 1u);
            }
        }
        loop {
            let _e406 = i_6;
            if (_e406 < len_1) {
            } else {
                break;
            }
            {
                let _e409 = i_6;
                let _e411 = at(base_1, _e332, (ins + _e409));
                let _e416 = i_6;
                let _e418 = segment[_e416];
                genome[_e411] = _e418;
            }
            continuing {
                let _e421 = i_6;
                i_6 = (_e421 + 1u);
            }
        }
    }
    let _e426 = chance(row_10, OP_RIS, STREAM_OPERATOR, _e20.ris_transpose);
    if _e426 {
        loop {
            let _e429 = trial;
            let def_c = (_e429 * 4u);
            let _e435 = trial;
            let _e437 = gp.n_genes;
            if (_e435 < ((2u * _e437) + 1u)) {
            } else {
                break;
            }
            {
                let _e444 = gp.n_genes;
                let _e445 = below(row_10, def_c, STREAM_RIS, _e444);
                let _e451 = gp.n_genes;
                let _e452 = below(row_10, (def_c + 1u), STREAM_RIS, _e451);
                n_fn = 0u;
                pos_2 = 0u;
                loop {
                    let _e458 = pos_2;
                    if (_e458 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e461 = pos_2;
                        let _e462 = at(base_1, _e445, _e461);
                        let _e467 = genome[_e462];
                        let _e469 = arity[_e467];
                        if (_e469 > 0u) {
                            let _e473 = n_fn;
                            n_fn = (_e473 + 1u);
                        }
                    }
                    continuing {
                        let _e477 = pos_2;
                        pos_2 = (_e477 + 1u);
                    }
                }
                let _e481 = n_fn;
                if (_e481 == 0u) {
                    continue;
                }
                let _e487 = n_fn;
                let _e488 = below(row_10, (def_c + 2u), STREAM_RIS, _e487);
                start = 0u;
                seen = 0u;
                pos_3 = 0u;
                loop {
                    let _e496 = pos_3;
                    if (_e496 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e499 = pos_3;
                        let _e500 = at(base_1, _e445, _e499);
                        let _e505 = genome[_e500];
                        let _e507 = arity[_e505];
                        if (_e507 > 0u) {
                            let _e510 = seen;
                            if (_e510 == _e488) {
                                let _e514 = pos_3;
                                start = _e514;
                            }
                            let _e517 = seen;
                            seen = (_e517 + 1u);
                        }
                    }
                    continuing {
                        let _e521 = pos_3;
                        pos_3 = (_e521 + 1u);
                    }
                }
                let _e528 = start;
                let _e532 = below(row_10, (def_c + 3u), STREAM_RIS, (min(vh, (ht_1 - _e528)) - 1u));
                let len_2 = (2u + _e532);
                i_7 = 0u;
                loop {
                    let _e538 = i_7;
                    if (_e538 < len_2) {
                    } else {
                        break;
                    }
                    {
                        let _e542 = start;
                        let _e543 = i_7;
                        let _e545 = at(base_1, _e445, (_e542 + _e543));
                        let _e549 = i_7;
                        let _e552 = genome[_e545];
                        segment[_e549] = _e552;
                    }
                    continuing {
                        let _e555 = i_7;
                        i_7 = (_e555 + 1u);
                    }
                }
                i_8 = 0u;
                loop {
                    let _e560 = i_8;
                    if (_e560 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e563 = i_8;
                        let _e564 = at(base_1, _e452, _e563);
                        let _e568 = i_8;
                        let _e571 = genome[_e564];
                        before[_e568] = _e571;
                    }
                    continuing {
                        let _e574 = i_8;
                        i_8 = (_e574 + 1u);
                    }
                }
                pos_4 = len_2;
                loop {
                    let _e578 = pos_4;
                    if (_e578 < vh) {
                    } else {
                        break;
                    }
                    {
                        let _e581 = pos_4;
                        let _e582 = at(base_1, _e452, _e581);
                        let _e587 = pos_4;
                        let _e590 = before[(_e587 - len_2)];
                        genome[_e582] = _e590;
                    }
                    continuing {
                        let _e593 = pos_4;
                        pos_4 = (_e593 + 1u);
                    }
                }
                i_9 = 0u;
                loop {
                    let _e598 = i_9;
                    if (_e598 < len_2) {
                    } else {
                        break;
                    }
                    {
                        let _e601 = i_9;
                        let _e602 = at(base_1, _e452, _e601);
                        let _e607 = i_9;
                        let _e609 = segment[_e607];
                        genome[_e602] = _e609;
                    }
                    continuing {
                        let _e612 = i_9;
                        i_9 = (_e612 + 1u);
                    }
                }
                break;
            }
            continuing {
                let _e616 = trial;
                trial = (_e616 + 1u);
            }
        }
    }
    let _e621 = chance(row_10, OP_GENE_T, STREAM_OPERATOR, _e20.gene_transpose);
    let _e625 = gp.n_genes;
    if ((_e625 > 1u) && _e621) {
        let _e633 = gp.n_genes;
        let _e635 = below(row_10, 0u, STREAM_GENE_T, (_e633 - 1u));
        let source = (1u + _e635);
        loop {
            let _e639 = pos_5;
            if (_e639 < width_1) {
            } else {
                break;
            }
            {
                let _e643 = pos_5;
                let _e644 = at(base_1, 0u, _e643);
                let held_3 = genome[_e644];
                let _e650 = pos_5;
                let _e651 = at(base_1, 0u, _e650);
                let _e653 = pos_5;
                let _e654 = at(base_1, source, _e653);
                let _e658 = genome[_e654];
                genome[_e651] = _e658;
                let _e660 = pos_5;
                let _e661 = at(base_1, source, _e660);
                genome[_e661] = held_3;
            }
            continuing {
                let _e666 = pos_5;
                pos_5 = (_e666 + 1u);
            }
        }
        loop {
            let _e671 = gp.n_rnc;
            let _e674 = k_1;
            let def_a_2 = ((rbase + (source * _e671)) + _e674);
            let _e678 = k_1;
            let _e680 = gp.n_rnc;
            if (_e678 < _e680) {
            } else {
                break;
            }
            {
                let _e684 = k_1;
                let held_4 = rnc[(rbase + _e684)];
                let _e690 = k_1;
                let _e694 = rnc[def_a_2];
                rnc[(rbase + _e690)] = _e694;
                rnc[def_a_2] = held_4;
            }
            continuing {
                let _e699 = k_1;
                k_1 = (_e699 + 1u);
            }
        }
    }
    loop {
        let _e703 = g_1;
        let _e705 = gp.n_genes;
        if (_e703 < _e705) {
        } else {
            break;
        }
        {
            pos_6 = ht_1;
            loop {
                let _e709 = pos_6;
                if (_e709 < width_1) {
                } else {
                    break;
                }
                {
                    let _e713 = g_1;
                    let _e715 = pos_6;
                    let slot_5 = ((_e713 * width_1) + _e715);
                    let _e719 = chance(row_10, slot_5, STREAM_DC_HIT, _e20.dc_point);
                    if _e719 {
                        let _e721 = g_1;
                        let _e723 = pos_6;
                        let _e724 = at(base_1, _e721, _e723);
                        let _e728 = gp.n_rnc;
                        let _e729 = below(row_10, slot_5, STREAM_DC_VALUE, _e728);
                        genome[_e724] = _e729;
                    }
                }
                continuing {
                    let _e734 = pos_6;
                    pos_6 = (_e734 + 1u);
                }
            }
        }
        continuing {
            let _e738 = g_1;
            g_1 = (_e738 + 1u);
        }
    }
    let _e743 = chance(row_10, OP_INVERT_DC, STREAM_OPERATOR, _e20.invert_dc);
    if ((t_1 >= 2u) && _e743) {
        let _e751 = gp.n_genes;
        let _e752 = below(row_10, 0u, STREAM_INVERT_DC, _e751);
        let _e757 = below(row_10, 1u, STREAM_INVERT_DC, (t_1 - 1u));
        let len_3 = (2u + _e757);
        let _e765 = below(row_10, 2u, STREAM_INVERT_DC, ((t_1 - len_3) + 1u));
        let start_1 = (ht_1 + _e765);
        let _e767 = at(base_1, _e752, start_1);
        let _e771 = at(base_1, _e752, ((start_1 + len_3) - 1u));
        reverse(_e767, _e771);
    }
    let _e775 = chance(row_10, OP_TRANSPOSE_DC, STREAM_OPERATOR, _e20.transpose_dc);
    if _e775 {
        let _e780 = gp.n_genes;
        let _e781 = below(row_10, 0u, STREAM_TRANSPOSE_DC, _e780);
        let _e786 = gp.n_genes;
        let _e787 = below(row_10, 1u, STREAM_TRANSPOSE_DC, _e786);
        let _e790 = below(row_10, 2u, STREAM_TRANSPOSE_DC, t_1);
        let len_4 = (1u + _e790);
        let def_b_2 = ((t_1 - len_4) + 1u);
        let _e798 = below(row_10, 3u, STREAM_TRANSPOSE_DC, def_b_2);
        let start_2 = (ht_1 + _e798);
        let _e802 = below(row_10, 4u, STREAM_TRANSPOSE_DC, def_b_2);
        let ins_1 = (ht_1 + _e802);
        loop {
            let _e805 = i_10;
            if (_e805 < len_4) {
            } else {
                break;
            }
            {
                let _e808 = i_10;
                let _e810 = at(base_1, _e781, (start_2 + _e808));
                let _e814 = i_10;
                let _e817 = genome[_e810];
                segment[_e814] = _e817;
            }
            continuing {
                let _e820 = i_10;
                i_10 = (_e820 + 1u);
            }
        }
        loop {
            let _e823 = i_11;
            if (_e823 < t_1) {
            } else {
                break;
            }
            {
                let _e826 = i_11;
                let _e828 = at(base_1, _e787, (ht_1 + _e826));
                let _e832 = i_11;
                let _e835 = genome[_e828];
                before[_e832] = _e835;
            }
            continuing {
                let _e838 = i_11;
                i_11 = (_e838 + 1u);
            }
        }
        pos_7 = (ins_1 + len_4);
        loop {
            let _e843 = pos_7;
            if (_e843 < width_1) {
            } else {
                break;
            }
            {
                let _e846 = pos_7;
                let _e847 = at(base_1, _e787, _e846);
                let _e852 = pos_7;
                let _e856 = before[((_e852 - len_4) - ht_1)];
                genome[_e847] = _e856;
            }
            continuing {
                let _e859 = pos_7;
                pos_7 = (_e859 + 1u);
            }
        }
        loop {
            let _e862 = i_12;
            if (_e862 < len_4) {
            } else {
                break;
            }
            {
                let _e865 = i_12;
                let _e867 = at(base_1, _e787, (ins_1 + _e865));
                let _e872 = i_12;
                let _e874 = segment[_e872];
                genome[_e867] = _e874;
            }
            continuing {
                let _e877 = i_12;
                i_12 = (_e877 + 1u);
            }
        }
    }
    loop {
        let _e880 = k_2;
        if (_e880 < rnc_w_1) {
        } else {
            break;
        }
        {
            let _e883 = k_2;
            let _e886 = chance(row_10, _e883, STREAM_RNC_HIT, _e20.rnc_point);
            if _e886 {
                let _e888 = k_2;
                let _e892 = gp.rnc_span;
                let _e893 = below(row_10, _e888, STREAM_RNC_VALUE, _e892);
                let _e897 = k_2;
                let _e901 = gp.rnc_lo;
                rnc[(rbase + _e897)] = f32((_e901 + i32(_e893)));
            }
        }
        continuing {
            let _e907 = k_2;
            k_2 = (_e907 + 1u);
        }
    }
    let _e912 = chance(row_10, OP_CLEANSE, STREAM_OPERATOR, _e20.cleanse);
    if _e912 {
        let _e917 = gp.n_genes;
        let _e918 = below(row_10, 0u, STREAM_CLEANSE, _e917);
        cleanse_gene(row_10, (base_1 + (_e918 * width_1)), _e20.cleanse_collapse);
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
