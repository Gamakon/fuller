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
    return vec2<u32>((((a & 65535u) * (b & 65535u)) + ((((a & 65535u) * (b >> 16u)) + ((a >> 16u) * (b & 65535u))) << 16u)), ((((a >> 16u) * (b >> 16u)) + (((((a & 65535u) * (b >> 16u)) + ((a >> 16u) * (b & 65535u))) >> 16u) | (select(0u, 1u, ((((a & 65535u) * (b >> 16u)) + ((a >> 16u) * (b & 65535u))) < ((a & 65535u) * (b >> 16u)))) << 16u))) + select(0u, 1u, ((((a & 65535u) * (b & 65535u)) + ((((a & 65535u) * (b >> 16u)) + ((a >> 16u) * (b & 65535u))) << 16u)) < ((a & 65535u) * (b & 65535u))))));
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
    return (_e10.y + select(0u, 1u, ((_e7.y + _e10.x) < _e7.y)));
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

    let _e5 = gp.generation;
    let _e7 = cohort_now[row_4];
    if (_e5 > _e7) {
        let _e14 = gp.generation;
        let _e16 = cohort_now[row_4];
        age = (_e14 - _e16);
    }
    let _e20 = age;
    if (_e20 >= merge) {
        return ELDERS;
    }
    let _e26 = cohort_now[row_4];
    return _e26;
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
    let _e5 = fitness_now[row_5];
    if ((bitcast<u32>(_e5) & 2147483647u) > 2139095040u) {
        return F32_MAX;
    }
    let _e14 = fitness_now[row_5];
    return min(_e14, F32_MAX);
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
        let _e18 = a_5;
        let _e20 = b_5;
        let _e22 = genome[_e20];
        genome[_e18] = _e22;
        let _e26 = b_5;
        let _e28 = a_5;
        let _e30 = genome[_e28];
        genome[_e26] = _e30;
        let _e33 = a_5;
        a_5 = (_e33 + 1u);
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

    let _e19 = gp.head;
    let _e21 = gp.tail;
    if ((_e19 + _e21) > MAX_CLEANSE) {
        return;
    }
    loop {
        let _e28 = need;
        let _e30 = n_2;
        let _e32 = gp.head;
        let _e34 = gp.tail;
        if ((_e28 <= 0i) || (_e30 >= (_e32 + _e34))) {
            break;
        }
        let _e44 = need;
        let _e45 = n_2;
        let _e48 = genome[(gene + _e45)];
        let _e50 = arity[_e48];
        need = ((_e44 + i32(_e50)) - 1i);
        let _e56 = n_2;
        n_2 = (_e56 + 1u);
    }
    let _e60 = need;
    if (_e60 > 0i) {
        return;
    }
    loop {
        let _e64 = i_14;
        let _e65 = n_2;
        if (_e64 < _e65) {
        } else {
            break;
        }
        {
            let _e70 = i_14;
            let _e72 = next_child_1;
            cl_child[_e70] = _e72;
            let _e78 = next_child_1;
            let _e79 = i_14;
            let _e82 = genome[(gene + _e79)];
            let _e84 = arity[_e82];
            next_child_1 = (_e78 + _e84);
            let _e89 = i_14;
            cl_ordinal[_e89] = NONE;
            let _e97 = gp.rnc_id;
            let _e99 = i_14;
            let _e102 = genome[(gene + _e99)];
            let _e104 = gp.rnc_id;
            if ((_e97 != NONE) && (_e102 == _e104)) {
                let _e110 = i_14;
                let _e112 = n_rnc;
                cl_ordinal[_e110] = _e112;
                let _e115 = n_rnc;
                n_rnc = (_e115 + 1u);
            }
            let _e122 = i_14;
            let _e125 = genome[(gene + _e122)];
            let _e127 = arity[_e125];
            if (_e127 > 0u) {
                let _e131 = n_fn_1;
                n_fn_1 = (_e131 + 1u);
            }
        }
        continuing {
            let _e135 = i_14;
            i_14 = (_e135 + 1u);
        }
    }
    let _e139 = n_fn_1;
    if (_e139 == 0u) {
        return;
    }
    let _e145 = n_fn_1;
    let _e146 = below(row_7, 1u, STREAM_CLEANSE, _e145);
    loop {
        let _e149 = i_15;
        let _e150 = n_2;
        if (_e149 < _e150) {
        } else {
            break;
        }
        {
            let _e157 = i_15;
            let _e160 = genome[(gene + _e157)];
            let _e162 = arity[_e160];
            if (_e162 > 0u) {
                let _e165 = seen_1;
                if (_e165 == _e146) {
                    let _e169 = i_15;
                    p = _e169;
                }
                let _e172 = seen_1;
                seen_1 = (_e172 + 1u);
            }
        }
        continuing {
            let _e176 = i_15;
            i_15 = (_e176 + 1u);
        }
    }
    let _e182 = chance(row_7, 2u, STREAM_CLEANSE, collapse_thr);
    let _e186 = gp.rnc_id;
    if !(((_e186 != NONE) && _e182)) {
        let _e197 = p;
        let _e200 = genome[(gene + _e197)];
        let _e202 = arity[_e200];
        let _e203 = below(row_7, 3u, STREAM_CLEANSE, _e202);
        let _e207 = p;
        let _e209 = cl_child[_e207];
        q_1 = (_e209 + _e203);
    }
    let _e216 = gp.n_rnc;
    let _e217 = below(row_7, 4u, STREAM_CLEANSE, _e216);
    let _e223 = q_1;
    let _e224 = p;
    cl_queue[0] = select(0u, _e223, (_e224 == 0u));
    loop {
        let _e229 = head;
        let _e230 = tail;
        if (_e229 >= _e230) {
            break;
        }
        let _e234 = head;
        head = (_e234 + 1u);
        let _e239 = head;
        let _e241 = cl_queue[_e239];
        if (_e241 == LEAF) {
            let _e246 = m;
            let _e249 = gp.rnc_id;
            cl_tok[_e246] = _e249;
            let _e252 = k_3;
            cl_dc[_e252] = _e217;
            let _e256 = k_3;
            k_3 = (_e256 + 1u);
            let _e260 = m;
            m = (_e260 + 1u);
            continue;
        }
        let _e268 = m;
        let _e270 = head;
        let _e272 = cl_queue[_e270];
        let _e275 = genome[(gene + _e272)];
        cl_tok[_e268] = _e275;
        let _e280 = head;
        let _e282 = cl_queue[_e280];
        let _e284 = cl_ordinal[_e282];
        if (_e284 != NONE) {
            let _e295 = k_3;
            let _e298 = gp.head;
            let _e300 = gp.tail;
            let _e303 = head;
            let _e305 = cl_queue[_e303];
            let _e307 = cl_ordinal[_e305];
            let _e310 = genome[((gene + (_e298 + _e300)) + _e307)];
            let _e311 = head;
            let _e313 = cl_queue[_e311];
            let _e315 = cl_ordinal[_e313];
            let _e317 = gp.tail;
            cl_dc[_e295] = select(0u, _e310, (_e315 < _e317));
            let _e322 = k_3;
            k_3 = (_e322 + 1u);
        }
        let _e326 = m;
        m = (_e326 + 1u);
        j = 0u;
        loop {
            let _e336 = j;
            let _e337 = head;
            let _e339 = cl_queue[_e337];
            let _e342 = genome[(gene + _e339)];
            let _e344 = arity[_e342];
            if (_e336 < _e344) {
            } else {
                break;
            }
            {
                let _e353 = tail;
                let _e355 = q_1;
                let _e356 = head;
                let _e358 = cl_queue[_e356];
                let _e360 = cl_child[_e358];
                let _e361 = j;
                let _e363 = head;
                let _e365 = cl_queue[_e363];
                let _e367 = cl_child[_e365];
                let _e368 = j;
                let _e370 = p;
                cl_queue[_e353] = select((_e360 + _e361), _e355, ((_e367 + _e368) == _e370));
                let _e375 = tail;
                tail = (_e375 + 1u);
            }
            continuing {
                let _e379 = j;
                j = (_e379 + 1u);
            }
        }
    }
    let _e383 = k_3;
    let _e385 = gp.tail;
    if (_e383 > _e385) {
        return;
    }
    let _e390 = gp.vhead;
    x_1 = _e390;
    loop {
        let _e393 = x_1;
        let _e394 = m;
        if (_e393 < _e394) {
        } else {
            break;
        }
        {
            let _e400 = x_1;
            let _e402 = cl_tok[_e400];
            let _e404 = arity[_e402];
            if (_e404 > 0u) {
                return;
            }
        }
        continuing {
            let _e408 = x_1;
            x_1 = (_e408 + 1u);
        }
    }
    loop {
        let _e412 = x_2;
        let _e413 = m;
        if (_e412 < _e413) {
        } else {
            break;
        }
        {
            let _e419 = x_2;
            let _e422 = x_2;
            let _e424 = cl_tok[_e422];
            genome[(gene + _e419)] = _e424;
        }
        continuing {
            let _e427 = x_2;
            x_2 = (_e427 + 1u);
        }
    }
    loop {
        let _e431 = x_3;
        let _e432 = k_3;
        if (_e431 < _e432) {
        } else {
            break;
        }
        {
            let _e440 = gp.head;
            let _e442 = gp.tail;
            let _e445 = x_3;
            let _e448 = x_3;
            let _e450 = cl_dc[_e448];
            genome[((gene + (_e440 + _e442)) + _e445)] = _e450;
        }
        continuing {
            let _e453 = x_3;
            x_3 = (_e453 + 1u);
        }
    }
    return;
}

fn swap_tokens(a_6: u32, b_6: u32, g_3: u32, from_pos: u32, to_pos: u32) {
    var pos_10: u32;

    pos_10 = from_pos;
    loop {
        let _e4 = pos_10;
        if (_e4 < to_pos) {
        } else {
            break;
        }
        {
            let _e14 = gp.n_genes;
            let _e16 = gp.head;
            let _e18 = gp.tail;
            let _e24 = gp.head;
            let _e26 = gp.tail;
            let _e31 = pos_10;
            let _e35 = gp.n_genes;
            let _e37 = gp.head;
            let _e39 = gp.tail;
            let _e45 = gp.head;
            let _e47 = gp.tail;
            let _e52 = pos_10;
            let _e55 = genome[(((b_6 * (_e35 * (_e37 + (2u * _e39)))) + (g_3 * (_e45 + (2u * _e47)))) + _e52)];
            genome[(((a_6 * (_e14 * (_e16 + (2u * _e18)))) + (g_3 * (_e24 + (2u * _e26)))) + _e31)] = _e55;
            let _e64 = gp.n_genes;
            let _e66 = gp.head;
            let _e68 = gp.tail;
            let _e74 = gp.head;
            let _e76 = gp.tail;
            let _e81 = pos_10;
            let _e85 = gp.n_genes;
            let _e87 = gp.head;
            let _e89 = gp.tail;
            let _e95 = gp.head;
            let _e97 = gp.tail;
            let _e102 = pos_10;
            let _e105 = genome[(((a_6 * (_e85 * (_e87 + (2u * _e89)))) + (g_3 * (_e95 + (2u * _e97)))) + _e102)];
            genome[(((b_6 * (_e64 * (_e66 + (2u * _e68)))) + (g_3 * (_e74 + (2u * _e76)))) + _e81)] = _e105;
        }
        continuing {
            let _e108 = pos_10;
            pos_10 = (_e108 + 1u);
        }
    }
    return;
}

fn swap_consts(a_7: u32, ga: u32, b_7: u32, gb: u32) {
    var k_4: u32 = 0u;

    loop {
        let _e3 = k_4;
        let _e5 = gp.n_rnc;
        if (_e3 < _e5) {
        } else {
            break;
        }
        {
            let _e15 = gp.n_genes;
            let _e17 = gp.n_rnc;
            let _e21 = gp.n_rnc;
            let _e24 = k_4;
            let _e28 = gp.n_genes;
            let _e30 = gp.n_rnc;
            let _e34 = gp.n_rnc;
            let _e37 = k_4;
            let _e40 = rnc[(((b_7 * (_e28 * _e30)) + (gb * _e34)) + _e37)];
            rnc[(((a_7 * (_e15 * _e17)) + (ga * _e21)) + _e24)] = _e40;
            let _e49 = gp.n_genes;
            let _e51 = gp.n_rnc;
            let _e55 = gp.n_rnc;
            let _e58 = k_4;
            let _e62 = gp.n_genes;
            let _e64 = gp.n_rnc;
            let _e68 = gp.n_rnc;
            let _e71 = k_4;
            let _e74 = rnc[(((a_7 * (_e62 * _e64)) + (ga * _e68)) + _e71)];
            rnc[(((b_7 * (_e49 * _e51)) + (gb * _e55)) + _e58)] = _e74;
        }
        continuing {
            let _e77 = k_4;
            k_4 = (_e77 + 1u);
        }
    }
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn first_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var w: u32 = NONE;
    var i: u32 = 0u;

    let _e6 = gp.pop;
    if (gid.x >= _e6) {
        return;
    }
    let _e10 = island_of(gid.x);
    loop {
        let _e12 = i;
        if (_e12 < _e10.tournsize) {
        } else {
            break;
        }
        {
            let _e18 = i;
            let _e23 = below(gid.x, _e18, STREAM_SELECT_1_, (_e10.hi - _e10.lo));
            let _e29 = w;
            let _e34 = better_mate(gid.x, (_e10.lo + _e23), _e29, (_e10.open_fight == 1u), _e10.cohort_merge);
            let _e37 = w;
            if ((_e37 == NONE) || _e34) {
                w = (_e10.lo + _e23);
            }
        }
        continuing {
            let _e45 = i;
            i = (_e45 + 1u);
        }
    }
    let _e52 = w;
    stage1_[gid.x] = _e52;
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

    let _e8 = gp.pop;
    if (gid_1.x >= _e8) {
        return;
    }
    let _e12 = island_of(gid_1.x);
    if (gid_1.x < (_e12.lo + _e12.elites)) {
        let _e24 = elite_src[gid_1.x];
        if (_e24 != NONE) {
            let _e33 = elite_src[gid_1.x];
            parent[gid_1.x] = _e33;
            return;
        }
        loop {
            let _e36 = e;
            if (_e36 <= (gid_1.x - _e12.lo)) {
            } else {
                break;
            }
            {
                best = NONE;
                r = _e12.lo;
                loop {
                    let _e46 = r;
                    if (_e46 < _e12.hi) {
                    } else {
                        break;
                    }
                    {
                        used = false;
                        q = 0u;
                        loop {
                            let _e55 = q;
                            let _e56 = e;
                            if (_e55 < _e56) {
                            } else {
                                break;
                            }
                            {
                                let _e61 = q;
                                let _e63 = taken[_e61];
                                let _e64 = r;
                                if (_e63 == _e64) {
                                    used = true;
                                }
                            }
                            continuing {
                                let _e70 = q;
                                q = (_e70 + 1u);
                            }
                        }
                        let _e73 = r;
                        let _e74 = key(_e73);
                        let _e76 = best;
                        let _e77 = key(_e76);
                        let _e81 = used;
                        let _e83 = best;
                        if (!(_e81) && ((_e83 == NONE) || (_e74 < _e77))) {
                            let _e90 = r;
                            best = _e90;
                        }
                    }
                    continuing {
                        let _e93 = r;
                        r = (_e93 + 1u);
                    }
                }
                let _e98 = e;
                let _e100 = best;
                taken[_e98] = _e100;
            }
            continuing {
                let _e103 = e;
                e = (_e103 + 1u);
            }
        }
        let _e110 = best;
        parent[gid_1.x] = _e110;
        return;
    }
    if ((((_e12.arrivals > 0u) && (_e12.arrival_children > 0u)) && (gid_1.x >= (_e12.lo + _e12.elites))) && (gid_1.x < ((_e12.lo + _e12.elites) + (_e12.arrivals * (_e12.arrival_children + 1u))))) {
        if (((gid_1.x - (_e12.lo + _e12.elites)) % (_e12.arrival_children + 1u)) == 0u) {
            parent[gid_1.x] = gid_1.x;
        } else {
            let _e169 = below(gid_1.x, ((gid_1.x - (_e12.lo + _e12.elites)) % (_e12.arrival_children + 1u)), STREAM_ARRIVAL_MATE, (_e12.hi - _e12.lo));
            parent[gid_1.x] = (_e12.lo + _e169);
        }
        return;
    }
    loop {
        let _e177 = t;
        if (_e177 < _e12.tournsize) {
        } else {
            break;
        }
        {
            let _e183 = t;
            let _e188 = below(gid_1.x, _e183, STREAM_SELECT_2_, (_e12.hi - _e12.lo));
            let _e195 = stage1_[(_e12.lo + _e188)];
            let _e197 = w_1;
            let _e202 = better_mate(gid_1.x, _e195, _e197, (_e12.open_fight == 1u), _e12.cohort_merge);
            let _e205 = w_1;
            if ((_e205 == NONE) || _e202) {
                let _e213 = stage1_[(_e12.lo + _e188)];
                w_1 = _e213;
            }
        }
        continuing {
            let _e216 = t;
            t = (_e216 + 1u);
        }
    }
    let _e223 = w_1;
    parent[gid_1.x] = _e223;
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

    let _e18 = gp.pop;
    if (gid_2.x >= _e18) {
        return;
    }
    let _e22 = island_of(gid_2.x);
    loop {
        let _e26 = i_1;
        let _e28 = gp.n_genes;
        let _e30 = gp.head;
        let _e32 = gp.tail;
        if (_e26 < (_e28 * (_e30 + (2u * _e32)))) {
        } else {
            break;
        }
        {
            let _e46 = gp.n_genes;
            let _e48 = gp.head;
            let _e50 = gp.tail;
            let _e55 = i_1;
            let _e60 = parent[gid_2.x];
            let _e62 = gp.n_genes;
            let _e64 = gp.head;
            let _e66 = gp.tail;
            let _e71 = i_1;
            let _e74 = genome_now[((_e60 * (_e62 * (_e64 + (2u * _e66)))) + _e71)];
            genome[((gid_2.x * (_e46 * (_e48 + (2u * _e50)))) + _e55)] = _e74;
        }
        continuing {
            let _e77 = i_1;
            i_1 = (_e77 + 1u);
        }
    }
    loop {
        let _e81 = i_2;
        let _e83 = gp.n_genes;
        let _e85 = gp.n_rnc;
        if (_e81 < (_e83 * _e85)) {
        } else {
            break;
        }
        {
            let _e96 = gp.n_genes;
            let _e98 = gp.n_rnc;
            let _e101 = i_2;
            let _e106 = parent[gid_2.x];
            let _e108 = gp.n_genes;
            let _e110 = gp.n_rnc;
            let _e113 = i_2;
            let _e116 = rnc_now[((_e106 * (_e108 * _e110)) + _e113)];
            rnc[((gid_2.x * (_e96 * _e98)) + _e101)] = _e116;
        }
        continuing {
            let _e119 = i_2;
            i_2 = (_e119 + 1u);
        }
    }
    let _e129 = parent[gid_2.x];
    let _e131 = wrapper_now[_e129];
    wrapper_id[gid_2.x] = _e131;
    let _e140 = parent[gid_2.x];
    let _e142 = cohort_now[_e140];
    cohort[gid_2.x] = _e142;
    if (gid_2.x < (_e22.lo + _e22.elites)) {
        let _e157 = parent[gid_2.x];
        let _e159 = fitness_now[_e157];
        fitness[gid_2.x] = _e159;
        return;
    }
    fitness[gid_2.x] = bitcast<f32>(NAN_BITS);
    loop {
        let _e168 = g;
        let _e170 = gp.n_genes;
        if (_e168 < _e170) {
        } else {
            break;
        }
        {
            let _e176 = gp.typed_depth;
            let _e179 = gp.head;
            let _e181 = gp.tail;
            if ((_e176 > 0u) && ((_e179 + _e181) <= MAX_HT)) {
                i_3 = 0u;
                loop {
                    let _e189 = i_3;
                    let _e191 = gp.head;
                    let _e193 = gp.tail;
                    if (_e189 < (_e191 + _e193)) {
                    } else {
                        break;
                    }
                    {
                        let _e199 = i_3;
                        let _e202 = gp.typed_depth;
                        mut_budget[_e199] = _e202;
                    }
                    continuing {
                        let _e205 = i_3;
                        i_3 = (_e205 + 1u);
                    }
                }
            }
            next_child = 1u;
            pos = 0u;
            loop {
                let _e213 = pos;
                let _e215 = gp.head;
                let _e217 = gp.tail;
                if (_e213 < (_e215 + _e217)) {
                } else {
                    break;
                }
                {
                    let _e226 = g;
                    let _e228 = gp.head;
                    let _e230 = gp.tail;
                    let _e234 = pos;
                    let _e238 = chance(gid_2.x, ((_e226 * (_e228 + (2u * _e230))) + _e234), STREAM_MUT_HIT, _e22.mut_point);
                    if _e238 {
                        let _e245 = g;
                        let _e247 = gp.head;
                        let _e249 = gp.tail;
                        let _e253 = pos;
                        let _e256 = coin(gid_2.x, ((_e245 * (_e247 + (2u * _e249))) + _e253), STREAM_MUT_KIND);
                        let _e259 = pos;
                        let _e261 = gp.vhead;
                        if ((_e259 < _e261) && _e256) {
                            let _e270 = gp.typed_depth;
                            let _e273 = gp.head;
                            let _e275 = gp.tail;
                            let _e279 = pos;
                            let _e281 = mut_budget[_e279];
                            if (((_e270 > 0u) && ((_e273 + _e275) <= MAX_HT)) && (_e281 == 0u)) {
                                let _e287 = gp.n_flat;
                                if (_e287 == 0u) {
                                    let _e294 = gp.n_genes;
                                    let _e296 = gp.head;
                                    let _e298 = gp.tail;
                                    let _e304 = g;
                                    let _e306 = pos;
                                    let _e307 = at((gid_2.x * (_e294 * (_e296 + (2u * _e298)))), _e304, _e306);
                                    let _e314 = g;
                                    let _e316 = gp.head;
                                    let _e318 = gp.tail;
                                    let _e322 = pos;
                                    let _e327 = gp.n_terminals;
                                    let _e328 = below(gid_2.x, ((_e314 * (_e316 + (2u * _e318))) + _e322), STREAM_MUT_SYMBOL, _e327);
                                    let _e333 = sample_terminals[_e328];
                                    genome[_e307] = _e333;
                                } else {
                                    let _e339 = gp.n_genes;
                                    let _e341 = gp.head;
                                    let _e343 = gp.tail;
                                    let _e349 = g;
                                    let _e351 = pos;
                                    let _e352 = at((gid_2.x * (_e339 * (_e341 + (2u * _e343)))), _e349, _e351);
                                    let _e359 = g;
                                    let _e361 = gp.head;
                                    let _e363 = gp.tail;
                                    let _e367 = pos;
                                    let _e372 = gp.n_flat;
                                    let _e373 = below(gid_2.x, ((_e359 * (_e361 + (2u * _e363))) + _e367), STREAM_MUT_SYMBOL, _e372);
                                    let _e378 = sample_flat[_e373];
                                    genome[_e352] = _e378;
                                }
                            } else {
                                let _e384 = gp.n_genes;
                                let _e386 = gp.head;
                                let _e388 = gp.tail;
                                let _e394 = g;
                                let _e396 = pos;
                                let _e397 = at((gid_2.x * (_e384 * (_e386 + (2u * _e388)))), _e394, _e396);
                                let _e404 = g;
                                let _e406 = gp.head;
                                let _e408 = gp.tail;
                                let _e412 = pos;
                                let _e417 = gp.n_functions;
                                let _e418 = below(gid_2.x, ((_e404 * (_e406 + (2u * _e408))) + _e412), STREAM_MUT_SYMBOL, _e417);
                                let _e423 = sample_functions[_e418];
                                genome[_e397] = _e423;
                            }
                        } else {
                            let _e429 = gp.n_genes;
                            let _e431 = gp.head;
                            let _e433 = gp.tail;
                            let _e439 = g;
                            let _e441 = pos;
                            let _e442 = at((gid_2.x * (_e429 * (_e431 + (2u * _e433)))), _e439, _e441);
                            let _e449 = g;
                            let _e451 = gp.head;
                            let _e453 = gp.tail;
                            let _e457 = pos;
                            let _e462 = gp.n_terminals;
                            let _e463 = below(gid_2.x, ((_e449 * (_e451 + (2u * _e453))) + _e457), STREAM_MUT_SYMBOL, _e462);
                            let _e468 = sample_terminals[_e463];
                            genome[_e442] = _e468;
                        }
                    }
                    let _e475 = gp.typed_depth;
                    let _e478 = gp.head;
                    let _e480 = gp.tail;
                    let _e484 = pos;
                    let _e485 = next_child;
                    if (((_e475 > 0u) && ((_e478 + _e480) <= MAX_HT)) && (_e484 < _e485)) {
                        let _e493 = gp.n_genes;
                        let _e495 = gp.head;
                        let _e497 = gp.tail;
                        let _e503 = g;
                        let _e505 = pos;
                        let _e506 = at((gid_2.x * (_e493 * (_e495 + (2u * _e497)))), _e503, _e505);
                        k = 0u;
                        loop {
                            let _e512 = k;
                            let _e514 = genome[_e506];
                            let _e516 = arity[_e514];
                            if (_e512 < _e516) {
                            } else {
                                break;
                            }
                            {
                                let _e520 = next_child;
                                let _e522 = gp.head;
                                let _e524 = gp.tail;
                                if (_e520 < (_e522 + _e524)) {
                                    let _e532 = next_child;
                                    let _e534 = pos;
                                    let _e536 = mut_budget[_e534];
                                    let _e537 = pos;
                                    let _e539 = mut_budget[_e537];
                                    let _e541 = genome[_e506];
                                    let _e543 = depth_cost[_e541];
                                    mut_budget[_e532] = (_e536 - min(_e539, _e543));
                                }
                                let _e548 = next_child;
                                next_child = (_e548 + 1u);
                            }
                            continuing {
                                let _e552 = k;
                                k = (_e552 + 1u);
                            }
                        }
                    }
                }
                continuing {
                    let _e556 = pos;
                    pos = (_e556 + 1u);
                }
            }
        }
        continuing {
            let _e560 = g;
            g = (_e560 + 1u);
        }
    }
    let _e567 = chance(gid_2.x, OP_INVERT, STREAM_OPERATOR, _e22.invert);
    if _e567 {
        let _e574 = gp.n_genes;
        let _e575 = below(gid_2.x, 0u, STREAM_INVERT, _e574);
        let _e583 = gp.vhead;
        let _e585 = below(gid_2.x, 1u, STREAM_INVERT, (_e583 - 1u));
        let _e594 = gp.vhead;
        let _e598 = below(gid_2.x, 2u, STREAM_INVERT, ((_e594 - (2u + _e585)) + 1u));
        let _e604 = gp.n_genes;
        let _e606 = gp.head;
        let _e608 = gp.tail;
        let _e613 = at((gid_2.x * (_e604 * (_e606 + (2u * _e608)))), _e575, _e598);
        let _e619 = gp.n_genes;
        let _e621 = gp.head;
        let _e623 = gp.tail;
        let _e631 = at((gid_2.x * (_e619 * (_e621 + (2u * _e623)))), _e575, (_e598 + (2u + _e585)));
        reverse(_e613, _e631);
    }
    let _e637 = chance(gid_2.x, OP_IS, STREAM_OPERATOR, _e22.is_transpose);
    if _e637 {
        let _e644 = gp.n_genes;
        let _e645 = below(gid_2.x, 0u, STREAM_IS, _e644);
        let _e652 = gp.n_genes;
        let _e653 = below(gid_2.x, 1u, STREAM_IS, _e652);
        let _e661 = gp.vhead;
        let _e663 = below(gid_2.x, 2u, STREAM_IS, (_e661 - 1u));
        let _e671 = gp.head;
        let _e673 = gp.tail;
        let _e678 = below(gid_2.x, 3u, STREAM_IS, (((_e671 + _e673) - (1u + _e663)) + 1u));
        let _e686 = gp.vhead;
        let _e689 = below(gid_2.x, 4u, STREAM_IS, (_e686 - (1u + _e663)));
        loop {
            let _e692 = i_4;
            if (_e692 < (1u + _e663)) {
            } else {
                break;
            }
            {
                let _e700 = gp.n_genes;
                let _e702 = gp.head;
                let _e704 = gp.tail;
                let _e710 = i_4;
                let _e712 = at((gid_2.x * (_e700 * (_e702 + (2u * _e704)))), _e645, (_e678 + _e710));
                let _e716 = i_4;
                let _e719 = genome[_e712];
                segment[_e716] = _e719;
            }
            continuing {
                let _e722 = i_4;
                i_4 = (_e722 + 1u);
            }
        }
        loop {
            let _e726 = i_5;
            let _e728 = gp.vhead;
            if (_e726 < _e728) {
            } else {
                break;
            }
            {
                let _e735 = gp.n_genes;
                let _e737 = gp.head;
                let _e739 = gp.tail;
                let _e745 = i_5;
                let _e746 = at((gid_2.x * (_e735 * (_e737 + (2u * _e739)))), _e653, _e745);
                let _e750 = i_5;
                let _e753 = genome[_e746];
                before[_e750] = _e753;
            }
            continuing {
                let _e756 = i_5;
                i_5 = (_e756 + 1u);
            }
        }
        pos_1 = ((1u + _e689) + (1u + _e663));
        loop {
            let _e765 = pos_1;
            let _e767 = gp.vhead;
            if (_e765 < _e767) {
            } else {
                break;
            }
            {
                let _e774 = gp.n_genes;
                let _e776 = gp.head;
                let _e778 = gp.tail;
                let _e784 = pos_1;
                let _e785 = at((gid_2.x * (_e774 * (_e776 + (2u * _e778)))), _e653, _e784);
                let _e791 = pos_1;
                let _e795 = before[(_e791 - (1u + _e663))];
                genome[_e785] = _e795;
            }
            continuing {
                let _e798 = pos_1;
                pos_1 = (_e798 + 1u);
            }
        }
        loop {
            let _e802 = i_6;
            if (_e802 < (1u + _e663)) {
            } else {
                break;
            }
            {
                let _e810 = gp.n_genes;
                let _e812 = gp.head;
                let _e814 = gp.tail;
                let _e822 = i_6;
                let _e824 = at((gid_2.x * (_e810 * (_e812 + (2u * _e814)))), _e653, ((1u + _e689) + _e822));
                let _e829 = i_6;
                let _e831 = segment[_e829];
                genome[_e824] = _e831;
            }
            continuing {
                let _e834 = i_6;
                i_6 = (_e834 + 1u);
            }
        }
    }
    let _e841 = chance(gid_2.x, OP_RIS, STREAM_OPERATOR, _e22.ris_transpose);
    if _e841 {
        loop {
            let _e846 = trial;
            let _e848 = gp.n_genes;
            if (_e846 < ((2u * _e848) + 1u)) {
            } else {
                break;
            }
            {
                let _e856 = trial;
                let _e861 = gp.n_genes;
                let _e862 = below(gid_2.x, (_e856 * 4u), STREAM_RIS, _e861);
                let _e868 = trial;
                let _e874 = gp.n_genes;
                let _e875 = below(gid_2.x, ((_e868 * 4u) + 1u), STREAM_RIS, _e874);
                n_fn = 0u;
                pos_2 = 0u;
                loop {
                    let _e882 = pos_2;
                    let _e884 = gp.vhead;
                    if (_e882 < _e884) {
                    } else {
                        break;
                    }
                    {
                        let _e891 = gp.n_genes;
                        let _e893 = gp.head;
                        let _e895 = gp.tail;
                        let _e901 = pos_2;
                        let _e902 = at((gid_2.x * (_e891 * (_e893 + (2u * _e895)))), _e862, _e901);
                        let _e907 = genome[_e902];
                        let _e909 = arity[_e907];
                        if (_e909 > 0u) {
                            let _e913 = n_fn;
                            n_fn = (_e913 + 1u);
                        }
                    }
                    continuing {
                        let _e917 = pos_2;
                        pos_2 = (_e917 + 1u);
                    }
                }
                let _e921 = n_fn;
                if (_e921 == 0u) {
                    continue;
                }
                let _e928 = trial;
                let _e933 = n_fn;
                let _e934 = below(gid_2.x, ((_e928 * 4u) + 2u), STREAM_RIS, _e933);
                start = 0u;
                seen = 0u;
                pos_3 = 0u;
                loop {
                    let _e943 = pos_3;
                    let _e945 = gp.vhead;
                    if (_e943 < _e945) {
                    } else {
                        break;
                    }
                    {
                        let _e952 = gp.n_genes;
                        let _e954 = gp.head;
                        let _e956 = gp.tail;
                        let _e962 = pos_3;
                        let _e963 = at((gid_2.x * (_e952 * (_e954 + (2u * _e956)))), _e862, _e962);
                        let _e968 = genome[_e963];
                        let _e970 = arity[_e968];
                        if (_e970 > 0u) {
                            let _e973 = seen;
                            if (_e973 == _e934) {
                                let _e977 = pos_3;
                                start = _e977;
                            }
                            let _e980 = seen;
                            seen = (_e980 + 1u);
                        }
                    }
                    continuing {
                        let _e984 = pos_3;
                        pos_3 = (_e984 + 1u);
                    }
                }
                let _e991 = trial;
                let _e999 = gp.vhead;
                let _e1001 = gp.head;
                let _e1003 = gp.tail;
                let _e1005 = start;
                let _e1009 = below(gid_2.x, ((_e991 * 4u) + 3u), STREAM_RIS, (min(_e999, ((_e1001 + _e1003) - _e1005)) - 1u));
                i_7 = 0u;
                loop {
                    let _e1014 = i_7;
                    if (_e1014 < (2u + _e1009)) {
                    } else {
                        break;
                    }
                    {
                        let _e1022 = gp.n_genes;
                        let _e1024 = gp.head;
                        let _e1026 = gp.tail;
                        let _e1033 = start;
                        let _e1034 = i_7;
                        let _e1036 = at((gid_2.x * (_e1022 * (_e1024 + (2u * _e1026)))), _e862, (_e1033 + _e1034));
                        let _e1040 = i_7;
                        let _e1043 = genome[_e1036];
                        segment[_e1040] = _e1043;
                    }
                    continuing {
                        let _e1046 = i_7;
                        i_7 = (_e1046 + 1u);
                    }
                }
                i_8 = 0u;
                loop {
                    let _e1052 = i_8;
                    let _e1054 = gp.vhead;
                    if (_e1052 < _e1054) {
                    } else {
                        break;
                    }
                    {
                        let _e1061 = gp.n_genes;
                        let _e1063 = gp.head;
                        let _e1065 = gp.tail;
                        let _e1071 = i_8;
                        let _e1072 = at((gid_2.x * (_e1061 * (_e1063 + (2u * _e1065)))), _e875, _e1071);
                        let _e1076 = i_8;
                        let _e1079 = genome[_e1072];
                        before[_e1076] = _e1079;
                    }
                    continuing {
                        let _e1082 = i_8;
                        i_8 = (_e1082 + 1u);
                    }
                }
                pos_4 = (2u + _e1009);
                loop {
                    let _e1089 = pos_4;
                    let _e1091 = gp.vhead;
                    if (_e1089 < _e1091) {
                    } else {
                        break;
                    }
                    {
                        let _e1098 = gp.n_genes;
                        let _e1100 = gp.head;
                        let _e1102 = gp.tail;
                        let _e1108 = pos_4;
                        let _e1109 = at((gid_2.x * (_e1098 * (_e1100 + (2u * _e1102)))), _e875, _e1108);
                        let _e1115 = pos_4;
                        let _e1119 = before[(_e1115 - (2u + _e1009))];
                        genome[_e1109] = _e1119;
                    }
                    continuing {
                        let _e1122 = pos_4;
                        pos_4 = (_e1122 + 1u);
                    }
                }
                i_9 = 0u;
                loop {
                    let _e1128 = i_9;
                    if (_e1128 < (2u + _e1009)) {
                    } else {
                        break;
                    }
                    {
                        let _e1136 = gp.n_genes;
                        let _e1138 = gp.head;
                        let _e1140 = gp.tail;
                        let _e1146 = i_9;
                        let _e1147 = at((gid_2.x * (_e1136 * (_e1138 + (2u * _e1140)))), _e875, _e1146);
                        let _e1152 = i_9;
                        let _e1154 = segment[_e1152];
                        genome[_e1147] = _e1154;
                    }
                    continuing {
                        let _e1157 = i_9;
                        i_9 = (_e1157 + 1u);
                    }
                }
                break;
            }
            continuing {
                let _e1161 = trial;
                trial = (_e1161 + 1u);
            }
        }
    }
    let _e1168 = chance(gid_2.x, OP_GENE_T, STREAM_OPERATOR, _e22.gene_transpose);
    let _e1172 = gp.n_genes;
    if ((_e1172 > 1u) && _e1168) {
        let _e1182 = gp.n_genes;
        let _e1184 = below(gid_2.x, 0u, STREAM_GENE_T, (_e1182 - 1u));
        loop {
            let _e1188 = pos_5;
            let _e1190 = gp.head;
            let _e1192 = gp.tail;
            if (_e1188 < (_e1190 + (2u * _e1192))) {
            } else {
                break;
            }
            {
                let _e1201 = gp.n_genes;
                let _e1203 = gp.head;
                let _e1205 = gp.tail;
                let _e1212 = pos_5;
                let _e1213 = at((gid_2.x * (_e1201 * (_e1203 + (2u * _e1205)))), 0u, _e1212);
                let _e1219 = gp.n_genes;
                let _e1221 = gp.head;
                let _e1223 = gp.tail;
                let _e1230 = pos_5;
                let _e1231 = at((gid_2.x * (_e1219 * (_e1221 + (2u * _e1223)))), 0u, _e1230);
                let _e1237 = gp.n_genes;
                let _e1239 = gp.head;
                let _e1241 = gp.tail;
                let _e1249 = pos_5;
                let _e1250 = at((gid_2.x * (_e1237 * (_e1239 + (2u * _e1241)))), (1u + _e1184), _e1249);
                let _e1254 = genome[_e1250];
                genome[_e1231] = _e1254;
                let _e1260 = gp.n_genes;
                let _e1262 = gp.head;
                let _e1264 = gp.tail;
                let _e1272 = pos_5;
                let _e1273 = at((gid_2.x * (_e1260 * (_e1262 + (2u * _e1264)))), (1u + _e1184), _e1272);
                let _e1277 = genome[_e1213];
                genome[_e1273] = _e1277;
            }
            continuing {
                let _e1280 = pos_5;
                pos_5 = (_e1280 + 1u);
            }
        }
        loop {
            let _e1284 = k_1;
            let _e1286 = gp.n_rnc;
            if (_e1284 < _e1286) {
            } else {
                break;
            }
            {
                let _e1295 = gp.n_genes;
                let _e1297 = gp.n_rnc;
                let _e1300 = k_1;
                let _e1305 = gp.n_genes;
                let _e1307 = gp.n_rnc;
                let _e1312 = gp.n_rnc;
                let _e1315 = k_1;
                let _e1318 = rnc[(((gid_2.x * (_e1305 * _e1307)) + ((1u + _e1184) * _e1312)) + _e1315)];
                rnc[((gid_2.x * (_e1295 * _e1297)) + _e1300)] = _e1318;
                let _e1326 = gp.n_genes;
                let _e1328 = gp.n_rnc;
                let _e1333 = gp.n_rnc;
                let _e1336 = k_1;
                let _e1341 = gp.n_genes;
                let _e1343 = gp.n_rnc;
                let _e1346 = k_1;
                let _e1349 = rnc[((gid_2.x * (_e1341 * _e1343)) + _e1346)];
                rnc[(((gid_2.x * (_e1326 * _e1328)) + ((1u + _e1184) * _e1333)) + _e1336)] = _e1349;
            }
            continuing {
                let _e1352 = k_1;
                k_1 = (_e1352 + 1u);
            }
        }
    }
    loop {
        let _e1356 = g_1;
        let _e1358 = gp.n_genes;
        if (_e1356 < _e1358) {
        } else {
            break;
        }
        {
            let _e1363 = gp.head;
            let _e1365 = gp.tail;
            pos_6 = (_e1363 + _e1365);
            loop {
                let _e1370 = pos_6;
                let _e1372 = gp.head;
                let _e1374 = gp.tail;
                if (_e1370 < (_e1372 + (2u * _e1374))) {
                } else {
                    break;
                }
                {
                    let _e1384 = g_1;
                    let _e1386 = gp.head;
                    let _e1388 = gp.tail;
                    let _e1392 = pos_6;
                    let _e1396 = chance(gid_2.x, ((_e1384 * (_e1386 + (2u * _e1388))) + _e1392), STREAM_DC_HIT, _e22.dc_point);
                    if _e1396 {
                        let _e1402 = gp.n_genes;
                        let _e1404 = gp.head;
                        let _e1406 = gp.tail;
                        let _e1412 = g_1;
                        let _e1414 = pos_6;
                        let _e1415 = at((gid_2.x * (_e1402 * (_e1404 + (2u * _e1406)))), _e1412, _e1414);
                        let _e1422 = g_1;
                        let _e1424 = gp.head;
                        let _e1426 = gp.tail;
                        let _e1430 = pos_6;
                        let _e1435 = gp.n_rnc;
                        let _e1436 = below(gid_2.x, ((_e1422 * (_e1424 + (2u * _e1426))) + _e1430), STREAM_DC_VALUE, _e1435);
                        genome[_e1415] = _e1436;
                    }
                }
                continuing {
                    let _e1441 = pos_6;
                    pos_6 = (_e1441 + 1u);
                }
            }
        }
        continuing {
            let _e1445 = g_1;
            g_1 = (_e1445 + 1u);
        }
    }
    let _e1452 = chance(gid_2.x, OP_INVERT_DC, STREAM_OPERATOR, _e22.invert_dc);
    let _e1456 = gp.tail;
    if ((_e1456 >= 2u) && _e1452) {
        let _e1465 = gp.n_genes;
        let _e1466 = below(gid_2.x, 0u, STREAM_INVERT_DC, _e1465);
        let _e1474 = gp.tail;
        let _e1476 = below(gid_2.x, 1u, STREAM_INVERT_DC, (_e1474 - 1u));
        let _e1485 = gp.tail;
        let _e1489 = below(gid_2.x, 2u, STREAM_INVERT_DC, ((_e1485 - (2u + _e1476)) + 1u));
        let _e1495 = gp.n_genes;
        let _e1497 = gp.head;
        let _e1499 = gp.tail;
        let _e1506 = gp.head;
        let _e1508 = gp.tail;
        let _e1511 = at((gid_2.x * (_e1495 * (_e1497 + (2u * _e1499)))), _e1466, ((_e1506 + _e1508) + _e1489));
        let _e1517 = gp.n_genes;
        let _e1519 = gp.head;
        let _e1521 = gp.tail;
        let _e1530 = gp.head;
        let _e1532 = gp.tail;
        let _e1538 = at((gid_2.x * (_e1517 * (_e1519 + (2u * _e1521)))), _e1466, ((((_e1530 + _e1532) + _e1489) + (2u + _e1476)) - 1u));
        reverse(_e1511, _e1538);
    }
    let _e1544 = chance(gid_2.x, OP_TRANSPOSE_DC, STREAM_OPERATOR, _e22.transpose_dc);
    if _e1544 {
        let _e1551 = gp.n_genes;
        let _e1552 = below(gid_2.x, 0u, STREAM_TRANSPOSE_DC, _e1551);
        let _e1559 = gp.n_genes;
        let _e1560 = below(gid_2.x, 1u, STREAM_TRANSPOSE_DC, _e1559);
        let _e1567 = gp.tail;
        let _e1568 = below(gid_2.x, 2u, STREAM_TRANSPOSE_DC, _e1567);
        let _e1576 = gp.tail;
        let _e1580 = below(gid_2.x, 3u, STREAM_TRANSPOSE_DC, ((_e1576 - (1u + _e1568)) + 1u));
        let _e1588 = gp.tail;
        let _e1592 = below(gid_2.x, 4u, STREAM_TRANSPOSE_DC, ((_e1588 - (1u + _e1568)) + 1u));
        loop {
            let _e1595 = i_10;
            if (_e1595 < (1u + _e1568)) {
            } else {
                break;
            }
            {
                let _e1603 = gp.n_genes;
                let _e1605 = gp.head;
                let _e1607 = gp.tail;
                let _e1615 = gp.head;
                let _e1617 = gp.tail;
                let _e1620 = i_10;
                let _e1622 = at((gid_2.x * (_e1603 * (_e1605 + (2u * _e1607)))), _e1552, (((_e1615 + _e1617) + _e1580) + _e1620));
                let _e1626 = i_10;
                let _e1629 = genome[_e1622];
                segment[_e1626] = _e1629;
            }
            continuing {
                let _e1632 = i_10;
                i_10 = (_e1632 + 1u);
            }
        }
        loop {
            let _e1636 = i_11;
            let _e1638 = gp.tail;
            if (_e1636 < _e1638) {
            } else {
                break;
            }
            {
                let _e1645 = gp.n_genes;
                let _e1647 = gp.head;
                let _e1649 = gp.tail;
                let _e1657 = gp.head;
                let _e1659 = gp.tail;
                let _e1661 = i_11;
                let _e1663 = at((gid_2.x * (_e1645 * (_e1647 + (2u * _e1649)))), _e1560, ((_e1657 + _e1659) + _e1661));
                let _e1667 = i_11;
                let _e1670 = genome[_e1663];
                before[_e1667] = _e1670;
            }
            continuing {
                let _e1673 = i_11;
                i_11 = (_e1673 + 1u);
            }
        }
        let _e1679 = gp.head;
        let _e1681 = gp.tail;
        pos_7 = (((_e1679 + _e1681) + _e1592) + (1u + _e1568));
        loop {
            let _e1689 = pos_7;
            let _e1691 = gp.head;
            let _e1693 = gp.tail;
            if (_e1689 < (_e1691 + (2u * _e1693))) {
            } else {
                break;
            }
            {
                let _e1702 = gp.n_genes;
                let _e1704 = gp.head;
                let _e1706 = gp.tail;
                let _e1712 = pos_7;
                let _e1713 = at((gid_2.x * (_e1702 * (_e1704 + (2u * _e1706)))), _e1560, _e1712);
                let _e1720 = pos_7;
                let _e1724 = gp.head;
                let _e1726 = gp.tail;
                let _e1730 = before[((_e1720 - (1u + _e1568)) - (_e1724 + _e1726))];
                genome[_e1713] = _e1730;
            }
            continuing {
                let _e1733 = pos_7;
                pos_7 = (_e1733 + 1u);
            }
        }
        loop {
            let _e1737 = i_12;
            if (_e1737 < (1u + _e1568)) {
            } else {
                break;
            }
            {
                let _e1745 = gp.n_genes;
                let _e1747 = gp.head;
                let _e1749 = gp.tail;
                let _e1757 = gp.head;
                let _e1759 = gp.tail;
                let _e1762 = i_12;
                let _e1764 = at((gid_2.x * (_e1745 * (_e1747 + (2u * _e1749)))), _e1560, (((_e1757 + _e1759) + _e1592) + _e1762));
                let _e1769 = i_12;
                let _e1771 = segment[_e1769];
                genome[_e1764] = _e1771;
            }
            continuing {
                let _e1774 = i_12;
                i_12 = (_e1774 + 1u);
            }
        }
    }
    loop {
        let _e1778 = k_2;
        let _e1780 = gp.n_genes;
        let _e1782 = gp.n_rnc;
        if (_e1778 < (_e1780 * _e1782)) {
        } else {
            break;
        }
        {
            let _e1788 = k_2;
            let _e1791 = chance(gid_2.x, _e1788, STREAM_RNC_HIT, _e22.rnc_point);
            if _e1791 {
                let _e1795 = k_2;
                let _e1799 = gp.rnc_span;
                let _e1800 = below(gid_2.x, _e1795, STREAM_RNC_VALUE, _e1799);
                let _e1807 = gp.n_genes;
                let _e1809 = gp.n_rnc;
                let _e1812 = k_2;
                let _e1816 = gp.rnc_lo;
                rnc[((gid_2.x * (_e1807 * _e1809)) + _e1812)] = f32((_e1816 + i32(_e1800)));
            }
        }
        continuing {
            let _e1822 = k_2;
            k_2 = (_e1822 + 1u);
        }
    }
    let _e1829 = chance(gid_2.x, OP_CLEANSE, STREAM_OPERATOR, _e22.cleanse);
    if _e1829 {
        let _e1836 = gp.n_genes;
        let _e1837 = below(gid_2.x, 0u, STREAM_CLEANSE, _e1836);
        let _e1845 = gp.n_genes;
        let _e1847 = gp.head;
        let _e1849 = gp.tail;
        let _e1855 = gp.head;
        let _e1857 = gp.tail;
        cleanse_gene(gid_2.x, ((gid_2.x * (_e1845 * (_e1847 + (2u * _e1849)))) + (_e1837 * (_e1855 + (2u * _e1857)))), _e22.cleanse_collapse);
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

    let _e6 = gp.pop;
    if (gid_3.x >= _e6) {
        return;
    }
    let _e10 = island_of(gid_3.x);
    if ((gid_3.x <= (_e10.lo + _e10.elites)) || (((gid_3.x - (_e10.lo + _e10.elites)) & 1u) == 0u)) {
        return;
    }
    let _e32 = chance(gid_3.x, OP_CX_1P, STREAM_OPERATOR, _e10.cx_one_point);
    if ((((_e10.arrivals > 0u) && (_e10.arrival_children > 0u)) && (gid_3.x < ((_e10.lo + _e10.elites) + (_e10.arrivals * (_e10.arrival_children + 1u))))) || _e32) {
        let _e59 = gp.n_genes;
        let _e60 = below(gid_3.x, 0u, STREAM_CX_1P, _e59);
        let _e68 = gp.head;
        let _e70 = gp.tail;
        let _e73 = below(gid_3.x, 1u, STREAM_CX_1P, (_e68 + (2u * _e70)));
        loop {
            let _e75 = whole;
            if (_e75 < _e60) {
            } else {
                break;
            }
            {
                let _e84 = whole;
                let _e89 = gp.head;
                let _e91 = gp.tail;
                swap_tokens((gid_3.x - 1u), gid_3.x, _e84, 0u, (_e89 + (2u * _e91)));
                let _e99 = whole;
                let _e103 = whole;
                swap_consts((gid_3.x - 1u), _e99, gid_3.x, _e103);
            }
            continuing {
                let _e106 = whole;
                whole = (_e106 + 1u);
            }
        }
        swap_tokens((gid_3.x - 1u), gid_3.x, _e60, 0u, (_e73 + 1u));
    }
    let _e122 = chance(gid_3.x, OP_CX_2P, STREAM_OPERATOR, _e10.cx_two_point);
    if _e122 {
        let _e129 = gp.n_genes;
        let _e130 = below(gid_3.x, 0u, STREAM_CX_2P, _e129);
        let _e137 = gp.n_genes;
        let _e138 = below(gid_3.x, 1u, STREAM_CX_2P, _e137);
        let _e146 = gp.head;
        let _e148 = gp.tail;
        let _e151 = below(gid_3.x, 2u, STREAM_CX_2P, (_e146 + (2u * _e148)));
        let _e159 = gp.head;
        let _e161 = gp.tail;
        let _e164 = below(gid_3.x, 3u, STREAM_CX_2P, (_e159 + (2u * _e161)));
        if (min(_e130, _e138) == max(_e130, _e138)) {
            swap_tokens((gid_3.x - 1u), gid_3.x, min(_e130, _e138), min(_e151, _e164), (max(_e151, _e164) + 1u));
        } else {
            whole_1 = (min(_e130, _e138) + 1u);
            loop {
                let _e184 = whole_1;
                if (_e184 < max(_e130, _e138)) {
                } else {
                    break;
                }
                {
                    let _e194 = whole_1;
                    let _e199 = gp.head;
                    let _e201 = gp.tail;
                    swap_tokens((gid_3.x - 1u), gid_3.x, _e194, 0u, (_e199 + (2u * _e201)));
                    let _e209 = whole_1;
                    let _e213 = whole_1;
                    swap_consts((gid_3.x - 1u), _e209, gid_3.x, _e213);
                }
                continuing {
                    let _e216 = whole_1;
                    whole_1 = (_e216 + 1u);
                }
            }
            let _e228 = gp.head;
            let _e230 = gp.tail;
            swap_tokens((gid_3.x - 1u), gid_3.x, min(_e130, _e138), _e151, (_e228 + (2u * _e230)));
            swap_tokens((gid_3.x - 1u), gid_3.x, max(_e130, _e138), 0u, (_e164 + 1u));
        }
    }
    let _e248 = chance(gid_3.x, OP_CX_GENE, STREAM_OPERATOR, _e10.cx_gene);
    if _e248 {
        let _e255 = gp.n_genes;
        let _e256 = below(gid_3.x, 0u, STREAM_CX_GENE, _e255);
        let _e263 = gp.n_genes;
        let _e264 = below(gid_3.x, 1u, STREAM_CX_GENE, _e263);
        loop {
            let _e268 = pos_8;
            let _e270 = gp.head;
            let _e272 = gp.tail;
            if (_e268 < (_e270 + (2u * _e272))) {
            } else {
                break;
            }
            {
                let _e285 = gp.n_genes;
                let _e287 = gp.head;
                let _e289 = gp.tail;
                let _e295 = gp.head;
                let _e297 = gp.tail;
                let _e302 = pos_8;
                let _e307 = gp.n_genes;
                let _e309 = gp.head;
                let _e311 = gp.tail;
                let _e317 = gp.head;
                let _e319 = gp.tail;
                let _e324 = pos_8;
                let _e327 = genome[(((gid_3.x * (_e307 * (_e309 + (2u * _e311)))) + (_e264 * (_e317 + (2u * _e319)))) + _e324)];
                genome[((((gid_3.x - 1u) * (_e285 * (_e287 + (2u * _e289)))) + (_e256 * (_e295 + (2u * _e297)))) + _e302)] = _e327;
                let _e336 = gp.n_genes;
                let _e338 = gp.head;
                let _e340 = gp.tail;
                let _e346 = gp.head;
                let _e348 = gp.tail;
                let _e353 = pos_8;
                let _e359 = gp.n_genes;
                let _e361 = gp.head;
                let _e363 = gp.tail;
                let _e369 = gp.head;
                let _e371 = gp.tail;
                let _e376 = pos_8;
                let _e379 = genome[((((gid_3.x - 1u) * (_e359 * (_e361 + (2u * _e363)))) + (_e256 * (_e369 + (2u * _e371)))) + _e376)];
                genome[(((gid_3.x * (_e336 * (_e338 + (2u * _e340)))) + (_e264 * (_e346 + (2u * _e348)))) + _e353)] = _e379;
            }
            continuing {
                let _e382 = pos_8;
                pos_8 = (_e382 + 1u);
            }
        }
        swap_consts((gid_3.x - 1u), _e256, gid_3.x, _e264);
        return;
    } else {
        return;
    }
}
