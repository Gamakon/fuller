// The K-expression linter on the device. A transcription of src/lint/flat.rs:
// one invocation per expression, greedy rounds, no recursion, fixed scratch,
// integer comparisons only.
//
// Layout. Expression i owns SLOT nodes at [i*SLOT, i*SLOT + len). A node is
// (op, arg0, arg1, konst). For a Var, arg0 is the data column. For a Num, arg1
// is the LITERAL ID the host gave that f64 value: the device never compares
// floats, and two literals are the same subtree iff their ids are equal.
// `aux` is, per node, (literal class id) | (predicate bits << 8), computed by
// the host in f64.

struct Cfg {
    n_expr: u32,
    n_guards: u32,
    // Weakest exactness admitted: 0 bit, 1 rounding, 2 finite.
    admit: u32,
    rounds: u32,
    // Invocations per dispatch row, for the 2-D dispatch.
    stride: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

const HEAP: u32 = 15u;        // pattern / template heap slots
const RULE_STRIDE: u32 = 126u; // 6 header words + 2 * 15 slots * 4 words
const GUARD_STRIDE: u32 = 8u;

const KIND_UNUSED: u32 = 0u;
const KIND_OP: u32 = 1u;
const KIND_MV: u32 = 2u;
const KIND_NUM: u32 = 3u;

const N_OPS: u32 = 28u;

@group(0) @binding(0) var<storage, read>       nodes_in:  array<Node>;
@group(0) @binding(1) var<storage, read>       aux_in:    array<u32>;
@group(0) @binding(2) var<storage, read>       len_in:    array<u32>;
// Rules sorted by (root opcode, rule id); bucket[op] .. bucket[op+1] is the
// range whose pattern root is `op`.
@group(0) @binding(3) var<storage, read>       rules:     array<u32>;
@group(0) @binding(4) var<storage, read>       bucket:    array<u32>;
@group(0) @binding(5) var<storage, read>       guards:    array<u32>;
// Literal pool a template may write: (konst bits, literal id, aux) per entry.
@group(0) @binding(6) var<storage, read>       pool:      array<u32>;
// Caller facts per data column.
@group(0) @binding(7) var<storage, read>       var_facts: array<u32>;
@group(0) @binding(8) var<storage, read_write> nodes_out: array<Node>;
// Per expression: (length, rewrites applied, weakest exactness used, 0).
@group(0) @binding(9) var<storage, read_write> info_out:  array<u32>;
@group(0) @binding(10) var<uniform>            cfg:       Cfg;

var<private> strict:  array<u32, 80>;
var<private> loose:   array<u32, 80>;
var<private> qb:      array<u32, 80>;
var<private> at:      array<u32, 15>;
var<private> home:    array<u32, 15>;
var<private> bind:    array<u32, 8>;

// K1. Backward scan: children sit at higher indices, so their facts are ready.
fn facts_pass() {
    var k: u32 = n_cur;
    loop {
        if (k == 0u) { break; }
        k = k - 1u;
        let nd = cur[k];
        var s: u32 = 0u;
        var l: u32 = 0u;
        if (nd.op == OP_VAR) {
            s = var_facts[nd.arg0];
            l = s;
        }
        // Implications chain through at most three fact bits.
        for (var it: u32 = 0u; it < 3u; it = it + 1u) {
            for (var g: u32 = 0u; g < cfg.n_guards; g = g + 1u) {
                let b = g * GUARD_STRIDE;
                let g_op = guards[b];
                let req0 = guards[b + 1u];
                let req1 = guards[b + 2u];
                let self_req = guards[b + 3u];
                let gives = guards[b + 4u];
                let range_only = guards[b + 5u];
                let pred_req = guards[b + 6u];
                var shape_s: bool = true;
                var shape_l: bool = true;
                if (g_op != NONE) {
                    if (nd.op != g_op) {
                        shape_s = false;
                        shape_l = false;
                    } else {
                        let a = arity(nd.op);
                        if (a >= 1u) {
                            shape_s = shape_s && ((strict[nd.arg0] & req0) == req0);
                            shape_l = shape_l && ((loose[nd.arg0] & req0) == req0);
                        }
                        if (a >= 2u) {
                            shape_s = shape_s && ((strict[nd.arg1] & req1) == req1);
                            shape_l = shape_l && ((loose[nd.arg1] & req1) == req1);
                        }
                        if (nd.op == OP_NUM) {
                            let ok = ((caux[k] >> 8u) & pred_req) == pred_req;
                            shape_s = shape_s && ok;
                            shape_l = shape_l && ok;
                        }
                    }
                }
                if (shape_l && ((l & self_req) == self_req)) { l = l | gives; }
                if (range_only == 0u && shape_s && ((s & self_req) == self_req)) { s = s | gives; }
            }
        }
        strict[k] = s;
        loose[k] = l;
    }
}

// Same-subtree test: walk both in lockstep through a queue of index pairs.
fn same_subtree(a: u32, b: u32) -> bool {
    var head: u32 = 0u;
    var tail: u32 = 1u;
    qa[0] = a;
    qb[0] = b;
    loop {
        if (head >= tail) { break; }
        let x = qa[head];
        let y = qb[head];
        head = head + 1u;
        if (x == y) { continue; }
        let nx = cur[x];
        let ny = cur[y];
        if (nx.op != ny.op) { return false; }
        let ar = arity(nx.op);
        if (ar == 0u) {
            if (nx.op == OP_NUM) {
                if (nx.arg1 != ny.arg1) { return false; }   // literal ids
            } else if (nx.arg0 != ny.arg0) {
                return false;                               // data columns
            }
        } else {
            if (tail + ar > WORK) { return false; }          // cannot verify: no match
            qa[tail] = nx.arg0;
            qb[tail] = ny.arg0;
            tail = tail + 1u;
            if (ar == 2u) {
                qa[tail] = nx.arg1;
                qb[tail] = ny.arg1;
                tail = tail + 1u;
            }
        }
    }
    return true;
}

// K2 at one position. Returns 0 no match, 1 match on strict facts, 2 match
// that needed a range-only fact.
fn match_at(r: u32, pos: u32, use_loose: bool) -> u32 {
    for (var i: u32 = 0u; i < HEAP; i = i + 1u) { at[i] = NONE; }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { bind[i] = NONE; }
    var needed_loose: bool = false;
    at[0] = pos;
    let pat = r * RULE_STRIDE + 6u;
    for (var slot: u32 = 0u; slot < HEAP; slot = slot + 1u) {
        let kind = rules[pat + slot * 4u];
        if (kind == KIND_UNUSED || at[slot] == NONE) { continue; }
        let id = rules[pat + slot * 4u + 1u];
        let req = rules[pat + slot * 4u + 2u];
        let lit = rules[pat + slot * 4u + 3u];
        let idx = at[slot];
        let nd = cur[idx];
        if (kind == KIND_OP) {
            if (nd.op != id) { return 0u; }
            let ar = arity(nd.op);
            if (ar >= 1u) { at[2u * slot + 1u] = nd.arg0; }
            if (ar >= 2u) { at[2u * slot + 2u] = nd.arg1; }
        } else if (kind == KIND_MV) {
            if (bind[id] == NONE) {
                bind[id] = idx;
                if ((strict[idx] & req) != req) {
                    if (use_loose && ((loose[idx] & req) == req)) {
                        needed_loose = true;
                    } else {
                        return 0u;
                    }
                }
            } else if (!same_subtree(bind[id], idx)) {
                return 0u;
            }
        } else {
            if (nd.op != OP_NUM) { return 0u; }
            if (lit != 0u) {
                if ((caux[idx] & 0xffu) != lit) { return 0u; }
            } else if (((caux[idx] >> 8u) & req) != req) {
                return 0u;
            }
        }
    }
    if (needed_loose) { return 2u; }
    return 1u;
}

// K3: append the template, then `splice` (splice.wgsl) redirects one pointer
// and renumbers breadth-first from the root. Orphans are never reached.
fn apply(r: u32, pos: u32) {
    let tm = r * RULE_STRIDE + 6u + HEAP * 4u;
    var n: u32 = n_cur;
    for (var slot: u32 = 0u; slot < HEAP; slot = slot + 1u) {
        let kind = rules[tm + slot * 4u];
        home[slot] = NONE;
        if (kind == KIND_OP || kind == KIND_NUM) {
            home[slot] = n;
            n = n + 1u;
        } else if (kind == KIND_MV) {
            home[slot] = bind[rules[tm + slot * 4u + 1u]];
        }
    }
    for (var slot: u32 = 0u; slot < HEAP; slot = slot + 1u) {
        let kind = rules[tm + slot * 4u];
        let id = rules[tm + slot * 4u + 1u];
        if (kind == KIND_OP) {
            let ar = arity(id);
            var a0: u32 = 0u;
            var a1: u32 = 0u;
            if (ar >= 1u) { a0 = home[2u * slot + 1u]; }
            if (ar >= 2u) { a1 = home[2u * slot + 2u]; }
            cur[home[slot]] = Node(id, a0, a1, 0.0);
            caux[home[slot]] = 0u;
        } else if (kind == KIND_NUM) {
            cur[home[slot]] = Node(OP_NUM, 0u, pool[id * 3u + 1u], bitcast<f32>(pool[id * 3u]));
            caux[home[slot]] = pool[id * 3u + 2u];
        }
    }
    splice(pos, home[0]);
}

@compute @workgroup_size(64, 1, 1)
fn lint_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.y * cfg.stride + gid.x;
    if (e >= cfg.n_expr) { return; }
    let base = e * SLOT;
    n_cur = len_in[e];
    if (n_cur == 0u || n_cur > SLOT) {
        info_out[e * 4u] = 0u;          // refused: the host keeps its own form
        info_out[e * 4u + 1u] = 0u;
        info_out[e * 4u + 2u] = 0u;
        info_out[e * 4u + 3u] = 1u;
        return;
    }
    for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
        cur[i] = nodes_in[base + i];
        caux[i] = aux_in[base + i];
    }

    var steps: u32 = 0u;
    var level: u32 = 0u;
    let use_loose = cfg.admit == 2u;
    for (var round: u32 = 0u; round < cfg.rounds; round = round + 1u) {
        facts_pass();
        var hit_rule: u32 = NONE;
        var hit_pos: u32 = 0u;
        var hit_kind: u32 = 0u;
        // First hit in (class A before B, position, rule id) order.
        for (var order: u32 = 0u; order < 2u && hit_rule == NONE; order = order + 1u) {
            for (var pos: u32 = 0u; pos < n_cur && hit_rule == NONE; pos = pos + 1u) {
                let op = cur[pos].op;
                for (var r: u32 = bucket[op]; r < bucket[op + 1u]; r = r + 1u) {
                    let h = r * RULE_STRIDE;
                    if (rules[h + 4u] != order || rules[h + 5u] > cfg.admit) { continue; }
                    let m = match_at(r, pos, use_loose);
                    if (m != 0u) {
                        hit_rule = r;
                        hit_pos = pos;
                        hit_kind = m;
                        break;
                    }
                }
            }
        }
        if (hit_rule == NONE) { break; }
        // match_at left the bindings of the hit in `bind`.
        var used: u32 = rules[hit_rule * RULE_STRIDE + 5u];
        if (hit_kind == 2u) { used = 2u; }
        level = max(level, used);
        apply(hit_rule, hit_pos);
        steps = steps + 1u;
    }

    for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
        nodes_out[base + i] = cur[i];
    }
    info_out[e * 4u] = n_cur;
    info_out[e * 4u + 1u] = steps;
    info_out[e * 4u + 2u] = level;
    info_out[e * 4u + 3u] = 0u;
}
