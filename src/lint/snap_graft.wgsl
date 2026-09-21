// Snap, stage 2: the graft. Compiled with splice.wgsl in front of it; the CPU
// twin is `snap_graft.rs::snap_graft`.
//
// One invocation per (expression, variant). Expression e owns SLOT nodes in
// `nodes_in` in the linter's encoded form — a Num's arg1 is its LITERAL ID — and
// SLOT (entry, sign) pairs in `hits`, written by `snap_main` earlier in the
// same command buffer (NONE for a node that is not a literal, or matched
// nothing). Variant v of expression e owns SLOT nodes of `nodes_out` at
// (e * VARIANTS + v) * SLOT, in the evaluator's node layout.
//
// An ATOM is a distinct literal id with a hit; atoms are numbered by the first
// node that holds them. Variants 0 .. VARIANTS-2 graft one atom each, at every
// site that holds it; the last grafts every atom, and exists only when there
// are two or more. A variant is whole or not at all: if its grafts do not all
// fit in SLOT nodes it comes back unchanged, NODE_OVERSIZE.
//
// A grafted Num has arg1 = NONE: its konst IS its value (template literals are
// exact in f32). A named constant is a Num whose konst is the constant's value
// and whose arg0 is 1 + its index in the table's names — no reader of a Num
// looks at arg0, so the evaluator runs the variant as it is and the name rides
// along for the write-back. Nothing here is a final decision: the host confirms
// every graft in f64 before it is reported or written into a gene.

struct GraftCfg {
    n_expr: u32,
    // Invocations per dispatch row, for the 2-D dispatch.
    stride: u32,
    pad0: u32,
    pad1: u32,
};

const VARIANTS: u32 = 4u;
const INFO_STRIDE: u32 = 4u;
const OP_NEG: u32 = 6u;

const NO_HIT: u32 = 0u;
const GRAFTED: u32 = 1u;
const NODE_OVERSIZE: u32 = 2u;
const REFUSED: u32 = 3u;

@group(0) @binding(0) var<storage, read>       nodes_in:   array<Node>;
@group(0) @binding(1) var<storage, read>       len_in:     array<u32>;
// Per node slot: (entry id or NONE, sign bit).
@group(0) @binding(2) var<storage, read>       hits:       array<u32>;
// Per entry: (template offset in nodes, template node count, flags, 0).
@group(0) @binding(3) var<storage, read>       info:       array<u32>;
// 4 words per template node: (op, arg0, arg1, f32 literal bits).
@group(0) @binding(4) var<storage, read>       templates:  array<u32>;
// f32 bits of each named constant's value.
@group(0) @binding(5) var<storage, read>       name_vals:  array<u32>;
@group(0) @binding(6) var<storage, read_write> nodes_out:  array<Node>;
// Per variant: (length, status, sites grafted, atoms in the expression).
@group(0) @binding(7) var<storage, read_write> vinfo_out:  array<u32>;
@group(0) @binding(8) var<uniform>             gcfg:       GraftCfg;

// Replace the literal at `pos` with the entry's form, under a Neg if `negative`.
fn graft(pos: u32, entry: u32, negative: u32) {
    let off = info[entry * INFO_STRIDE];
    let cnt = info[entry * INFO_STRIDE + 1u];
    var first: u32 = n_cur;
    if (negative != 0u) {
        cur[n_cur] = Node(OP_NEG, n_cur + 1u, 0u, 0.0);
        caux[n_cur] = 0u;
        first = n_cur + 1u;
    }
    for (var k: u32 = 0u; k < cnt; k = k + 1u) {
        let w = (off + k) * 4u;
        let op = templates[w];
        if (op == OP_VAR) {
            let name = templates[w + 1u];
            cur[first + k] = Node(OP_NUM, name + 1u, NONE, bitcast<f32>(name_vals[name]));
        } else if (op == OP_NUM) {
            cur[first + k] = Node(OP_NUM, 0u, NONE, bitcast<f32>(templates[w + 3u]));
        } else {
            let ar = arity(op);
            var a0: u32 = 0u;
            var a1: u32 = 0u;
            if (ar >= 1u) { a0 = first + templates[w + 1u]; }
            if (ar >= 2u) { a1 = first + templates[w + 2u]; }
            cur[first + k] = Node(op, a0, a1, 0.0);
        }
        caux[first + k] = 0u;
    }
    splice(pos, n_cur);
}

@compute @workgroup_size(64, 1, 1)
fn graft_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y * gcfg.stride + gid.x;
    if (t >= gcfg.n_expr * VARIANTS) { return; }
    let e = t / VARIANTS;
    let v = t % VARIANTS;
    let base = e * SLOT;
    let dst = t * SLOT;
    n_cur = len_in[e];
    if (n_cur == 0u || n_cur > SLOT) {
        vinfo_out[t * 4u] = 0u;         // refused: the host keeps its own form
        vinfo_out[t * 4u + 1u] = REFUSED;
        vinfo_out[t * 4u + 2u] = 0u;
        vinfo_out[t * 4u + 3u] = 0u;
        return;
    }

    // caux, for a literal with a hit: atom number (1-based) | sign << 7 |
    // entry << 8. It travels with the node through every relayout.
    var atoms: u32 = 0u;
    for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
        cur[i] = nodes_in[base + i];
        caux[i] = 0u;
        let entry = hits[(base + i) * 2u];
        if (cur[i].op != OP_NUM || entry == NONE) { continue; }
        var atom: u32 = 0u;
        for (var j: u32 = 0u; j < i; j = j + 1u) {
            if (caux[j] != 0u && cur[j].arg1 == cur[i].arg1) {
                atom = caux[j] & 0x7fu;
                break;
            }
        }
        if (atom == 0u) {
            atoms = atoms + 1u;
            atom = atoms;
        }
        caux[i] = atom | (hits[(base + i) * 2u + 1u] << 7u) | (entry << 8u);
    }

    // The atom this variant grafts: 0 = every atom, NONE = nothing to do.
    var want: u32 = NONE;
    if (v + 1u < VARIANTS) {
        if (v < atoms) { want = v + 1u; }
    } else if (atoms >= 2u) {
        want = 0u;
    }

    var sites: u32 = 0u;
    var grown: u32 = n_cur;
    for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
        let a = caux[i];
        if (want == NONE || a == 0u || (want != 0u && (a & 0x7fu) != want)) { continue; }
        sites = sites + 1u;
        grown = grown + info[(a >> 8u) * INFO_STRIDE + 1u] + ((a >> 7u) & 1u) - 1u;
    }

    var status: u32 = NO_HIT;
    if (sites != 0u && grown > SLOT) {
        status = NODE_OVERSIZE;
        sites = 0u;
    } else if (sites != 0u) {
        status = GRAFTED;
        // Lowest site first. A grafted node carries caux 0, so each pass
        // consumes one site.
        for (var done: u32 = 0u; done < sites; done = done + 1u) {
            for (var pos: u32 = 0u; pos < n_cur; pos = pos + 1u) {
                let a = caux[pos];
                if (a != 0u && (want == 0u || (a & 0x7fu) == want)) {
                    graft(pos, a >> 8u, (a >> 7u) & 1u);
                    break;
                }
            }
        }
    }

    for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
        nodes_out[dst + i] = cur[i];
    }
    vinfo_out[t * 4u] = n_cur;
    vinfo_out[t * 4u + 1u] = status;
    vinfo_out[t * 4u + 2u] = sites;
    vinfo_out[t * 4u + 3u] = atoms;
}
