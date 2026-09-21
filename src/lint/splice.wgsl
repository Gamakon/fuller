// What every splice on the device shares: the node, the per-invocation work
// area, and the relayout. `kernel.wgsl` (a rule's template) and
// `snap_graft.wgsl` (a constant's form) are each compiled with this file in
// front of them. `flat.rs::splice` is the CPU twin.

struct Node {
    op: u32,
    arg0: u32,
    arg1: u32,
    konst: f32,
};

const SLOT: u32 = 64u;        // nodes per expression, in and out
const WORK: u32 = 80u;        // SLOT + the most nodes one template can add (15)
const NONE: u32 = 0xffffffffu;

const OP_VAR: u32 = 0u;
const OP_NUM: u32 = 1u;

fn arity(op: u32) -> u32 {
    switch (op) {
        case 0u, 1u: { return 0u; }
        case 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 16u, 17u, 18u, 20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u: { return 1u; }
        default: { return 2u; }
    }
}

var<private> cur:     array<Node, 80>;
var<private> caux:    array<u32, 80>;
var<private> nxt:     array<Node, 80>;
var<private> naux:    array<u32, 80>;
var<private> qa:      array<u32, 80>;
var<private> n_cur:   u32;

// The subtree rooted at `new_root` has been appended after the `n_cur` live
// nodes: point whatever pointed at `pos` at it, then renumber breadth-first
// from the root. `caux` travels with its node.
fn splice(pos: u32, new_root: u32) {
    var root: u32 = 0u;
    if (pos == 0u) {
        root = new_root;
    } else {
        for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
            let ar = arity(cur[i].op);
            if (ar >= 1u && cur[i].arg0 == pos) { cur[i].arg0 = new_root; }
            if (ar >= 2u && cur[i].arg1 == pos) { cur[i].arg1 = new_root; }
        }
    }
    // qa[i] = old index of the node that lands at new index i.
    var head: u32 = 0u;
    var tail: u32 = 1u;
    qa[0] = root;
    loop {
        if (head >= tail) { break; }
        let old = cur[qa[head]];
        let oaux = caux[qa[head]];
        let ar = arity(old.op);
        let first = tail;
        if (ar >= 1u) { qa[tail] = old.arg0; tail = tail + 1u; }
        if (ar >= 2u) { qa[tail] = old.arg1; tail = tail + 1u; }
        var a0: u32 = old.arg0;
        var a1: u32 = old.arg1;
        if (ar >= 1u) { a0 = first; a1 = 0u; }
        if (ar >= 2u) { a1 = first + 1u; }
        nxt[head] = Node(old.op, a0, a1, old.konst);
        naux[head] = oaux;
        head = head + 1u;
    }
    n_cur = tail;
    for (var i: u32 = 0u; i < n_cur; i = i + 1u) {
        cur[i] = nxt[i];
        caux[i] = naux[i];
    }
}

