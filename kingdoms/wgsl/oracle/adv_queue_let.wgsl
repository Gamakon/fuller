// The step-5 fault: a let bound from a queue, the head advanced, the let
// used after the advance. Inlining the let at its use re-reads the queue.
@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var head = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let pos = q[head];
        head = head + 1u;
        out[i] = pos + q[head];
    }
}
