// Workgroup memory with barriers across 64 invocations: a load after a
// barrier is a new version. Device only (the interpreter runs one
// invocation); every lane of out is 126 and of out2 127.
var<workgroup> w: array<u32, 64>;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read_write> out2: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    w[lid.x] = lid.x * 2u;
    workgroupBarrier();
    out[lid.x] = w[63u - lid.x] + w[lid.x];
    workgroupBarrier();
    w[lid.x] = out[lid.x] + 1u;
    workgroupBarrier();
    out2[lid.x] = w[(lid.x + 1u) % 64u];
}
