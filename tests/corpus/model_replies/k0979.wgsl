@compute @workgroup_size(256, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    // The kernel text must begin with the nine binding declarations; they follow below this entry point's attributes in WGSL module order.
    _unused();
}
 
fn _unused() {}
