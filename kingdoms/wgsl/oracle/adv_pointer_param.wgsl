// Aliasing through a ptr<function> parameter and a pointer let: a store
// through the parameter bumps the caller's local at that call site only; a
// store through the let bumps its base.
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
fn bump(p: ptr<function, f32>, by: f32) { *p = *p + by; }
@compute @workgroup_size(1)
fn main() {
    var x = 1.0;
    var y = 10.0;
    let a = x + y;
    bump(&x, 2.0);
    out[0] = x + y;
    out[1] = a;
    bump(&y, x);
    out[2] = x + y;
    let p = &out[3];
    out[3] = 5.0;
    *p = *p * 2.0;
    out[4] = out[3];
}
