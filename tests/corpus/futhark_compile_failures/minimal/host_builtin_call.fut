-- host_builtin_call, minimal: f32.isnan in host code is a Call of isnan32, which rust-wgpu looked up among the program's own functions (patch 0014).
entry main (x: f32) : bool = f32.isnan x
