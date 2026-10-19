# cuneus kernels (third-party, MIT)

The compute kernels of cuneus 0.6.4 (github.com/altunenes/cuneus, commit
51925d3, written for naga 30) that bind no texture or sampler, copied
unchanged as fixtures for fuller's reader (2026-10-19, the naga/wgpu 30
upgrade, docs/PLAN_naga_wgpu_30.md section 3). License: LICENSE (MIT, Enes
Altun).

- instances.wgsl: src/mesh/instances.wgsl
- meshlab_swarm.wgsl: examples/shaders/meshlab_swarm.wgsl
- skin.wgsl: src/mesh/skin.wgsl
- radix_sort.wgsl: src/radix_sort/shader.wgsl with the constant prelude and
  the three `{…}` workgroup sizes cuneus fills in src/radix_sort/mod.rs
  (histogram 256, prefix 128, scatter 256; subgroup size 1, as its first
  builder sets it; rs_mem_dwords 256 + 15 x 256 = 4096; sweep offsets 0, 256,
  512).

fallback.wgsl and vertex.wgsl have no compute entry point; every other
cuneus kernel binds images or samplers, which the kingdom has no rows for.
