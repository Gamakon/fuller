@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<u32>;
var<workgroup> sh_a: array<u32, 256>;

fn nblocks() -> u32 { return (params.x + 255u) / 256u; }
// scratch layout: [0, nb*k) per-(block,cluster) counts (block-major),
// then k totals, then k starts, then n sorted point indices.
fn scratch_ok() -> bool {
  let nb = nblocks();
  return arrayLength(&scratch) >= nb * params.y + 2u * params.y + params.x;
}

@compute @workgroup_size(256, 1, 1)
fn hist(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let k = params.y;
  let nb = nblocks();
  let g = gid.x;
  if (g >= nb * k || !scratch_ok()) { return; }
  let b = g / k;
  let c = g % k;
  let base = b * 256u;
  let end = min(base + 256u, params.x);
  var cnt: u32 = 0u;
  for (var q: u32 = 0u; q < 256u; q = q + 1u) {
    let idx = base + q;
    if (idx >= end) { break; }
    if (assignments[idx] == c) { cnt = cnt + 1u; }
  }
  scratch[g] = cnt;
}

@compute @workgroup_size(256, 1, 1)
fn offsets(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let k = params.y;
  let c = gid.x;
  if (c >= k || !scratch_ok()) { return; }
  let nb = nblocks();
  var run: u32 = 0u;
  for (var b: u32 = 0u; b < 4096u; b = b + 1u) {
    if (b >= nb) { break; }
    let v = scratch[b * k + c];
    scratch[b * k + c] = run;
    run = run + v;
  }
  scratch[nb * k + c] = run;
}

@compute @workgroup_size(256, 1, 1)
fn starts(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  if (lidx != 0u || !scratch_ok()) { return; }
  let k = params.y;
  let nb = nblocks();
  let cb = nb * k;
  var run: u32 = 0u;
  for (var c: u32 = 0u; c < k; c = c + 1u) {
    let v = scratch[cb + c];
    scratch[cb + k + c] = run;
    run = run + v;
  }
}

@compute @workgroup_size(64, 2, 1)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let n = params.x;
  let k = params.y;
  let w = wid.x;
  let b = w / 2u;
  let half = w & 1u;
  let a0 = b * 256u + lidx;
  let a1 = a0 + 128u;
  sh_a[lidx] = select(0xffffffffu, assignments[min(a0, arrayLength(&assignments) - 1u)], a0 < n);
  sh_a[lidx + 128u] = select(0xffffffffu, assignments[min(a1, arrayLength(&assignments) - 1u)], a1 < n);
  workgroupBarrier();
  let i = w * 128u + lidx;
  if (i >= n || !scratch_ok()) { return; }
  let p = half * 128u + lidx;
  let c = sh_a[p];
  if (c >= k) { return; }
  var rank: u32 = 0u;
  for (var q: u32 = 0u; q < 256u; q = q + 1u) {
    if (q >= p) { break; }
    if (sh_a[q] == c) { rank = rank + 1u; }
  }
  let nb = nblocks();
  let cb = nb * k;
  let pos = scratch[cb + k + c] + scratch[b * k + c] + rank;
  scratch[cb + 2u * k + pos] = i;
}

@compute @workgroup_size(256, 1, 1)
fn mean(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let d = params.z;
  let k = params.y;
  let col = gid.x;
  if (col >= k * d) { return; }
  let c = col / d;
  let j = col % d;
  var s: f32 = 0.0;
  var comp: f32 = 0.0;
  var cnt: u32 = 0u;
  if (scratch_ok()) {
    let nb = nblocks();
    let cb = nb * k;
    cnt = scratch[cb + c];
    let st = scratch[cb + k + c];
    let sb = cb + 2u * k + st;
    for (var m: u32 = 0u; m < params.x; m = m + 1u) {
      if (m >= cnt) { break; }
      let i = scratch[sb + m];
      let y = vectors[min(i * d + j, arrayLength(&vectors) - 1u)] - comp;
      let t = s + y;
      comp = (t - s) - y;
      s = t;
    }
  } else {
    let nx = min(params.x, 16777216u);
    for (var i: u32 = 0u; i < nx; i = i + 1u) {
      if (assignments[min(i, arrayLength(&assignments) - 1u)] == c) {
        let y = vectors[min(i * d + j, arrayLength(&vectors) - 1u)] - comp;
        let t = s + y;
        comp = (t - s) - y;
        s = t;
        cnt = cnt + 1u;
      }
    }
  }
  if (cnt > 0u) {
    centroids[col] = s / f32(cnt);
  }
}
