@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<u32>;

var<workgroup> hcnt: array<atomic<u32>, 512>;
var<workgroup> sh4: array<vec4<u32>, 64>;
var<workgroup> sh_scan: array<u32, 256>;

fn kah(s: f32, c: f32, x: f32) -> vec2<f32> {
  let y = x - c;
  let t = s + y;
  return vec2<f32>(t, (t - s) - y);
}

@compute @workgroup_size(256, 1, 1)
fn hist(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
  let n = params.x;
  let k = params.y;
  let nb = (n + 255u) / 256u;
  let total = nb * k;
  let base = wid.x * 256u;
  let s0 = base + lidx;
  let la = arrayLength(&assignments) - 1u;
  let ls = arrayLength(&scratch) - 1u;
  if (k < 8u) {
    if (s0 < total) {
      let blk = s0 / k;
      let c = s0 % k;
      var cnt = 0u;
      var j = 0u;
      loop {
        if (j >= 256u) { break; }
        let e = blk * 256u + j;
        if (e < n) {
          if (assignments[min(e, la)] == c) { cnt = cnt + 1u; }
        }
        j = j + 1u;
      }
      scratch[min(s0, ls)] = cnt;
    }
    return;
  }
  atomicStore(&hcnt[lidx], 0u);
  atomicStore(&hcnt[256u + lidx], 0u);
  workgroupBarrier();
  let lastS = min(base + 255u, total - 1u);
  let b0 = base / k;
  let b1 = lastS / k;
  let nit = min(b1 - b0 + 1u, 4096u);
  var i = 0u;
  loop {
    if (i >= nit) { break; }
    let b = b0 + i;
    let p = (i & 1u) * 256u;
    let lo = max(base, b * k);
    let hi = min(lastS, b * k + k - 1u);
    let clo = lo - b * k;
    let span = hi - lo;
    let e = b * 256u + lidx;
    if (e < n) {
      let a = assignments[min(e, la)];
      if (a >= clo) {
        let sl = a - clo;
        if (sl <= span) { atomicAdd(&hcnt[p + sl], 1u); }
      }
    }
    workgroupBarrier();
    if (s0 >= lo && s0 <= hi) {
      let idx = p + (s0 - lo);
      let v = atomicLoad(&hcnt[idx]);
      atomicStore(&hcnt[idx], 0u);
      scratch[min(s0, ls)] = v;
    }
    i = i + 1u;
  }
}

@compute @workgroup_size(256, 1, 1)
fn offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let k = params.y;
  if (c >= k) { return; }
  let nb = (params.x + 255u) / 256u;
  let st3 = nb * k;
  let ls = arrayLength(&scratch) - 1u;
  let nx = min(nb, 16777216u);
  var run = 0u;
  var kx = 0u;
  loop {
    if (kx + 8u > nx) { break; }
    let r = kx * k + c;
    let a0 = scratch[min(r, ls)];
    let a1 = scratch[min(r + k, ls)];
    let a2 = scratch[min(r + 2u * k, ls)];
    let a3 = scratch[min(r + 3u * k, ls)];
    let a4 = scratch[min(r + 4u * k, ls)];
    let a5 = scratch[min(r + 5u * k, ls)];
    let a6 = scratch[min(r + 6u * k, ls)];
    let a7 = scratch[min(r + 7u * k, ls)];
    let w = st3 + r;
    scratch[min(w, ls)] = run; run = run + a0;
    scratch[min(w + k, ls)] = run; run = run + a1;
    scratch[min(w + 2u * k, ls)] = run; run = run + a2;
    scratch[min(w + 3u * k, ls)] = run; run = run + a3;
    scratch[min(w + 4u * k, ls)] = run; run = run + a4;
    scratch[min(w + 5u * k, ls)] = run; run = run + a5;
    scratch[min(w + 6u * k, ls)] = run; run = run + a6;
    scratch[min(w + 7u * k, ls)] = run; run = run + a7;
    kx = kx + 8u;
  }
  loop {
    if (kx >= nx) { break; }
    let r = kx * k + c;
    let a0 = scratch[min(r, ls)];
    scratch[min(st3 + r, ls)] = run;
    run = run + a0;
    kx = kx + 1u;
  }
  scratch[min(2u * nb * k + c, ls)] = run;
}

@compute @workgroup_size(256, 1, 1)
fn starts(@builtin(local_invocation_index) lidx: u32) {
  let k = params.y;
  let nb = (params.x + 255u) / 256u;
  let st1 = 2u * nb * k;
  let st2 = st1 + k;
  let ls = arrayLength(&scratch) - 1u;
  let cs = min((k + 255u) / 256u, 4096u);
  let e0 = lidx * cs;
  let e1 = min(k, e0 + cs);
  var sum = 0u;
  var e = e0;
  loop {
    if (e >= e1) { break; }
    sum = sum + scratch[min(st1 + e, ls)];
    e = e + 1u;
  }
  sh_scan[lidx] = sum;
  workgroupBarrier();
  var s = 0u;
  loop {
    if (s >= 8u) { break; }
    let off = 1u << s;
    var v = sh_scan[lidx];
    if (lidx >= off) { v = v + sh_scan[lidx - off]; }
    workgroupBarrier();
    sh_scan[lidx] = v;
    workgroupBarrier();
    s = s + 1u;
  }
  var run = sh_scan[lidx] - sum;
  e = e0;
  loop {
    if (e >= e1) { break; }
    let t = scratch[min(st1 + e, ls)];
    scratch[min(st2 + e, ls)] = run;
    run = run + t;
    e = e + 1u;
  }
}

@compute @workgroup_size(256, 1, 1)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
  let s0 = gid.x;
  let n = params.x;
  let k = params.y;
  let nb = (n + 255u) / 256u;
  let la = arrayLength(&assignments) - 1u;
  let ls = arrayLength(&scratch) - 1u;
  let blk = s0 / 256u;
  var a = 0xffffffffu;
  if (s0 < n) { a = assignments[min(s0, la)]; }
  sh4[lidx >> 2u][lidx & 3u] = a;
  workgroupBarrier();
  if (s0 < n) {
    var r = 0u;
    var q = 0u;
    loop {
      if (q >= 64u || q * 4u >= lidx) { break; }
      let v = sh4[q];
      let j = q * 4u;
      r = r + u32(j < lidx && v.x == a) + u32(j + 1u < lidx && v.y == a)
            + u32(j + 2u < lidx && v.z == a) + u32(j + 3u < lidx && v.w == a);
      q = q + 1u;
    }
    let st7 = 2u * nb * k;
    let pos = (scratch[min(st7 + k + a, ls)] + scratch[min(nb * k + blk * k + a, ls)]) + r;
    scratch[min(st7 + 2u * k + pos, ls)] = s0;
  }
}

@compute @workgroup_size(256, 1, 1)
fn mean(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s0 = gid.x;
  let k = params.y;
  let d = params.z;
  if (s0 >= k * d) { return; }
  let nb = (params.x + 255u) / 256u;
  let st3 = 2u * nb * k;
  let st5 = st3 + 2u * k;
  let c = s0 / d;
  let dim = s0 % d;
  let ls = arrayLength(&scratch) - 1u;
  let lv = arrayLength(&vectors) - 1u;
  let cnt = scratch[min(st3 + c, ls)];
  let st = scratch[min(st3 + k + c, ls)];
  let base = st5 + st;
  let nx = min(cnt, min(params.x, 16777216u));
  var sc = vec2<f32>(0.0, 0.0);
  var kx = 0u;
  loop {
    if (kx + 4u > nx) { break; }
    let i0 = scratch[min(base + kx, ls)];
    let i1 = scratch[min(base + kx + 1u, ls)];
    let i2 = scratch[min(base + kx + 2u, ls)];
    let i3 = scratch[min(base + kx + 3u, ls)];
    let x0 = vectors[min(i0 * d + dim, lv)];
    let x1 = vectors[min(i1 * d + dim, lv)];
    let x2 = vectors[min(i2 * d + dim, lv)];
    let x3 = vectors[min(i3 * d + dim, lv)];
    sc = kah(sc.x, sc.y, x0);
    sc = kah(sc.x, sc.y, x1);
    sc = kah(sc.x, sc.y, x2);
    sc = kah(sc.x, sc.y, x3);
    kx = kx + 4u;
  }
  loop {
    if (kx >= nx) { break; }
    let i0 = scratch[min(base + kx, ls)];
    let x0 = vectors[min(i0 * d + dim, lv)];
    sc = kah(sc.x, sc.y, x0);
    kx = kx + 1u;
  }
  if (cnt > 0u) {
    centroids[s0] = sc.x / f32(cnt);
  }
}
