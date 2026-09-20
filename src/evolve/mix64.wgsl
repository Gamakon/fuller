// mix64 — the project's counter-addressed generator, u64 emulated on u32 pairs.
//
// COPIED VERBATIM from minkymorgan/qdrant
//   lib/udv23-umap/src/ultradim/umap/kernels_optimize.rs :: WGSL_MIX64_LIB
// (SplitMix64 finaliser, Stafford mix13; the recipe pinned by rp-formula's
// known-answer vectors, which evolve/mod.rs asserts against too). Do not edit
// the functions below: a change here is a change of every random draw.
//
// A 64-bit value is vec2<u32>(lo, hi). No native u64 in WGSL.

fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
    // 16-bit limbs: 4 partial products with explicit carries.
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + p10;                        // wraps mod 2^32
    let mid_carry = select(0u, 1u, mid < p01);  // carry out of the add
    let mid_lo = mid << 16u;
    let mid_hi = (mid >> 16u) | (mid_carry << 16u);
    let lo = p00 + mid_lo;                      // wraps
    let lo_carry = select(0u, 1u, lo < p00);
    let hi = p11 + mid_hi + lo_carry;
    return vec2<u32>(lo, hi);
}

fn mul64_lo(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // Low 64 bits of the 64x64 product: cross terms contribute their low
    // words to the high limb only (their high words wrap past bit 64).
    let ll = mul32x32_64(a.x, b.x);
    return vec2<u32>(ll.x, ll.y + a.x * b.y + a.y * b.x);
}

fn shr64(v: vec2<u32>, s: u32) -> vec2<u32> {
    // Valid for 0 < s < 32 (the mixer shifts by 30 / 27 / 31 only).
    return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}

fn mix64p(x_in: vec2<u32>) -> vec2<u32> {
    // SplitMix64 finaliser (Stafford mix13) — constants from umap/rng.rs.
    var x = x_in;
    x = x ^ shr64(x, 30u);
    x = mul64_lo(x, vec2<u32>(0x1CE4E5B9u, 0xBF58476Du)); // 0xBF58476D1CE4E5B9
    x = x ^ shr64(x, 27u);
    x = mul64_lo(x, vec2<u32>(0x133111EBu, 0x94D049BBu)); // 0x94D049BB133111EB
    x = x ^ shr64(x, 31u);
    return x;
}

fn mix64_3p(domain: vec2<u32>, a: vec2<u32>, b: vec2<u32>, c: vec2<u32>) -> vec2<u32> {
    return mix64p(mix64p(mix64p(domain ^ a) ^ b) ^ c);
}

fn mix64_modp(domain: vec2<u32>, a: vec2<u32>, b: vec2<u32>, c: vec2<u32>, n: u32) -> u32 {
    // Lemire multiply-shift: bits [64, 96) of the 96-bit product h * n.
    let h = mix64_3p(domain, a, b, c);
    let p = mul32x32_64(h.x, n);
    let q = mul32x32_64(h.y, n);
    let mid = p.y + q.x;
    let carry = select(0u, 1u, mid < p.y);
    return q.y + carry;
}
// ---- end mix64 library ----
