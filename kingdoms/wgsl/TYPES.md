# WGSL kingdom — the type table

Revision 2, 2026-10-08. A row's typed arity names a **slot type** and a
**content type** for every input and its output. Slot types are what the
hardware stores. Content types are what the value means and which operators
are legal on it. naga admits no implicit conversion, so every row's arity is
exact; pack and unpack rows are the only crossings between content types,
`as` and `bitcast` the only crossings between slot types.

## 1. Slot types

| group | slot types | count |
|---|---|---|
| scalars | `f32`, `i32`, `u32`, `bool` (`f16` with the device feature) | 4 (5) |
| vectors | `vec2<T>`, `vec3<T>`, `vec4<T>` for each scalar T | 12 |
| matrices | `matCxR<f32>` for C, R in {2, 3, 4} | 9 |
| arrays | `array<T>` of any scalar or vector, indexed by `u32` | one per buffer |
| root | `store` — the statement type a head gene's root produces | 1 |

Rules the slot column implies:

- A load is a typed terminal of its value's type. The pointer and its address
  space (`function`, `private`, `workgroup`, `uniform`, `storage`) never
  appear in a gene; they are the terminal's identity in the symbol table.
- `as_f32`, `as_i32`, `as_u32` and `bitcast` are the only rows that change
  the slot type.
- `swizzle` and `access` move from vector to scalar; `compose` builds a
  vector from scalars and a matrix from vectors.
- Comparisons produce `bool` or `vecN<bool>`; `select` consumes them; `any`
  and `all` collapse a vector of them.
- A head gene's root is `store`; the stored value's slot and content are
  fixed by the target buffer, which keeps every decoded gene a legal
  statement.

Out of the first table: atomics, images, samplers, ray queries, binding
arrays.

## 2. Content types

| content | carried in slot | meaning | legal operators | crossings |
|---|---|---|---|---|
| `real` | `f32`, `f16`, vectors, matrices | a plain number | arithmetic, transcendentals, comparisons, `dot`, `length`, `mix`, `fma` | `quantise_k`, `to_fixed_n` |
| `fixed Qm.n` | `i32`, `vecN<i32>`; two `i32` limbs for wide accumulation | fixed point, scale 2⁻ⁿ (faxl's `Q15.16` is the first instance) | saturating add and subtract, multiply with rescale, shifts, comparisons; never raw `*` | `to_real_n`, `to_fixed_n` |
| `index` | `u32`, `vecN<u32>` | a position: into an array, a row, a gene, a node | add, subtract, multiply, divide, modulo, min, max, comparisons; never float operators | `index_of` from `i32` floored at zero; `clamp_index` |
| `count` | `u32` | a cardinality, a length | add, subtract to zero, min, max, comparisons | `to_index` |
| `code` | `u32`; `u8` lanes packed in `u32` | a codebook entry (noctiluca's 4-, 8-, 16- and 1,024-level Gaussian codebooks; snap tables) | lookup; equality only | `quantise_k` from `real`, `dequantise_k` to `real` |
| `bits` | `u32`, `vecN<u32>` | a bit field, a mask, a packed sign, a hash state at rest | and, or, xor, not, shifts, `countOneBits`, `extractBits`, `insertBits`; never arithmetic | `pack` from lanes, `unpack` to lanes |
| `hash` | `u32`; `vec2<u32>` for 64 bits | a counter-based hash state (`mix64`, `fmix32`) | mix, combine, split; equality only | `to_bits`; `to_index` by masked rejection, never modulo; `to_real` by scale |
| `sign` | `bool`; packed in `bits` | ±1 carried as one bit (noctiluca's sign-packed wiring) | negate-if, `select` | `to_real` as ±1.0 |
| `flag` | `bool`, `vecN<bool>` | a predicate | and, or, not, `any`, `all`, `select` | from any comparison |
| `opaque` | any | a value carried through unchanged | none | none |

## 3. Row families the content column creates

| family | rows | from → to |
|---|---|---|
| quantise | `quantise_k` for each codebook k ∈ {4, 8, 16, 1024} | `real` → `code` |
| dequantise | `dequantise_k` | `code` → `real` |
| fixed point | `to_fixed_n`, `to_real_n`, `fixed_mul_n`, `fixed_add_sat`, `fixed_sub_sat`, `fixed_shl`, `fixed_shr` | `real` ↔ `fixed`; `fixed` × `fixed` → `fixed` |
| packing | `pack4_u8`, `unpack4_u8`, `pack_sign`, `unpack_sign`, `extract_bits`, `insert_bits` | lanes ↔ `bits` |
| hashing | `mix64`, `fmix32`, `combine`, `split_lo`, `split_hi` | `hash` → `hash`; `hash` → `bits` |
| indexing | `index_of`, `to_index`, `clamp_index`, `band_of` | `i32`/`count` → `index` |
| conversion | `as_f32`, `as_i32`, `as_u32`, `bitcast` | slot → slot, content preserved or renamed by the row |

## 4. Fallback leaves (what Design C substitutes for a mismatched codon)

| type | fallback |
|---|---|
| `real` | `0.0` of the slot's width and lanes |
| `fixed` | `0` |
| `index`, `count` | `0u` |
| `code` | the codebook's first entry |
| `bits`, `hash` | `0u` |
| `sign` | `+1` |
| `flag` | `false` |
| `store` | no-op (the root is dropped and the codon counted as masked) |

## 5. What the two columns buy

- A gene cannot multiply two `fixed` values with a raw `*`, add a `hash` to a
  `real`, or index with a `float`: the row does not exist, the projection
  substitutes the fallback, the codon is counted as masked. Same mechanism as
  the transcendental depth ladder, applied to meaning rather than depth.
- One `u32` slot carries `index`, `count`, `code`, `bits` and `hash` with
  disjoint operator sets, which is where most hand-written kernel bugs live.
- Reading an existing kernel in must infer the content column from use;
  where it cannot, the value is `opaque` and the gene round-trips it
  untouched. Inference improves as rows are added; `opaque` is never wrong,
  only unevolvable.

## 6. Open

1. Whether `f16` is in the first table or added when a device needs it.
2. Whether non-square matrices earn rows before a kernel uses them.
3. Whether `fixed` widths beyond `Q15.16` are rows or a parameter of one row.
4. How `opaque` content is reported in the masked-codon telemetry so its
   share of a read-in kernel is visible.
