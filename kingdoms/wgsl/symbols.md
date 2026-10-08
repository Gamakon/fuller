# WGSL — the symbol table

Specification, 2026-10-08. The rows the geneframe loader will build for the
`WGSL` kingdom. Each row is `semantic_id  in {slot·content: n, …}  out
{slot·content: 1}`; the semantic id is naga's own operator or function name so
the reader and the writer have nothing to translate. Types are `TYPES.md`'s:
a slot (`f32`, `u32`, `vec3<f32>`, …) and a content (`real`, `index`, …).

## Instantiation rule

A row template is written once over a **lane shape** `L ∈ {1, 2, 3, 4}` and a
**scalar** `S`; the loader expands it to one row per legal `(S, L)`. `vec1`
means the scalar itself. Lanes never mix inside one row except where naga
allows a scalar beside a vector (`*` by scalar, `select` on a scalar
condition, `mix` with scalar weight); those are listed as their own rows. A
row exists only where the content column permits the operator (`TYPES.md`
§2).

## Arity

naga's `Select`, `Math::Fma`, `Clamp`, `Mix`, `SmoothStep` and `ExtractBits`
take three operands; `InsertBits` takes four. The decoders' structural limit
is total arity 2 today (`regex` composed its three-argument interval from two
rows). This kingdom needs the limit at 3, and `InsertBits` composed from two
rows (`insert_hi`, `insert_lo`). Raising the limit is a decoder change in
phylu and the one engine change this kingdom asks for.

## 1. Terminals — `in {}`, one row each

| row | out | identity |
|---|---|---|
| `literal` | any slot, content `real`/`index`/`count`/`code`/`bits`/`flag` as declared | an RNC-style constant drawn from the kingdom's range per content |
| `load <buffer>[i]` | the buffer's element slot · declared content | a storage or uniform buffer element at index `i` (an `index` terminal is its own gene input, see `access`) |
| `load <uniform>.field` | the field's slot · declared content | a uniform struct field |
| `load <local>` | the local's slot · content | a function-local or private variable at this point in the scaffold |
| `arg <name>` | the argument's slot · content | a function argument |
| `global_invocation_id` | `vec3<u32>` · `index` | builtin |
| `local_invocation_id` | `vec3<u32>` · `index` | builtin |
| `workgroup_id` | `vec3<u32>` · `index` | builtin |
| `local_invocation_index` | `u32` · `index` | builtin |
| `num_workgroups` | `vec3<u32>` · `count` | builtin |

A read-in kernel's terminal set is exactly its loads, literals, arguments and
builtins; nothing else is drawable until a row says so.

## 2. Arithmetic — `real` (naga `Binary`, `Unary`, `Math`)

```
Add Subtract Multiply Divide     in {S·real: 2}  out {S·real}         S ∈ f32, f16; L ∈ 1..4
Multiply (scalar × vector)       in {f32·real: 1, vecL<f32>·real: 1} out {vecL<f32>·real}
Multiply (matrix × vector)       in {matCxR·real: 1, vecC·real: 1}  out {vecR·real}
Multiply (matrix × matrix)       in {matAxB·real: 1, matBxC·real: 1} out {matAxC·real}
Negate                           in {S·real: 1}  out {S·real}
Abs Sign Floor Ceil Round Fract Trunc Saturate
                                 in {S·real: 1}  out {S·real}
Min Max Step                     in {S·real: 2}  out {S·real}
Clamp Mix Fma SmoothStep         in {S·real: 3}  out {S·real}         (arity 3)
Mix (scalar weight)              in {vecL·real: 2, f32·real: 1} out {vecL·real}
Sqrt InverseSqrt Exp Exp2 Log Log2
                                 in {S·real: 1}  out {S·real}
Sin Cos Tan Sinh Cosh Tanh Asin Acos Atan Asinh Acosh Atanh Radians Degrees
                                 in {S·real: 1}  out {S·real}
Atan2 Pow Ldexp                  in {S·real: 2}  out {S·real}
Dot                              in {vecL·real: 2}  out {f32·real}
Cross                            in {vec3·real: 2}  out {vec3·real}
Length                           in {vecL·real: 1}  out {f32·real}
Distance                         in {vecL·real: 2}  out {f32·real}
Normalize                        in {vecL·real: 1}  out {vecL·real}
Outer                            in {vecR·real: 1, vecC·real: 1} out {matCxR·real}
Transpose                        in {matCxR·real: 1} out {matRxC·real}
Determinant Inverse              in {matNxN·real: 1} out {f32·real} / {matNxN·real}
FaceForward Reflect              in {vecL·real: 2..3} out {vecL·real}
Refract                          in {vecL·real: 2, f32·real: 1} out {vecL·real}
```

`Modf`, `Frexp` return structs and are out of the first table.

## 3. Integer arithmetic — `index`, `count`

```
Add Subtract Multiply Divide Modulo   in {u32·index: 2}  out {u32·index}     L ∈ 1..4
Min Max Clamp                         in {u32·index: 2 or 3} out {u32·index}
Add Subtract(to zero) Min Max         in {u32·count: 2}  out {u32·count}
index_of                              in {i32·real-as-int: 1} out {u32·index}   floor at 0 (see TYPES §3)
to_index                              in {u32·count: 1}  out {u32·index}
clamp_index                           in {u32·index: 1, u32·count: 1} out {u32·index}   i < n
band_of                               in {u32·index: 1, u32·count: 1} out {u32·index}   i / band
```

No row adds an `index` to a `real`, a `count` to a `code`, or divides an
`index` by a `float`. Signed `i32` arithmetic rows exist only for `real-as-int`
content (a signed integer used as a number, e.g. a gene's RNC integer), with
the same shape as the `real` block minus the transcendentals.

## 4. Fixed point — `fixed Qm.n`

```
fixed_add_sat fixed_sub_sat      in {i32·fixed: 2}  out {i32·fixed}
fixed_mul_n                      in {i32·fixed: 2}  out {i32·fixed}      (x*y) >> n with rounding
fixed_shl fixed_shr              in {i32·fixed: 1, u32·count: 1} out {i32·fixed}
Min Max Less LessEqual …         as integers, content preserved
to_fixed_n                       in {f32·real: 1}   out {i32·fixed}
to_real_n                        in {i32·fixed: 1}  out {f32·real}
wide_mul_n                       in {i32·fixed: 2}  out {vec2<i32>·fixed-wide}  two limbs, no rescale
wide_add_sat                     in {vec2<i32>·fixed-wide: 2} out {vec2<i32>·fixed-wide}
narrow_n                         in {vec2<i32>·fixed-wide: 1} out {i32·fixed}
```

`n` is a row parameter; the first instance is `Q15.16` (faxl's ESN). Raw
`Multiply` on `i32·fixed` has no row.

## 5. Codes and quantisation — `code`

```
quantise_k                       in {f32·real: 1}   out {u32·code}      k ∈ {4, 8, 16, 1024}: nearest codebook level
dequantise_k                     in {u32·code: 1}   out {f32·real}      the level's value
Equal NotEqual                   in {u32·code: 2}   out {bool·flag}
pack4_u8                         in {u32·code: 4}   out {u32·bits}      (arity 4: composed as pack_lo, pack_hi)
unpack4_u8                       in {u32·bits: 1, u32·index: 1} out {u32·code}   lane i
```

Codebooks are rows' constants, loaded with the kingdom (noctiluca's Gaussian
tables and fuller's snap table are the first).

## 6. Bits — `bits` (naga `Binary` bitwise, `Math` bit functions)

```
And InclusiveOr ExclusiveOr      in {u32·bits: 2}  out {u32·bits}       L ∈ 1..4
BitwiseNot                       in {u32·bits: 1}  out {u32·bits}
ShiftLeft ShiftRight             in {u32·bits: 1, u32·count: 1} out {u32·bits}
CountOneBits CountLeadingZeros CountTrailingZeros FindLsb FindMsb
                                 in {u32·bits: 1}  out {u32·count}
ReverseBits                      in {u32·bits: 1}  out {u32·bits}
ExtractBits                      in {u32·bits: 1, u32·index: 1, u32·count: 1} out {u32·bits}   (arity 3)
insert_lo insert_hi              composing naga InsertBits (arity 4) from two arity-3 rows
pack_sign unpack_sign            in {bool·sign: 1} out {u32·bits} / inverse
Pack4x8snorm Pack4x8unorm Pack2x16snorm Pack2x16unorm Pack2x16float
                                 in {vecL<f32>·real: 1} out {u32·bits}
Unpack4x8snorm … Unpack2x16float in {u32·bits: 1}  out {vecL<f32>·real}
```

No arithmetic row takes `bits`.

## 7. Hashing — `hash`

```
mix64                            in {vec2<u32>·hash: 1} out {vec2<u32>·hash}    SplitMix64 finaliser (two limbs)
fmix32                           in {u32·hash: 1}   out {u32·hash}            MurmurHash3 finaliser
combine                          in {u32·hash: 1, u32·bits: 1} out {u32·hash}  xor-in a key or an index
split_lo split_hi                in {vec2<u32>·hash: 1} out {u32·hash}
to_bits                          in {u32·hash: 1}   out {u32·bits}
to_index_rejecting               in {u32·hash: 1, u32·count: 1} out {u32·index}   masked rejection, never modulo
to_real_unit                     in {u32·hash: 1}   out {f32·real}            scale to [0, 1)
```

These are the primitives noctiluca, faxl and phylu's `mix64.wgsl` already
share; the rows name them once.

## 8. Comparison, logic, selection — `flag`, `sign`

```
Equal NotEqual Less LessEqual Greater GreaterEqual
                                 in {S·c: 2}  out {bool·flag}          any content c that permits comparison; L lanes → vecL<bool>
LogicalAnd LogicalOr             in {bool·flag: 2}  out {bool·flag}
LogicalNot                       in {bool·flag: 1}  out {bool·flag}
All Any                          in {vecL<bool>·flag: 1} out {bool·flag}
IsNan IsInf                      in {S·real: 1}  out {bool·flag}
Select                           in {S·c: 2, bool·flag: 1} out {S·c}        (arity 3) both branches the same slot·content
Select (lane-wise)               in {vecL·c: 2, vecL<bool>·flag: 1} out {vecL·c}
negate_if                        in {S·real: 1, bool·sign: 1} out {S·real}
sign_to_real                     in {bool·sign: 1}  out {f32·real}         ±1.0
```

## 9. Shape — vectors and matrices (naga `Swizzle`, `Access`, `Compose`, `Splat`)

```
access_i                         in {vecL·c: 1}  out {S·c}               i < L, one row per i
access (dynamic)                 in {array<T>·c: 1, u32·index: 1} out {T·c}
swizzle_xy swizzle_xyz …         in {vecL·c: 1}  out {vecK·c}            the swizzles a read-in kernel used, added as met
compose_2 compose_3 compose_4    in {S·c: L}     out {vecL·c}            (arity up to 4: composed from compose_2 rows)
compose_mat                      in {vecR·real: C} out {matCxR·real}
splat_L                          in {S·c: 1}     out {vecL·c}
```

## 10. Conversion (naga `As`)

```
as_f32                           in {i32·real-as-int: 1} / {u32·count|index: 1} out {f32·real}
as_i32                           in {f32·real: 1} out {i32·real-as-int}       truncating
as_u32                           in {f32·real: 1} out {u32·count}             truncating, floor at 0
bitcast_f32_u32 bitcast_u32_f32  in {f32·real: 1} out {u32·bits} / inverse
```

Content is renamed only by the row that says so; `bitcast` always lands in
`bits`.

## 11. Roots — `store`

```
store <buffer>[i]                in {u32·index: 1, T·c: 1} out {store}       T·c fixed by the buffer
store <local>                    in {T·c: 1}  out {store}
```

A head gene's root is a `store`; the scaffold supplies which store. A branch
condition root is `in {bool·flag: 1} out {store}` by the same mechanism.

## What is deliberately absent

Atomics, barriers, images, samplers, derivatives, ray queries, subgroup
operations, pointers as values, implicit conversions, raw `*` on `fixed`,
`%` on `hash`, arithmetic on `bits`, any row of total arity above 3.

## Loader notes

- Kingdom name `WGSL`; `geneframe::WGSL`.
- Rows are generated from the templates above by the instantiation rule;
  the loader's test asserts every generated row's semantic id is a naga
  variant or one of the named composite rows, and that no row crosses
  content except those in `TYPES.md` §3.
- The terminal set is per kernel (its buffers, uniforms, arguments), loaded
  from the read-in scaffold, not from this file.
