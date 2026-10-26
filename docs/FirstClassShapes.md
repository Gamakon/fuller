# Design Question: First-Class Shape Types in Multi-Typed GEP

**Project:** nucleotable / Fuller / `futhark_true`  
**Status:** Architectural proposal for technical review  
**Objective:** Determine whether symbolic array shapes can become first-class elements of the Karva type system while preserving true multi-typed GEP's legality-by-construction guarantee.

## 1. The problem

We have built a Futhark-inspired evolutionary programming kingdom using nucleotable's true multi-typed Gene Expression Programming (GEP).

The current kingdom is deliberately compact:

- 4 types
- 38 symbols
- Native Karva genetic representation
- Typed AST expression
- Futhark compilation
- Fuller e-graph optimisation and write-back into Karva

The major achievement of our architecture is that **every gene expresses a legal AST by construction**. We do not repair, filter, reject or substitute malformed genetic expressions.

However, the current Futhark kingdom makes a questionable design assumption:

> Shape is a value, not a type.

This keeps the type system small, but sacrifices one of Futhark's most powerful features: the ability to express and enforce relationships between array dimensions.

Consider matrix multiplication:

```futhark
def matmul [n][m][p]
  (A: [n][m]f32)
  (B: [m][p]f32) : [n][p]f32
```

The type signature establishes that the inner dimensions must match, and that the result has shape `[n][p]`.

These are not incidental properties of the computation. They are fundamental constraints on which programs are meaningful.

Our existing four-type kingdom cannot express these relationships.

As the language expands to multidimensional arrays, matrix operations, broadcasting, gathering, reshaping and other parallel algorithms, this becomes a serious limitation.

## 2. Why the obvious solution is undesirable

We could introduce a separate concrete type for every array shape.

For example:

```text
Vector_f32_128
Vector_f32_256
Matrix_f32_128_256
Matrix_f32_256_512
...
```

But this creates a combinatorial explosion in the number of types, reading frames and potentially symbols.

We have already encountered the consequences of a very large genetic search space in `wgsl_true`, which has thousands of symbols and hundreds of types.

Our new Futhark kingdom was created specifically to avoid that complexity.

We therefore want:

**The expressive power of dependent shape typing without the combinatorial expansion of the genetic alphabet.**

## 3. Proposed solution: symbolic shape types

Introduce parametric type expressions into the kingdom.

Examples:

```text
Scalar(f32)
Vector(f32, n)
Matrix(f32, n, m)
Tensor(f32, [n, m, p])
```

Here `n`, `m` and `p` are symbolic dimensions.

Operators would declare relationships between these dimensions.

For example:

```text
transpose:
  Matrix(T,n,m)
    -> Matrix(T,m,n)

matmul:
  Matrix(T,n,m) × Matrix(T,m,p)
    -> Matrix(T,n,p)

map:
  Vector(T,n)
    -> Vector(U,n)

zipWith:
  Vector(T,n) × Vector(U,n)
    -> Vector(V,n)

reduce:
  Vector(T,n)
    -> Scalar(T)
```

The symbol table contains one parametric signature per operation, not a separate symbol for every possible shape.

Dimensions are unified during construction.

An operator can only be selected when its output type is compatible with the type required at the current position in the expressed program.

Its child requirements are then determined by the instantiated signature.

For example, if an expression requires:

```text
Matrix(f32, n, p)
```

and selects `matmul`, its children must have compatible shapes:

```text
Matrix(f32, n, m)
Matrix(f32, m, p)
```

Both occurrences of `m` must refer to the same dimension.

## 4. The difficult part: preserving GEP legality by construction

Our current true multi-typed GEP uses one reading frame per output type.

Every frame has a typed head and tail, and the symbol-table relations determine exactly which symbols can appear in each position.

This makes illegal ASTs unrepresentable.

Symbolic shapes introduce a new challenge.

Two expressions may share the same basic type, such as `Matrix(f32)`, while having different dimensional constraints.

We cannot simply ignore those constraints and ask Futhark to catch invalid shapes after expression.

That would violate the central design principle of nucleotable.

We therefore propose investigating **constraint-aware genetic reading frames**.

Conceptually:

1. The required output type includes symbolic shape constraints.
2. A symbol is eligible only if its output signature unifies with those constraints.
3. The selected symbol establishes further constraints on its children.
4. These constraints remain consistent throughout genetic expression.
5. Mutation, crossover and COMPRESS must preserve the same guarantees.

The desired result is that every genetically expressible program remains well typed, including its symbolic dimensions.

However, it is not yet established that this mechanism can be incorporated into Ferreira-style fixed Karva reading frames without compromising their existing guarantees.

That is the principal research question.

## 5. Alternative architectures worth considering

The proposed constraint-aware reading-frame approach is not necessarily the best answer.

We would like the team to assess at least three options.

**A. Parametric reading frames**

Extend typed GEP so that reading frames support parameterised type families and symbolic unification during expression.

This offers generality but may require fundamental changes to gene layout and genetic operators.

**B. Finite shape families**

Instantiate a deliberately small set of shape-compatible type families from a common parametric specification.

This may preserve the existing GEP machinery but risks alphabet growth.

**C. Shape-safe primitives**

Keep the existing type structure and provide high-level operators whose shapes are correct by construction, such as shape-preserving maps, matched-shape zip operations and constrained matrix multiplication.

This is simpler, but may limit which combinations of independently shaped arrays can be expressed.

We should also consider hybrids.

For example, parametric shape relationships might be resolved at the kingdom interface or gene-layout construction stage, producing a finite, shape-safe typed alphabet for each evolutionary run.

This may preserve the current engine while avoiding a global explosion of specialised symbols.

## 6. Invariants we must preserve

Any acceptable design must maintain the following properties:

- Every gene expresses a legal, shape-correct AST without repair or rejection.
- Mutation and crossover preserve legality by construction.
- Karva remains the canonical genetic representation.
- GEN → AST → GENE round-tripping remains valid.
- Fuller's e-graph rewrites preserve type and shape relationships.
- COMPRESS and `write_back` preserve those relationships when extracting equivalent programs.
- The number of reading frames and symbol rows remains manageable.
- Runtime dimensions may remain symbolic, but their required relationships must be guaranteed or made explicit as input-interface contracts.

We must distinguish shape compatibility guaranteed by the program from shape requirements placed on external inputs.

For example, a polymorphic matrix multiplication can be a valid program even though its runtime dimensions are not known until execution.

The important property is that the type signature guarantees compatibility whenever its declared input contract is satisfied.

## 7. Proposed proof of concept

Before implementing full dependent typing, construct a minimal experimental kingdom supporting:

```text
Scalar(f32)
Vector(f32,n)
Matrix(f32,n,m)
```

With only four operations:

```text
map
zipWith
transpose
matmul
```

Test whether the genetic architecture can express these operations with symbolic dimensions while maintaining shape correctness.

The experiment should include:

1. Random gene construction.
2. Mutation and crossover.
3. Expression to typed AST.
4. AST re-encoding into Karva.
5. Fuller e-graph import, rewriting and extraction.
6. Compilation through Futhark.
7. Measurement of genome width, alphabet size and construction overhead.

Tests must include independently shaped inputs, shared symbolic dimensions and incompatible shape relationships that should be impossible to construct.

A compiler rejection caused by an incompatible internal shape must be treated as a failure of the proposed genetic design, not a normal fitness outcome.

## 8. The question for the engineering team

**Can we incorporate symbolic, dependent shape types into nucleotable's true multi-typed GEP while preserving legality by construction, fixed genetic representation and efficient mutation/crossover?**

Specifically:

- Can the current per-type Karva reading frames support symbolic unification, or would this require a different framing mechanism?
- Can we avoid generating concrete types for every possible dimension combination?
- Can shape compatibility be encoded in the genetic layout rather than checked and repaired after expression?
- Will homologous crossover continue to work when compatible basic types carry different symbolic dimension bindings?
- Can Fuller's egglog sorts and rewrite rules accommodate parametric shape types without reproducing the same combinatorial explosion?
- Would a small, interface-specialised kingdom be a better solution than modifying nucleotable's core?
- What is the simplest implementation that genuinely preserves all existing invariants?

## 9. Requested response

Please inspect the actual implementations of `geneframe`, `typed_gep`, `KingdomRelations`, the Futhark kingdom generator, `write_back`, and `futhark_egraph`.

**Do not implement anything yet.**

We want an architectural assessment, not speculative code.

Please explain:

1. Whether the proposed approach is feasible.
2. Which existing invariants or implementation assumptions it would affect.
3. Which of the three architectural options is most promising.
4. Whether the current fixed reading-frame design can be retained.
5. What the minimum proof of concept should be.

If the proposal cannot preserve legality by construction, say so explicitly and recommend an alternative.

**The central question is not whether we can type-check an evolved program. It is whether we can design its genetic representation so that a dimensionally invalid program cannot be expressed in the first place.**
