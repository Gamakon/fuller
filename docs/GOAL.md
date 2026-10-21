# THE GOAL

**A true multi-typed Gene Expression Programming kingdom for WGSL: every gene expresses a legal GPU program, because the symbol table's types describe exactly the legal programs.**

## The design

1. **The chromosome.** A fixed number of genes joined by a linking function. Each gene is one Karva gene: a head of length `h` and a tail of length `t = h(n-1) + 1`, `n` the largest arity (Ferreira).

2. **The symbol table is the kingdom.** Each row is a symbol with an output type and ordered input types. The types are columns of the gene: every position of the head and tail holds one symbol per type, so a gene carries an open reading frame for each type.

3. **Expression.** Karva reading from the root: each argument slot asks for a type, and the next symbol of that type's reading frame fills it. Every expressed tree is a program the symbol table can spell.

4. **The types carry every rule of a legal program.** For WGSL that means:
   - an **index** type that is in range for the array it indexes;
   - a **cell** type a thread owns, which is the only place it stores;
   - a **uniform** context type, the only context that holds a barrier;
   - a **bounded** loop type.

   A symbol exists in the table only in the types where its use is legal. So any gene, at any position, from any operator, expresses a legal kernel.

5. **Evolution is the shared typed core.** Selection, point mutation, inversion, IS and RIS transposition, gene transposition, and one-point, two-point and gene crossover act on the gene, each draw taken from the type column of the cell it writes. The same operators serve every kingdom, on the GPU.

6. **Fitness** runs the expressed kernel and measures it: correctness against the reference, then speed.

## Why

Andrew, 2026-10-21: "our project plan is to construct a TRUE Gene Expression Programming kingdom for WGSL." A true GEP kingdom for GPU programs lets us evolve our own tools, turning evolution back onto the evolutionary machinery.

## How every design step is checked

Ask: does this step put the rule into the symbol table's types, so every expressed program obeys it? A step that does is on the design.
