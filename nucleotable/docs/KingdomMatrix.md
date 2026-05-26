# Kingdom Matrix Architecture

The core architectural diagram for the nucleotable / GeneFrame system.

Each **Kingdom** is a row. Each row has three subgroups:
- **Authoring Subgroup** (left): two paths into GeneFrame — a human language author and an evolutionary agent
- **GeneFrame** (centre): the shared canonical document artefact — NOT part of any authoring lane
- **Expression Subgroup** (right): Compilation → Heuristic Optimisation → Deployment

All arrows between Authoring and GeneFrame are **bidirectional** — the human reads back from GeneFrame, and the evolutionary agent reads back to assess fitness.

The **BotJi Kingdom** is the only row where the human authoring path is labelled **"Human Language"** rather than "Textual Language Author" — because natural language has no formal grammar to author.

---

## Mermaid diagram (full internals, paper-quality)

```mermaid
flowchart LR

%% =========================================================
%% SQL KINGDOM
%% =========================================================
subgraph SQLK[SQL Kingdom]
  direction LR
  subgraph SQLA[Authoring Subgroup]
    direction TB
    subgraph SQLH[Textual Language Author: SQL]
      direction LR
      SQLEDIT[Edit] --> SQLCODE[Hand Written Code]
      SQLCODE --> SQLPARSE[Parse / Visit]
      SQLPARSE --> SQLCODE
      SQLGRAM[ANTLR Grammar] --> SQLPARSE
      SQLCODE --> SQLEDIT
    end
    subgraph SQLEVO[Evolutionary Agent]
      direction LR
      SQLEVOLVE[Evolve] --> SQLKARVA[Karva Genes]
      SQLKARVA --> SQLEXPRESS[Express / Visit]
      SQLEXPRESS --> SQLKARVA
      SQLSYMS[GEP Symbol Table] --> SQLEXPRESS
      SQLSYMS --> SQLPOP[Generate Population]
      SQLPOP --> SQLKARVA
      SQLKARVA --> SQLEVOLVE
    end
  end
  subgraph SQLGF[GeneFrame]
    SQLAST[GeneFrame / AST]
  end
  subgraph SQLX[Expression Subgroup]
    direction TB
    SQLOPT[Compilation / Optimizer]
    SQLPLAN[Execution Plan]
    SQLEXEC[Deployment / Executable]
    SQLOPT --> SQLPLAN --> SQLEXEC
  end
  SQLPARSE <--> SQLAST
  SQLEXPRESS <--> SQLAST
  SQLAST --> SQLOPT
end

%% =========================================================
%% REGEX KINGDOM
%% =========================================================
subgraph REGEXK[Regex Kingdom]
  direction LR
  subgraph REGEXA[Authoring Subgroup]
    direction TB
    subgraph REGEXH[Textual Language Author: Regex]
      direction LR
      RXEDIT[Edit] --> RXCODE[Hand Written Code]
      RXCODE --> RXPARSE[Parse / Visit]
      RXPARSE --> RXCODE
      RXGRAM[ANTLR Grammar] --> RXPARSE
      RXCODE --> RXEDIT
    end
    subgraph REGEXEVO[Evolutionary Agent]
      direction LR
      RXEVOLVE[Evolve] --> RXKARVA[Karva Genes]
      RXKARVA --> RXEXPRESS[Express / Visit]
      RXEXPRESS --> RXKARVA
      RXSYMS[GEP Symbol Table] --> RXEXPRESS
      RXSYMS --> RXPOP[Generate Population]
      RXPOP --> RXKARVA
      RXKARVA --> RXEVOLVE
    end
  end
  subgraph REGEXGF[GeneFrame]
    RXAST[GeneFrame / AST]
  end
  subgraph REGEXX[Expression Subgroup]
    direction TB
    RXOPT[Compilation / Optimizer]
    RXPLAN[Execution Plan]
    RXEXEC[Deployment / Executable]
    RXOPT --> RXPLAN --> RXEXEC
  end
  RXPARSE <--> RXAST
  RXEXPRESS <--> RXAST
  RXAST --> RXOPT
end

%% =========================================================
%% SYMBOLIC REGRESSION KINGDOM
%% =========================================================
subgraph SYMK[SymbolicRegression Kingdom]
  direction LR
  subgraph SYMA[Authoring Subgroup]
    direction TB
    subgraph SYMH[Textual Language Author: SymPy]
      direction LR
      SYMEDIT[Edit] --> SYMCODE[Hand Written Code]
      SYMCODE --> SYMPARSE[Parse / Visit]
      SYMPARSE --> SYMCODE
      SYMGRAM[Grammar / Parser] --> SYMPARSE
      SYMCODE --> SYMEDIT
    end
    subgraph SYMEVO[Evolutionary Agent]
      direction LR
      SYMEVOLVE[Evolve] --> SYMKARVA[Karva Genes]
      SYMKARVA --> SYMEXPRESS[Express / Visit]
      SYMEXPRESS --> SYMKARVA
      SYMSYMS[GEP Symbol Table] --> SYMEXPRESS
      SYMSYMS --> SYMPOP[Generate Population]
      SYMPOP --> SYMKARVA
      SYMKARVA --> SYMEVOLVE
    end
  end
  subgraph SYMGF[GeneFrame]
    SYMAST[GeneFrame / AST]
  end
  subgraph SYMX[Expression Subgroup]
    direction TB
    SYMOPT[Compilation / Optimizer]
    SYMPLAN[Execution Plan]
    SYMEXEC[Deployment / Executable]
    SYMOPT --> SYMPLAN --> SYMEXEC
  end
  SYMPARSE <--> SYMAST
  SYMEXPRESS <--> SYMAST
  SYMAST --> SYMOPT
end

%% =========================================================
%% TERRAFORM KINGDOM
%% =========================================================
subgraph TFK[Terraform Kingdom]
  direction LR
  subgraph TFA[Authoring Subgroup]
    direction TB
    subgraph TFH[Textual Language Author: Terraform]
      direction LR
      TFEDIT[Edit] --> TFCODE[Hand Written Code]
      TFCODE --> TFPARSE[Parse / Visit]
      TFPARSE --> TFCODE
      TFGRAM[ANTLR Grammar] --> TFPARSE
      TFCODE --> TFEDIT
    end
    subgraph TFEVO[Evolutionary Agent]
      direction LR
      TFEVOLVE[Evolve] --> TFKARVA[Karva Genes]
      TFKARVA --> TFEXPRESS[Express / Visit]
      TFEXPRESS --> TFKARVA
      TFSYMS[GEP Symbol Table] --> TFEXPRESS
      TFSYMS --> TFPOP[Generate Population]
      TFPOP --> TFKARVA
      TFKARVA --> TFEVOLVE
    end
  end
  subgraph TFGF[GeneFrame]
    TFAST[GeneFrame / AST]
  end
  subgraph TFX[Expression Subgroup]
    direction TB
    TFOPT[Compilation / Optimizer]
    TFPLAN[Execution Plan]
    TFEXEC[Deployment / Executable]
    TFOPT --> TFPLAN --> TFEXEC
  end
  TFPARSE <--> TFAST
  TFEXPRESS <--> TFAST
  TFAST --> TFOPT
end

%% =========================================================
%% BOTJI KINGDOM
%% =========================================================
subgraph BOTJIK[BotJi Kingdom]
  direction LR
  subgraph BOTJIA[Authoring Subgroup]
    direction TB
    subgraph BOTJIH[Human Language]
      direction LR
      BJEDIT[Edit] --> BJCODE[Natural Language]
      BJCODE --> BJPARSE[Parse / Visit]
      BJPARSE --> BJCODE
      BJGRAM[spaCy + Symbol Table] --> BJPARSE
      BJCODE --> BJEDIT
    end
    subgraph BOTJIEVO[Evolutionary Agent]
      direction LR
      BJEVOLVE[Evolve] --> BJKARVA[Karva Genes]
      BJKARVA --> BJEXPRESS[Express / Visit]
      BJEXPRESS --> BJKARVA
      BJSYMS[GEP Symbol Table] --> BJEXPRESS
      BJSYMS --> BJPOP[Generate Population]
      BJPOP --> BJKARVA
      BJKARVA --> BJEVOLVE
    end
  end
  subgraph BOTJIGF[GeneFrame]
    BJAST[GeneFrame / AST]
  end
  subgraph BOTJIX[Expression Subgroup]
    direction TB
    BJOPT[Compilation / Optimizer]
    BJPLAN[Execution Plan]
    BJEXEC[Deployment / Executable]
    BJOPT --> BJPLAN --> BJEXEC
  end
  BJPARSE <--> BJAST
  BJEXPRESS <--> BJAST
  BJAST --> BJOPT
end
```

---

## The central claim

> Any language that humans write, which resolves to an AST, can also be written by GEP — provided the symbol table enforces correctly typed arities.

Multi-typed GEP with correct arity constraints **cannot construct invalid ASTs**. The evolutionary path is therefore structurally equivalent to the human path, and both converge to the same GeneFrame artefact.

Each new Kingdom requires only:
1. A symbol table (typed GEP symbols with input/output signatures)
2. A grammar / parser (for the human authoring path)
3. An expression subgroup (domain-specific compilation/deployment)

Everything else — the GeneFrame schema, the evolutionary loop, the fitness infrastructure — is shared.

---

## Compact summary table

| Kingdom | Human Author | Grammar / Parser | GeneFrame artifact | Expression target |
|---------|-------------|------------------|--------------------|-------------------|
| SQL | SQL text | ANTLR | Query AST | Optimizer → Execution Plan → DB |
| Regex | Regex pattern | ANTLR | Regex AST | Compiler → Automaton → Matcher |
| SymbolicRegression | SymPy expression | SymPy parser | Expression tree | Simplify → Evaluate |
| Terraform | HCL config | ANTLR | IR | Planner → Dep graph → Apply |
| **BotJi** | **Natural language** | **spaCy + Symbol Table** | **Phylo GeneFrame** | **Compress → Transmit → Decode** |
