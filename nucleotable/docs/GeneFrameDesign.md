# Initial Population Generation for Gene Expression Programming

## Objective
To generate an initial population for Gene Expression Programming (GEP) projects using SQL within a DuckDB database.

## Scope
This document outlines the requirements for setting up the database, configuring parameters, and generating the initial population for GEP.

## Requirements

### 1. Database Setup
- Create and manage `symbols` and `terminals` tables within a DuckDB database located at `../nucleotable/data/geneframe.duckdb`.

### 2. Configuration Parameters
- Define and set configuration parameters including:
  - `populationID`
  - `popsize`
  - `internalchromosomes`
  - `generationID`
  - `kingdomID`
  - `HeadLength`
  - `TailLength`

### 3. Max Arity Calculation
- Calculate `MaxArity` from the input arities in the `symbols` table.
- Compute `TailLength` using the formula: `TailLength = HeadLength * (MaxArity - 1) + 1`.

### 4. Initial Population Generation
- Use SQL queries to generate the initial population.
- Ensure random symbol selection for `Head` and `Tail` regions.
- Join generated population with `symbols` and `terminals` tables to retrieve `symbol_name`.

### 5. Data Display
- Use Pandas to fetch and display the generated population data for validation.

## Implementation

1. **Connect to DuckDB and Manage Tables**:
   - Connect to the DuckDB database.
   - Execute SQL queries to drop and create `symbols` and `terminals` tables.
   - Insert predefined symbols and terminals into the tables.

2. **Set Configuration Parameters and Calculate Max Arity**:
   - Define and set initial configuration parameters.
   - Calculate `MaxArity` based on the input arities in the `symbols` table.
   - Calculate `TailLength` using the formula: `TailLength = HeadLength * (MaxArity - 1) + 1`.

3. **Generate Initial Population**:
   - Use SQL to generate the initial population with a structure that includes:
     - `population_id`
     - `generation_id`
     - `individual_id`
     - `chromosome_id`
     - `region` (Head/Tail)
     - `gene_position`
     - `symbol`
   - Ensure the population is generated with random symbols using `RANDOM()` and properly cast the results.
   - Join the generated population with the `symbols` and `terminals` tables to include `symbol_name`.

4. **Display Data Using Pandas**:
   - Use Pandas to fetch and display the results.
   - Ensure the data is presented in a readable format for validation.

By following this specification, the initial population for GEP can be generated, validated, and demonstrated effectively using SQL within DuckDB.

