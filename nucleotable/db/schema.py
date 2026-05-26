# nucleotable/db/schema.py

def create_schema(conn):
    """
    Create the initial database schema.
    This function can be customized to define your own table structures.
    """
    # Example table creation queries
    tables = {
        "geneframe": """
            CREATE TABLE IF NOT EXISTS geneframe (
            -- this flat dataframe structure is designed for duckDB and exchange in parquet/arrow
            -- exogenetic data
                user_name      INTEGER    -- set as a default to Karva
             ,  project        STRING     -- each project has a name
             ,  subproject     INTEGER    -- defaults to 0
             ,  version_major  INTEGER    -- defaults to 0
             ,  version_minor  INTEGER    -- defaults to 0
             ,  version_patch  INTEGER    -- defaults to 1 
             ,  experiment_id  INTEGER    -- an experiment ID, part of a user's project/subproject and working directory

            -- phylogenetic data
             ,  population_id  INTEGER    -- all populations ever seen are in this table ... potentially
             ,  generation_id  INTEGER    -- this is a monotomically increasing id as the evolution takes place
             ,  island_id      INTEGER    -- populations with migrants helps us scale up evolution across machine/processes
             ,  ind_id         INTEGER    -- genetic individual, part of an island, a generation, and a population
             ,  birth_gen      INTEGER    -- useful for studying phylogenetic lineage and relationships
             ,  age            INTEGER    -- useful for age layered population evolutionary developement

            -- genetic data
             ,  chromosome_id     INTEGER -- an individual usually has one, but may have many chromosomes. Each has a symbol domain. 
             ,  kingdom           INTEGER -- sets out the genetic domain (symbol set) to express this chromosome
             ,  ishead            BOOLEAN -- denotes if the value is part of the head, or tail as described in GEP.
             ,  gene_seq_id       INTEGER -- this is a monotomically increasing id inside the chromosome, controlling seq position
             ,  symbol            INTEGER -- this is ref ID to functions in the symbol table, OR terminals in the catelogue 
 
            );
        """,
        "symbols": """
            CREATE TABLE IF NOT EXISTS symbols (

                  kingdom            VARCHAR   -- the name of the "expression kingdom" the symbol is a part of
                , symbol             INTEGER   -- the symbol in the geneframe. (kingdom.symbol must be unique)
                , symbol_name        VARCHAR   -- human readable name
                , alias              VARCHAR   -- name of the function in the target expression language
                , in_S               INTEGER   -- the arity of string inputs
                , in_I               INTEGER   -- the arity of integer inputs
                , in_F               INTEGER   -- the arity of float inputs
                , in_B               INTEGER   -- the arity of boolean inputs
                , in_A               INTEGER   -- the arity of array typed inputs (like numpy array inputs :-)
                , in_L               INTEGER   -- the arity of list typed inputs
                , out_S              INTEGER   -- the arity of string outputs 
                , out_I              INTEGER   -- the arity of integer outputs 
                , out_F              INTEGER   -- the arity of float outputs 
                , out_B              INTEGER   -- the arity of boolean outputs
                , out_A              INTEGER   -- the arity of array typed outputs (like numpy arrays)
                , out_L              INTEGER   -- the arity of list typed outputs

            );
        """
    }

    # Execute each table creation query
    for table_name, create_query in tables.items():
        conn.execute(create_query)
        print(f"Table '{table_name}' created or already exists.")

# Example usage
if __name__ == "__main__":
    from connection import DuckDBConnection
    
    db_conn = DuckDBConnection('nucleotable.db')
    conn = db_conn.connect()
    
    create_schema(conn)
    
    db_conn.close()

