# nucleotable/db/kingdoms.py

from .connection import DuckDBConnection

def load_kingdoms(conn):
    """
    Insert initial data into the symbols table for various kingdoms.
    """
    queries = [
        # Symbolic Regression
        """
        INSERT INTO symbols (kingdom, symbol, symbol_name, alias, in_S, in_I, in_F, in_B, in_A, in_L, out_S, out_I, out_F, out_B, out_A, out_L)
        VALUES 
        ('Symbolic Regression', 1, 'Addition', '+', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('Symbolic Regression', 2, 'Subtraction', '-', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('Symbolic Regression', 3, 'Multiplication', '*', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('Symbolic Regression', 4, 'Division', '/', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0);
        """,
        # REGEX
        """
        INSERT INTO symbols (kingdom, symbol, symbol_name, alias, in_S, in_I, in_F, in_B, in_A, in_L, out_S, out_I, out_F, out_B, out_A, out_L)
        VALUES 
        ('REGEX', 1, 'Concatenate', 'concat', 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ('REGEX', 2, 'Match', 'match', 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        ('REGEX', 3, 'Left', 'left', 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ('REGEX', 4, 'SubString', 'substring', 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
        """,
        # SQL Functions
        """
        INSERT INTO symbols (kingdom, symbol, symbol_name, alias, in_S, in_I, in_F, in_B, in_A, in_L, out_S, out_I, out_F, out_B, out_A, out_L)
        VALUES 
        -- Arithmetic Functions
        ('SQL', 1, 'Addition (Float)', '+', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 2, 'Addition (String)', '+', 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ('SQL', 3, 'Subtraction', '-', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 4, 'Multiplication', '*', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 5, 'Division', '/', 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),

        -- String Functions
        ('SQL', 6, 'Left', 'LEFT', 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ('SQL', 7, 'Substring', 'SUBSTRING', 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ('SQL', 8, 'Concatenate', 'CONCAT', 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ('SQL', 9, 'Length', 'LENGTH', 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),

        -- Aggregate Functions
        ('SQL', 10, 'Sum', 'SUM', 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 11, 'Average', 'AVG', 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 12, 'Minimum', 'MIN', 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 13, 'Maximum', 'MAX', 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ('SQL', 14, 'Count', 'COUNT', 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),

        -- Logical Functions
        ('SQL', 15, 'And', 'AND', 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0),
        ('SQL', 16, 'Or', 'OR', 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0),
        ('SQL', 17, 'Not', 'NOT', 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0),

        -- Date Functions
        ('SQL', 18, 'Current Date', 'CURRENT_DATE', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        ('SQL', 19, 'Current Timestamp', 'CURRENT_TIMESTAMP', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        ('SQL', 20, 'Date Part', 'DATE_PART', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        """
    ]

    for query in queries:
        conn.execute(query)
        print("Inserted initial data for kingdom.")


# Example usage
if __name__ == "__main__":
    db_conn = DuckDBConnection('nucleotable.db')
    conn = db_conn.connect()
    
    load_kingdoms(conn)

    # Add this line to select and print all data from the symbols table
    print(conn.execute("SELECT * FROM symbols").fetchall())    

    db_conn.close()

