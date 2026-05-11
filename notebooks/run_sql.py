"""
run_sql.py
----------
Tiny helper for running any of the SQL files in this project against the CSVs
without setting up a database. Useful for demos and interviews.

Usage:
    python notebooks/run_sql.py sql/04_revenue_concentration_pareto.sql

    # or run all of them in order
    python notebooks/run_sql.py --all
"""

import os
import sys
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SQL  = os.path.join(HERE, "..", "sql")


def make_connection():
    con = duckdb.connect()
    con.execute(f"CREATE VIEW customers AS "
                f"SELECT * FROM read_csv_auto('{os.path.join(DATA, 'customers.csv')}', header=true)")
    con.execute(f"CREATE VIEW subscriptions AS "
                f"SELECT * FROM read_csv_auto('{os.path.join(DATA, 'subscriptions.csv')}', header=true)")
    con.execute(f"CREATE VIEW transactions AS "
                f"SELECT * FROM read_csv_auto('{os.path.join(DATA, 'transactions.csv')}', header=true)")
    return con


def first_statement(sql_text):
    """Return the first non-comment statement from a .sql file."""
    cleaned = "\n".join(line.split("--")[0] for line in sql_text.splitlines())
    parts = [p.strip() for p in cleaned.split(";") if p.strip()]
    return parts[0] if parts else None


def run_file(con, path):
    print(f"\n{'='*72}\n  {path}\n{'='*72}")
    with open(path) as f:
        stmt = first_statement(f.read())
    if not stmt:
        print("  (no executable statement)")
        return
    rs = con.execute(stmt).fetchall()
    cols = [d[0] for d in con.description]
    print("  " + " | ".join(cols))
    print("  " + "-" * 70)
    for row in rs[:25]:
        print("  " + " | ".join(str(v) for v in row))
    if len(rs) > 25:
        print(f"  ... {len(rs)-25} more rows ({len(rs)} total)")


def main():
    con = make_connection()
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--all":
        for fname in sorted(os.listdir(SQL)):
            if fname.endswith(".sql"):
                run_file(con, os.path.join(SQL, fname))
    else:
        for path in sys.argv[1:]:
            run_file(con, path)


if __name__ == "__main__":
    main()
