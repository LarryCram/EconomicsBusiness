"""Run SQL lines from a file in a DuckDB session."""

from __future__ import annotations
from pathlib import Path

import sys
from typing import Iterable, List

import duckdb


SQL_IN = "./.sql/notebook.sql"
print(f'{Path(SQL_IN).exists() = }')


def read_sql(path: str) -> List[str]:
	"""Read SQL statements from a file, splitting by semicolon."""
	with open(path, "r", encoding="utf-8") as handle:
		sql_text = handle.read()
	# Remove leading/trailing quotes if the whole file is quoted
	sql_text = sql_text.strip()
	if (sql_text.startswith('"') and sql_text.endswith('"')) or (sql_text.startswith("'") and sql_text.endswith("'")):
		# Remove the quotes and split into lines
		sql_text = sql_text[1:-1]
		lines = sql_text.splitlines()
		# Remove comments and empty lines
		lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('--')]
		# Ensure each line ends with a semicolon and strip stray quotes
		statements = []
		for line in lines:
			stmt = line if line.endswith(';') else line + ';'
			stmt = stmt.strip('"\'')
			statements.append(stmt)
		return statements
	# Remove comments (lines starting with --) and lines that are just a quote
	lines = [line for line in sql_text.splitlines()
			 if not line.strip().startswith('--')
			 and line.strip() not in {'"', "'"}]
	# Add a semicolon to each line (for testing)
	lines = [line if line.endswith(';') else line + ';' for line in lines if line]
	sql_clean = '\n'.join(lines)
	# Split by semicolon, filter empty and lines that are just a quote, and strip stray quotes
	statements = [stmt.strip('"\'').strip() for stmt in sql_clean.split(';')
				  if stmt.strip('"\'').strip() and stmt.strip('"\'').strip() not in {'"', "'"}]
	return statements


def main(argv: Iterable[str] | None = None) -> int:
	args = list(argv) if argv is not None else sys.argv[1:]
	sql_path = args[0] if len(args) >= 1 else SQL_IN
	db_path = args[1] if len(args) >= 2 else ":memory:"

	statements = read_sql(sql_path)
	if not statements:
		print("No SQL statements found.")
		return 0

	print("\n--- First Two SQL Statements ---")
	for idx, statement in enumerate(statements[:2], 1):
		print(f"[{idx}] {statement}")
	print("--- End Preview ---\n")

	confirm = input("Review above SQL. Proceed with execution? (y/N): ").strip().lower()
	if confirm != "y":
		print("Execution cancelled.")
		return 0

	with duckdb.connect(db_path) as conn:
		for statement in statements:
			conn.execute(statement)
	print("SQL execution complete.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())