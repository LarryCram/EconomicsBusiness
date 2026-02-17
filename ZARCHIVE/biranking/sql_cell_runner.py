"""
SQL Cell Runner - Clean Version
Reads a .sql file, extracts cells, reorders by embedded numbers, and runs them in DuckDB.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple
import duckdb

# Configuration
CONFIG = {
    'sql_file': 'y.sql/notebook.sql',
    'db_path': ':memory:',
    'continue_on_error': False
}


def extract_raw_cells(file_path: str) -> List[str]:
    """Extract raw SQL cells from notebook JSON export."""
    # Try multiple encodings for robustness
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    lines = None
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if lines is None:
        with open(file_path, "rb") as f:
            content = f.read().decode('utf-8', errors='replace')
        lines = content.splitlines(keepends=True)
    
    # Extract cells in their original JSON order
    cells = []
    current = []
    in_cell = False
    
    for line in lines:
        if line.strip().startswith('"') and not in_cell:
            in_cell = True
            if line.strip() == '"':
                continue
            current.append(line.lstrip('"'))
        elif line.strip().endswith('"') and in_cell:
            current.append(line.rstrip('"\n') + '\n')
            cell_content = ''.join(current).strip()
            if cell_content.endswith('"'):
                cell_content = cell_content[:-1].rstrip()
            cells.append(cell_content)
            current = []
            in_cell = False
        elif in_cell:
            current.append(line)
    
    return cells


def find_cell_by_execution_order(raw_cells: List[str], target_order: int) -> str | None:
    """Find cell by execution order number."""
    for cell_content in raw_cells:
        first_line = cell_content.split('\n')[0][:100].strip()
        if len(first_line) > 3 and first_line.startswith('--'):
            marker = first_line[3]
            if marker.isdigit() and int(marker) == target_order:
                return cell_content
    return None


def get_execution_order_range(raw_cells: List[str]) -> Tuple[int, int]:
    """Get the min and max execution order numbers from cells."""
    orders = []
    for cell_content in raw_cells:
        first_line = cell_content.split('\n')[0][:100].strip()
        if len(first_line) > 3 and first_line.startswith('--'):
            marker = first_line[3]
            if marker.isdigit():
                orders.append(int(marker))
    
    if not orders:
        return -1, -1
    
    return min(orders), max(orders)


def split_cell_into_statements(cell_content: str) -> List[str]:
    """Split a cell into individual SQL statements."""
    lines = [line for line in cell_content.splitlines()
             if not line.strip().startswith('--') and line.strip()]
    
    if not lines:
        return []
    
    sql_text = '\n'.join(lines)
    return [stmt.strip() for stmt in sql_text.split(';') if stmt.strip()]


def run_sql_cell(cell_content: str, execution_order: int,
                 connection: duckdb.DuckDBPyConnection) -> bool:
    """Execute a SQL cell. Returns True if successful."""
    statements = split_cell_into_statements(cell_content)
    
    if not statements:
        print(f"Cell {execution_order}: No SQL statements found")
        return True
    
    print(f"\n--- Cell {execution_order} ({len(statements)} statements) ---")
    
    for i, statement in enumerate(statements):
        print(f"  [{i+1}] {statement[:100]}{'...' if len(statement) > 100 else ''}")
    
    if not connection:
        print(f"Error: No database connection for cell {execution_order}")
        return False

    print(f"--- Executing Cell {execution_order} ---")
    
    try:
        for i, statement in enumerate(statements):
            print(f"  Executing statement {i+1}/{len(statements)}...")
            
            # Add memory management for large operations
            try:
                # Force garbage collection before heavy operations
                import gc
                gc.collect()
                
                statement_upper = statement.strip().upper()
                
                # Set memory limits and safer settings for large operations
                connection.execute("SET memory_limit='2GB'")
                connection.execute("SET threads=2")  # Limit threads to reduce memory pressure
                
                if statement_upper.startswith('SELECT') or statement_upper.startswith('SHOW'):
                    result = connection.sql(statement)
                    result.show()
                else:
                    connection.execute(statement)
                    connection.commit()
                    
            except Exception as stmt_error:
                error_msg = str(stmt_error)
                print(f"Error in statement {i+1}: {stmt_error}")
                
                # Try unicode fix
                if 'unicode' in error_msg.lower() or 'byte sequence' in error_msg.lower():
                    print(f"Attempting unicode fix...")
                    try:
                        connection.execute("SET default_collation='C'")
                        if statement_upper.startswith('SELECT') or statement_upper.startswith('SHOW'):
                            result = connection.sql(statement)
                            result.show()
                        else:
                            connection.execute(statement)
                            connection.commit()
                        continue
                    except Exception as retry_error:
                        print(f"Unicode fix failed: {retry_error}")
                
                # For memory errors, try to recover
                if 'memory' in error_msg.lower() or 'allocation' in error_msg.lower():
                    print(f"Memory error detected, attempting recovery...")
                    try:
                        connection.execute("PRAGMA enable_optimizer=false")
                        connection.execute("SET memory_limit='2GB'")
                        gc.collect()
                        continue
                    except Exception:
                        pass
                
                raise  # Re-raise the error if no recovery worked
                    
        print(f"Cell {execution_order} executed successfully")
        return True
    except Exception as e:
        print(f"Error in Cell {execution_order}: {e}")
        return False


def main() -> int:
    """Main function."""
    sql_file = CONFIG['sql_file']
    db_path = CONFIG['db_path']
    
    if not Path(sql_file).exists():
        print(f"Error: SQL file '{sql_file}' not found")
        return 1
    
    print(f"Reading SQL from: {sql_file}")
    print(f"Database: {db_path}")
    
    try:
        raw_cells = extract_raw_cells(sql_file)
        min_order, max_order = get_execution_order_range(raw_cells)
        
        print(f"Found {len(raw_cells)} total cells")
        print(f"Execution order range: {min_order}-{max_order}")
        
        if min_order < 0:
            print("No cells with execution order numbers found")
            return 1
            
    except Exception as e:
        print(f"Error reading SQL file: {e}")
        return 1
    
    # Print execution order info
    print("\nCells found:")
    for order in range(min_order, max_order + 1):
        cell = find_cell_by_execution_order(raw_cells, order)
        if cell:
            print(f"  Cell {order}: found")
        else:
            print(f"  Cell {order}: missing")
    
    # Process all cells using range
    cells_to_process = list(range(min_order, max_order + 1))
    
    # Confirm execution
    confirm_msg = f"Execute all cells ({min_order}-{max_order})? (y/N): "
    confirm = input(confirm_msg).strip().lower()
    
    if confirm != "y":
        print("Execution cancelled")
        return 0
    
    # Execute cells
    try:
        with duckdb.connect(db_path) as conn:
            try:
                conn.execute("SET default_collation='C'")
            except Exception:
                pass
            
            failed_cells = []
            for order in cells_to_process:
                cell = find_cell_by_execution_order(raw_cells, order)
                if cell:
                    success = run_sql_cell(cell, order, conn)
                    if not success:
                        failed_cells.append(order)
                        if not CONFIG['continue_on_error']:
                            print(f"Stopping execution due to error in cell {order}")
                            return 1
                        else:
                            print(f"Continuing despite error in cell {order}")
                else:
                    print(f"Cell {order}: not found, skipping")
            
            if failed_cells:
                print(f"\nExecution completed with errors in cells: {failed_cells}")
                return 1
            else:
                print("\nSQL execution completed successfully")
                return 0
                    
    except Exception as e:
        print(f"Database connection error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())