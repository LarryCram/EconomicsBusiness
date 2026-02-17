"""
Combined SQL Cell Runner
Reads a .sql file, converts it to a list of cells, and runs them in DuckDB.
Combines functionality from sql_test.py and sql_from_ui.py.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List
import duckdb

# Configuration
CONFIG = {
    'sql_file': '.sql/notebook.sql',
    'db_path': ':memory:',
    'run_first_only': False,  # Run all cells except last
    'preview_only': False,
    'continue_on_error': False
}


def extract_sql_cells(file_path: str) -> List[str]:
    """
    Extract SQL cells from a file where each cell is wrapped in double quotes.
    """
    # Try multiple encodings for robustness
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    lines = None
    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if lines is None:
        # Last resort: read as bytes and handle errors
        with open(file_path, "rb") as f:
            content = f.read().decode('utf-8', errors='replace')
        lines = content.splitlines(keepends=True)
    
    cells = []
    current = []
    in_cell = False
    
    for line in lines:
        # Start of a cell
        if line.strip().startswith('"') and not in_cell:
            in_cell = True
            # If the line is just a quote, skip it
            if line.strip() == '"':
                continue
            # Otherwise, start cell and include line (minus leading quote)
            current.append(line.lstrip('"'))
            continue
            
        # End of a cell
        if line.strip().endswith('"') and in_cell:
            # End of cell, include line (minus trailing quote)
            current.append(line.rstrip('"\n') + '\n')
            # Join and clean up the cell content
            cell_content = ''.join(current).strip()
            if cell_content.endswith('"'):
                cell_content = cell_content[:-1].rstrip()
            cells.append(cell_content)
            current = []
            in_cell = False
            continue
            
        if in_cell:
            current.append(line)
    
    # Now reorder cells by embedded index and filter out 'x' guards
    cell_dict = {}
    for cell in cells:
        first_line = cell.split('\\n')[0][:100].strip()
        if len(first_line) > 3:
            marker = first_line[3]
            if marker == 'x':
                # Skip cells marked with 'x'
                continue
            elif marker.isdigit():
                cell_number = int(marker)
                cell_dict[cell_number] = cell
    
    # Return cells sorted by their embedded numbers
    sorted_cells = [cell_dict[i] for i in sorted(cell_dict.keys())]
    return sorted_cells


def split_cell_into_statements(cell_content: str) -> List[str]:
    """
    Split a cell containing multiple SQL statements into individual statements.
    """
    # Remove comments and empty lines
    lines = [line for line in cell_content.splitlines()
             if not line.strip().startswith('--') and line.strip()]
    
    if not lines:
        return []
    
    # Join lines and split by semicolon
    sql_text = '\n'.join(lines)
    statements = [stmt.strip() for stmt in sql_text.split(';')
                  if stmt.strip()]
    
    return statements


def run_sql_cell(cell_content: str, connection: duckdb.DuckDBPyConnection, 
                 cell_index: int = 0, preview_only: bool = False) -> bool:
    """
    Execute a single SQL cell in the given DuckDB connection.
    Returns True if successful, False otherwise.
    """
    statements = split_cell_into_statements(cell_content)
    
    if not statements:
        print(f"Cell {cell_index + 1}: No SQL statements found")
        return True
    
    print(f"\n--- Cell {cell_index + 1} ({len(statements)} statements) ---")
    
    for i, statement in enumerate(statements):
        print(f"  [{i+1}] {statement[:100]}{'...' if len(statement) > 100 else ''}")
    
    if preview_only:
        return True
    
    if not connection:
        print(f"Error: No database connection for cell {cell_index + 1}")
        return False

    print(f"--- Executing Cell {cell_index + 1} ---")
    
    try:
        for i, statement in enumerate(statements):
            print(f"  Executing statement {i+1}/{len(statements)}...")
            
            try:
                # Use show() for SELECT/SHOW statements, execute() for others
                statement_upper = statement.strip().upper()
                if statement_upper.startswith('SELECT') or statement_upper.startswith('SHOW'):
                    result = connection.sql(statement)
                    result.show()
                else:
                    connection.execute(statement)
                    connection.commit()
            except Exception as stmt_error:
                error_msg = str(stmt_error)
                if 'unicode' in error_msg.lower() or 'byte sequence' in error_msg.lower():
                    print(f"Unicode error in statement {i+1}: {stmt_error}")
                    print(f"Attempting unicode fallback...")
                    
                    try:
                        connection.execute("SET default_collation='C'")
                        if statement_upper.startswith('SELECT') or statement_upper.startswith('SHOW'):
                            result = connection.sql(statement)
                            result.show()
                        else:
                            connection.execute(statement)
                            connection.commit()
                        print(f"  Unicode fallback successful")
                        continue
                    except Exception as retry_error:
                        print(f"  Unicode fallback failed: {retry_error}")
                
                print(f"Error in statement {i+1}: {stmt_error}")
                raise
                    
        print(f"Cell {cell_index + 1} executed successfully")
        return True
    except Exception as e:
        print(f"Error in Cell {cell_index + 1}: {e}")
        print(f"\n--- Failed Cell {cell_index + 1} SQL Content ---")
        print(cell_content)
        print(f"--- End of Cell {cell_index + 1} ---\n")
        return False


def main() -> int:
    """
    Main function to read SQL file, extract cells, and run them in DuckDB.
    """
    sql_file = CONFIG['sql_file']
    db_path = CONFIG['db_path']
    run_first_only = CONFIG['run_first_only']
    preview_only = CONFIG['preview_only']
    
    # Check if SQL file exists
    if not Path(sql_file).exists():
        print(f"Error: SQL file '{sql_file}' not found")
        return 1
    
    print(f"Reading SQL from: {sql_file}")
    print(f"Database: {db_path}")
    
    # Extract cells
    try:
        cells = extract_sql_cells(sql_file)
        print(f"Found {len(cells)} SQL cells")
        
        # Check cell order to detect shuffling
        print("\\n=== CHECKING CELL ORDER ===")
        print("Python Index | Embedded Number | Status")
        print("-------------|-----------------|-------")
        
        extracted_numbers = []
        for i, cell in enumerate(cells):
            first_line = cell.split('\\n')[0][:100].strip()
            python_index = i  # Start from 0 to match embedded numbers
            
            if len(first_line) > 3 and first_line[3].isdigit():
                embedded_number = int(first_line[3])
                extracted_numbers.append(embedded_number)
                
                if embedded_number == python_index:
                    status = "CORRECT"
                else:
                    status = "MISMATCH"
                    
                print(f"     {python_index:2d}      |       {embedded_number:2d}        | {status}")
            else:
                extracted_numbers.append(None)
                print(f"     {python_index:2d}      |      ???        | NO NUMBER")
        
        print()
        expected_sequence = list(range(0, len(cells)))  # Starting from 0
        if extracted_numbers == expected_sequence:
            print("RESULT: Cell sequence is correct")
        else:
            print("RESULT: CELLS ARE SHUFFLED")
            print(f"Python extracted order: {extracted_numbers}")
            print(f"Expected order:         {expected_sequence}")
        print("=== END ORDER CHECK ===\\n")
        
        # Debug: Show the cell dictionary
        print("=== CELL DICTIONARY ===")
        cells_raw = []
        current = []
        in_cell = False
        
        # Re-extract cells to show the dict
        with open(sql_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            if line.strip().startswith('"') and not in_cell:
                in_cell = True
                if line.strip() == '"':
                    continue
                current.append(line.lstrip('"'))
                continue
            if line.strip().endswith('"') and in_cell:
                current.append(line.rstrip('"\\n') + '\\n')
                cell_content = ''.join(current).strip()
                if cell_content.endswith('"'):
                    cell_content = cell_content[:-1].rstrip()
                cells_raw.append(cell_content)
                current = []
                in_cell = False
                continue
            if in_cell:
                current.append(line)
        
        # Build the dict as done in extract_sql_cells
        cell_dict = {}
        for cell in cells_raw:
            first_line = cell.split('\\n')[0][:100].strip()
            if len(first_line) > 3:
                marker = first_line[3]
                if marker == 'x':
                    print(f"SKIPPED (x): {first_line}")
                elif marker.isdigit():
                    cell_number = int(marker)
                    cell_dict[cell_number] = cell
                    print(f"DICT[{cell_number}]: {first_line}")
        
        print(f"\\nFinal dict keys: {sorted(cell_dict.keys())}")
        print("=== END CELL DICTIONARY ===\\n")
        for i, cell in enumerate(cells):
            first_line = cell.split('\\n')[0][:100].strip()
            print(f"Extracted as Cell {i+1}: {first_line}...")
            
            # Extract the 1-digit integer from index 3 of first line
            try:
                if len(first_line) > 3 and first_line[3].isdigit():  # "-- x COMMENTS" -> index 3 is the number
                    cell_number = int(first_line[3])
                    extracted_numbers.append(cell_number)
                    print(f"  -> Found cell number: {cell_number}")
                    if cell_number != i + 1:
                        print(f"  -> *** SEQUENCE ERROR! Expected {i+1}, got {cell_number} ***")
                else:
                    print(f"  -> No cell number found at index 3")
                    extracted_numbers.append(None)
            except (IndexError, ValueError) as e:
                print(f"  -> Error extracting cell number: {e}")
                extracted_numbers.append(None)
            
            if 'edge_list' in cell.lower():
                if 'create' in cell.lower():
                    print(f"  -> CREATES edge_list")
                elif 'select' in cell.lower() or 'from' in cell.lower():
                    print(f"  -> USES edge_list")
        
        print(f"\\nExtracted sequence: {extracted_numbers}")

    except Exception as e:
        print(f"Error reading SQL file: {e}")
        return 1
    
    if not cells:
        print("No SQL cells found")
        return 1
    
    # Process cells (excluding last export cell)
    cells_to_process = [0] if run_first_only else list(range(len(cells)))[:-1]
    
    if preview_only:
        for idx in cells_to_process:
            if idx < len(cells):
                run_sql_cell(cells[idx], None, idx, preview_only=True)
        return 0
    
    # Confirm execution
    if not run_first_only:
        confirm = input(f"Execute all {len(cells)} cells? (y/N): ").strip().lower()
    else:
        confirm = input("Execute first cell? (y/N): ").strip().lower()
    
    if confirm != "y":
        print("Execution cancelled")
        return 0
    
    # Execute cells
    try:
        with duckdb.connect(db_path) as conn:
            # Configure for unicode tolerance
            try:
                conn.execute("SET default_collation='C'")
            except Exception:
                pass
            
            failed_cells = []
            for idx in cells_to_process:
                success = run_sql_cell(cells[idx], conn, idx, preview_only=False)
                if not success:
                    failed_cells.append(idx + 1)
                    if not CONFIG['continue_on_error']:
                        print(f"Stopping execution due to error in cell {idx + 1}")
                        return 1
                    else:
                        print(f"Continuing despite error in cell {idx + 1}")
            
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