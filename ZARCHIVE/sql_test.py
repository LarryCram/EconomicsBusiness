"""
Script to extract and print the first two SQL cells from a DuckDB UI notebook export, assuming each cell is wrapped in double quotes.
"""

def print_first_two_sql_cells(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
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
            current.append(line.rstrip('"'))
            # Join and remove any trailing quote left at the end of the cell
            cell_content = ''.join(current).strip()
            if cell_content.endswith('"'):
                cell_content = cell_content[:-1].rstrip()
            cells.append(cell_content)
            current = []
            in_cell = False
            if len(cells) == 2:
                break
            continue
        if in_cell:
            current.append(line)
    # Print the first two cells
    for i, cell in enumerate(cells, 1):
        print(f"--- CELL {i} ---\n{cell}\n")

if __name__ == "__main__":
    sql_path = ".sql/notebook.sql"
    print_first_two_sql_cells(sql_path)
