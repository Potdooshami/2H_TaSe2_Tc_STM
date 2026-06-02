import json
import os

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "E06_boundary_examples.ipynb"))

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
new_cells = []

# Cells to delete because their contents are now in external modules or merged
deleted_ids = {"013feba2", "7b101f91", "e2ba5d17", "f9ae285c"}

for cell in cells:
    cell_id = cell.get("id")
    if cell_id in deleted_ids:
        continue
    
    # 1. CDW Collection Cell (ID: 2f5644f9) -> Import cdw_collection instead
    if cell_id == "2f5644f9":
        cell["source"] = ["from crypy_examples.cdw_collection import *\n"]
        cell["outputs"] = []
        cell["execution_count"] = None
        new_cells.append(cell)
        continue
        
    # 2. RGB Boundary Cell (ID: 895eaef7) -> Import RgbBoundary instead
    if cell_id == "895eaef7":
        cell["source"] = ["from rgb_boundary import RgbBoundary\n"]
        cell["outputs"] = []
        cell["execution_count"] = None
        new_cells.append(cell)
        continue
        
    # 3. Test cell for RGB boundary (ID: d4d453e4) -> Keep as interactive test cell, but clean up endregion comment
    if cell_id == "d4d453e4":
        source = cell.get("source", [])
        source = [line for line in source if "# endregion" not in line]
        cell["source"] = source
        new_cells.append(cell)
        continue
        
    # 4. Heisenberg Mat Cell (ID: 11e572e4) -> Import all from heisenberg_mat module instead of defining in-place
    if cell_id == "11e572e4":
        cell["source"] = ["from heisenberg_mat import M, r, g, b, q_C, q_A, heisenberg_product, check_nCA\n"]
        cell["outputs"] = []
        cell["execution_count"] = None
        new_cells.append(cell)
        continue
        
    new_cells.append(cell)

nb["cells"] = new_cells

# Save notebook back with clean indentation
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook refactoring completed successfully!")
