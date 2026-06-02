import json
import sys
import os
from IPython.core.interactiveshell import InteractiveShell

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Initialize IPython shell for notebook simulation
shell = InteractiveShell.instance()

# Inject the same path into IPython's sys.path
shell.run_cell("import sys; sys.path.insert(0, '" + project_root.replace("\\", "\\\\") + "')")
shell.run_cell("import matplotlib; matplotlib.use('Agg')")

notebook_path = os.path.join(project_root, "E06_boundary_examples.ipynb")

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Starting sequential execution of E06_boundary_examples.ipynb...")

success = True
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        if not code.strip():
            continue
        print(f"\n--- Executing Cell {i} (ID: {cell.get('id')}) ---")
        print(code)
        
        # Execute code in the IPython shell
        result = shell.run_cell(code)
        if result.error_in_exec:
            print(f"ERROR in Cell {i}:", result.error_in_exec)
            success = False
            break

if success:
    print("\nAll cells executed sequentially with ZERO errors!")
else:
    sys.exit(1)
