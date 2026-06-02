import sys
import os
import traceback

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("IPython/Python environment import test:")
try:
    import crypy_examples.cdw_collection
    print("SUCCESS: crypy_examples.cdw_collection imported successfully!")
except Exception as e:
    print("FAILED to import:")
    traceback.print_exc()
