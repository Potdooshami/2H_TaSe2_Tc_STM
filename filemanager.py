from datetime import datetime
from pathlib import Path
class FileManager:
    '''
    Manages directory paths and generates filepaths for data/plot exports.
    '''
    def __init__(self, root, id_notebook, id_block):
        self.root = root
        self.id_notebook = id_notebook
        self.id_block = id_block
    def timestamp(self):
        now = datetime.now()
        self.id_tmp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    def get_filepath(self, id_file, extension="svg"):
        
        nm_file = f"{'_'.join([self.id_notebook, self.id_block, id_file, self.id_tmp])}.{extension}"
        path = Path(self.root) / self.id_notebook / self.id_block / self.id_tmp
        
        # Create parent directories automatically if they do not exist
        path.mkdir(parents=True, exist_ok=True)
        
        filepath = path / nm_file
        return filepath