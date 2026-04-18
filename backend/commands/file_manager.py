import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger("JARVIS.FileManager")

def delete_file(file_path: str) -> bool:
    """
    Securely deletes a file using Python's modern 'pathlib'.
    
    Why pathlib? In the old days, people used 'os.remove()', but 'pathlib' treats 
    file paths as logical objects, making it cross-platform compatible without having 
    to worry about Windows backslashes (\) vs Linux forward slashes (/).
    """
    target = Path(file_path)
    
    # Always check if the file exists AND is actually a file (not a whole folder) before deleting!
    if target.exists() and target.is_file():
        try:
            target.unlink() # deletes the file
            logger.info(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Could not delete {file_path}. Permission denied or file locked: {e}")
            return False
            
    logger.warning(f"File not found: {file_path}")
    return False
