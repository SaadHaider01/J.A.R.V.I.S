import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger("JARVIS.FileManager")


def create_file(file_path: str, content: str = "") -> bool:
    """
    Creates a new file at the specified path with optional content.
    Parent directories are created automatically if they don't exist.
    """
    try:
        # Expand user (~) and env vars in path
        target = Path(os.path.expandvars(os.path.expanduser(file_path)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info(f"Created file: {target}")
        return True
    except Exception as e:
        logger.error(f"Failed to create file '{file_path}': {e}")
        return False


def create_directory(dir_path: str) -> bool:
    """Creates a new directory/folder at the specified path."""
    try:
        target = Path(os.path.expandvars(os.path.expanduser(dir_path)))
        target.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {target}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory '{dir_path}': {e}")
        return False


def delete_file(file_path: str) -> bool:
    """
    Safely deletes a file by sending it to the Recycle Bin instead of
    permanently destroying it. Falls back to permanent delete if send2trash
    is not available.
    """
    target = Path(file_path)
    if not target.exists():
        logger.warning(f"File not found: {file_path}")
        return False
    try:
        try:
            from send2trash import send2trash as _trash
            _trash(str(target))
            logger.info(f"Sent to Recycle Bin: {file_path}")
        except ImportError:
            target.unlink() if target.is_file() else shutil.rmtree(str(target))
            logger.info(f"Permanently deleted (send2trash not installed): {file_path}")
        return True
    except Exception as e:
        logger.error(f"Could not delete {file_path}: {e}")
        return False


def move_file(source: str, destination: str) -> bool:
    """Moves a file or folder to a new location."""
    try:
        shutil.move(source, destination)
        logger.info(f"Moved '{source}' -> '{destination}'")
        return True
    except Exception as e:
        logger.error(f"Failed to move '{source}': {e}")
        return False


def copy_file(source: str, destination: str) -> bool:
    """Copies a file or folder to a new location."""
    try:
        src = Path(source)
        if src.is_dir():
            shutil.copytree(source, destination)
        else:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        logger.info(f"Copied '{source}' -> '{destination}'")
        return True
    except Exception as e:
        logger.error(f"Failed to copy '{source}': {e}")
        return False


def rename_file(file_path: str, new_name: str) -> bool:
    """Renames a file or folder in-place (keeps same directory)."""
    try:
        target = Path(file_path)
        new_path = target.parent / new_name
        target.rename(new_path)
        logger.info(f"Renamed '{file_path}' -> '{new_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to rename '{file_path}': {e}")
        return False


def search_files(query: str, directory: str = "") -> list[str]:
    """
    Walks a directory tree and returns file paths whose names contain the query.
    Defaults to the user's home directory if no directory is specified.
    """
    if not directory:
        directory = str(Path.home())
    results = []
    query_lower = query.lower()
    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden and system directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in (
                '__pycache__', 'node_modules', '.git', 'AppData'
            )]
            for fname in files:
                if query_lower in fname.lower():
                    results.append(os.path.join(root, fname))
                    if len(results) >= 20:  # Cap results
                        return results
    except PermissionError:
        pass
    return results


def open_file(file_path: str) -> bool:
    """Opens a file with its default Windows application."""
    if not Path(file_path).exists():
        logger.warning(f"File not found: {file_path}")
        return False
    try:
        os.startfile(file_path)
        logger.info(f"Opened file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to open '{file_path}': {e}")
        return False


def list_directory(directory: str = "") -> list[str]:
    """
    Lists the contents of a directory. Returns a list of filenames.
    Defaults to the user's Desktop if no directory is specified.
    """
    if not directory:
        directory = str(Path.home() / "Desktop")
    try:
        items = []
        for item in Path(directory).iterdir():
            prefix = "[DIR] " if item.is_dir() else "[FILE] "
            items.append(f"{prefix}{item.name}")
        return sorted(items)
    except Exception as e:
        logger.error(f"Failed to list directory '{directory}': {e}")
        return []
