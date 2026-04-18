import pytest
from unittest.mock import patch
from backend.commands.app_launcher import launch_app
from backend.commands.file_manager import delete_file

@patch('backend.commands.app_launcher.os.system')
def test_launch_app_success(mock_system):
    result = launch_app("notepad")
    assert result is True
    mock_system.assert_called_with("start notepad")

def test_launch_app_failure():
    result = launch_app("nonexistent_app")
    assert result is False

@patch('backend.commands.file_manager.Path.exists')
@patch('backend.commands.file_manager.Path.is_file')
@patch('backend.commands.file_manager.Path.unlink')
def test_delete_file_success(mock_unlink, mock_is_file, mock_exists):
    mock_exists.return_value = True
    mock_is_file.return_value = True
    
    result = delete_file("test.txt")
    assert result is True
    mock_unlink.assert_called_once()

@patch('backend.commands.file_manager.Path.exists')
def test_delete_file_not_found(mock_exists):
    mock_exists.return_value = False
    result = delete_file("missing.txt")
    assert result is False
