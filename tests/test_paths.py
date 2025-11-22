import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.utils.paths import get_paths


@pytest.fixture
def mock_config():
    """Mock configuration data"""
    return {
        "local": {
            "data_dir": "./data",
            "log_dir": "./logs",
            "model_dir": "./models",
            "results_dir": "./results",
        },
        "raven": {
            "data_dir": "/ptmp/dtvu/data/tfbind",
            "log_dir": "/ptmp/dtvu/logs/tfbind",
            "model_dir": "/ptmp/dtvu/models/tfbind",
            "results_dir": "/ptmp/dtvu/results/tfbind",
        },
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_get_paths_local_environment(mock_config, temp_dir):
    """Test get_paths returns correct paths for local environment"""
    # Update mock config to use temp directory
    test_config = mock_config.copy()
    for key in test_config["local"]:
        test_config["local"][key] = str(temp_dir / key)

    mock_yaml = yaml.dump(test_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            paths = get_paths("local")

            # Verify correct paths are returned
            assert "data_dir" in paths
            assert "log_dir" in paths
            assert "model_dir" in paths
            assert "results_dir" in paths

            # Verify mkdir was called for each path
            assert mock_mkdir.call_count == 4


def test_get_paths_raven_environment(mock_config, temp_dir):
    """Test get_paths returns correct paths for raven environment"""
    # Update mock config to use temp directory
    test_config = mock_config.copy()
    for key in test_config["raven"]:
        test_config["raven"][key] = str(temp_dir / key)

    mock_yaml = yaml.dump(test_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            paths = get_paths("raven")

            # Verify correct paths are returned
            assert "data_dir" in paths
            assert "log_dir" in paths
            assert "model_dir" in paths
            assert "results_dir" in paths

            # Verify mkdir was called for each path
            assert mock_mkdir.call_count == 4


def test_get_paths_creates_directories(mock_config, temp_dir):
    """Test that get_paths creates directories if they don't exist"""
    # Update mock config to use temp directory
    test_config = mock_config.copy()
    for key in test_config["local"]:
        test_config["local"][key] = str(temp_dir / key)

    mock_yaml = yaml.dump(test_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            get_paths("local")

            # Verify mkdir was called with correct arguments
            for call in mock_mkdir.call_args_list:
                _, kwargs = call
                assert kwargs["parents"] is True
                assert kwargs["exist_ok"] is True


def test_get_paths_invalid_environment(mock_config):
    """Test get_paths raises KeyError for invalid environment"""
    mock_yaml = yaml.dump(mock_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with pytest.raises(KeyError):
            get_paths("invalid_environment")


def test_get_paths_returns_all_required_keys(mock_config, temp_dir):
    """Test that get_paths returns all expected keys"""
    test_config = mock_config.copy()
    for key in test_config["local"]:
        test_config["local"][key] = str(temp_dir / key)

    mock_yaml = yaml.dump(test_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("pathlib.Path.mkdir"):
            paths = get_paths("local")

            required_keys = {"data_dir", "log_dir", "model_dir", "results_dir"}
            assert set(paths.keys()) == required_keys


def test_get_paths_with_actual_config_file():
    """Integration test using the actual config file"""
    # This test uses the actual paths.yaml file
    # It will create real directories, so we need to be careful
    paths = get_paths("local")

    # Verify all expected keys are present
    assert "data_dir" in paths
    assert "log_dir" in paths
    assert "model_dir" in paths
    assert "results_dir" in paths

    # Verify directories were created
    for path in paths.values():
        assert Path(path).exists()
        assert Path(path).is_dir()
