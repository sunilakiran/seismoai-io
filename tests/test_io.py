import numpy as np
import os
import pytest
from seismoai_io import load_sgy, load_folder, normalize_traces


SGY_FOLDER = "C:/Users/Kiran/OneDrive/Desktop/Correlated_Shot_Gathers"
# ⚠️ Upar wali line mein apna actual folder path daalo


# ── load_sgy ─────────────────────────────────────────────────────────────────

def test_load_sgy_shape():
    """Loaded traces must be (167, 4001)."""
    files = [f for f in os.listdir(SGY_FOLDER) if f.endswith('.sgy')]
    traces = load_sgy(os.path.join(SGY_FOLDER, files[0]))
    assert traces.shape == (167, 4001)

def test_load_sgy_returns_ndarray():
    files = [f for f in os.listdir(SGY_FOLDER) if f.endswith('.sgy')]
    traces = load_sgy(os.path.join(SGY_FOLDER, files[0]))
    assert isinstance(traces, np.ndarray)

def test_load_sgy_amplitude_range():
    """Real data: amplitude between -221 and +758."""
    files = [f for f in os.listdir(SGY_FOLDER) if f.endswith('.sgy')]
    traces = load_sgy(os.path.join(SGY_FOLDER, files[0]))
    assert traces.max() < 800
    assert traces.min() > -300

def test_load_sgy_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_sgy("nonexistent_file.sgy")


# ── load_folder ──────────────────────────────────────────────────────────────
def test_load_folder_count():
    """Folder has 166 SGY files."""
    data = load_folder(SGY_FOLDER)
    assert len(data) == 166

def test_load_folder_returns_dict():
    data = load_folder(SGY_FOLDER)
    assert isinstance(data, dict)

def test_load_folder_not_found():
    with pytest.raises(FileNotFoundError):
        load_folder("nonexistent_folder/")


# ── normalize_traces ─────────────────────────────────────────────────────────

def test_normalize_range():
    """Normalized live traces must stay in [-1, 1]."""
    traces = np.random.randn(10, 100) * 50
    result = normalize_traces(traces)
    assert result.max() <= 1.0 + 1e-6
    assert result.min() >= -1.0 - 1e-6

def test_normalize_dead_trace_unchanged():
    """Zero trace must stay zero after normalization."""
    traces = np.zeros((3, 100))
    traces[1] = np.random.randn(100)
    result = normalize_traces(traces)
    assert np.all(result[0] == 0)
    assert np.all(result[2] == 0)

def test_normalize_raises_on_1d():
    with pytest.raises(ValueError):
        normalize_traces(np.zeros(100))