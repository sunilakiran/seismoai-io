"""
seismoai_io — Load and prepare SGY seismic files.

Forge 2D Survey (2017):
    - 167 traces per file, 4001 samples, 1ms sample interval
    - Format: IEEE float, Little Endian
    - This module is the foundation — all other modules use its output.
"""

import struct
import os
import numpy as np


def load_sgy(filepath: str) -> np.ndarray:
    """
    Load a single SGY file and return traces as a 2D numpy array.

    Reads Forge 2D Survey SGY files (IEEE float, Little Endian format).
    Skips the 3200-byte text header and 400-byte binary header, then
    reads each trace (240-byte header + samples).

    Parameters
    ----------
    filepath : str
        Path to the .sgy file on disk.

    Returns
    -------
    np.ndarray
        2D array of shape (n_traces, n_samples).
        For Forge data: (167, 4001).

    Raises
    ------
    FileNotFoundError
        If the given filepath does not exist.
    ValueError
        If the file is too small to be a valid SGY file.

    Examples
    --------
    >>> traces = load_sgy("data/my_file.sgy")
    >>> traces.shape
    (167, 4001)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        total_size = f.tell()
        if total_size < 3600:
            raise ValueError(f"File too small to be a valid SGY: {filepath}")

        # Read binary header (Little Endian — INOVA instrument format)
        f.seek(3200)
        bin_hdr = f.read(400)
        n_samples = struct.unpack('<H', bin_hdr[20:22])[0]

        trace_bytes = 240 + n_samples * 4
        n_traces = (total_size - 3600) // trace_bytes

        traces = []
        for i in range(n_traces):
            f.seek(3600 + i * trace_bytes)
            f.read(240)  # skip trace header
            raw = f.read(n_samples * 4)
            if len(raw) < n_samples * 4:
                break
            trace = np.frombuffer(raw, dtype='<f4').copy()
            traces.append(trace)

    return np.array(traces)


def load_folder(folder_path: str) -> dict:
    """
    Load all SGY files from a folder.

    Finds every .sgy file in the given folder and loads each one
    using load_sgy(). Returns a dictionary mapping filename to traces.

    Parameters
    ----------
    folder_path : str
        Path to folder containing .sgy files.

    Returns
    -------
    dict
        Keys: filename (str), Values: np.ndarray of shape (n_traces, n_samples).

    Raises
    ------
    FileNotFoundError
        If folder_path does not exist.

    Examples
    --------
    >>> data = load_folder("data/")
    >>> len(data)
    50
    >>> list(data.values())[0].shape
    (167, 4001)
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    result = {}
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith('.sgy'):
            fpath = os.path.join(folder_path, fname)
            result[fname] = load_sgy(fpath)

    return result


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    """
    Normalize each trace to the range [-1, 1].

    Divides each trace by its maximum absolute amplitude. Traces with
    zero amplitude (dead traces) are left unchanged to avoid division
    by zero.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples).

    Returns
    -------
    np.ndarray
        Normalized 2D array of same shape. Each live trace has
        values in [-1, 1].

    Raises
    ------
    ValueError
        If traces is not a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> traces = np.array([[0, 4, -8], [0, 0, 0]], dtype=float)
    >>> normalize_traces(traces)
    array([[ 0.  ,  0.5 , -1.  ],
           [ 0.  ,  0.  ,  0.  ]])
    """
    if traces.ndim != 2:
        raise ValueError(
            f"traces must be 2D (n_traces, n_samples), got shape {traces.shape}"
        )

    normalized = np.zeros_like(traces, dtype=np.float32)
    max_amps = np.max(np.abs(traces), axis=1)

    for i, max_amp in enumerate(max_amps):
        if max_amp > 0:
            normalized[i] = traces[i] / max_amp

    return normalized