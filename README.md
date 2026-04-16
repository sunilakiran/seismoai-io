# seismoai-io

SGY file loader for seismic data — part of the **seismoai** pipeline.

Built on real data from the **Forge 2D Survey (2017)**: 166 SGY files, 27,722 total traces.

## Install

```bash
pip install seismoai-io
```

## Usage

```python
from seismoai_io import load_sgy, load_folder, normalize_traces

# Single file load karo
traces = load_sgy("path/to/file.sgy")
print(traces.shape)  # (167, 4001)

# Poora folder load karo
data = load_folder("path/to/folder/")
print(len(data))  # 166

# Normalize karo
normalized = normalize_traces(traces)
print(normalized.max())   # 1.0
print(normalized.min())   # -1.0
```

## Functions

| Function | What it does |
|---|---|
| `load_sgy(filepath)` | Single SGY file load karta hai → (167, 4001) numpy array |
| `load_folder(folder_path)` | Poora folder load karta hai → dict of arrays |
| `normalize_traces(traces)` | Har trace ko [-1, 1] range mein laata hai |

## Real Data Stats

Forge 2D Survey (2017) — measured from actual files:

- **Total files**: 166 SGY files
- **Per file**: 167 traces × 4001 samples (1ms interval)
- **Format**: IEEE float, Little Endian (INOVA instrument)
- **Amplitude range**: -221.50 to +758.22
- **Total traces**: 27,722

## Pipeline Position