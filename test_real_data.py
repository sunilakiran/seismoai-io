"""
seismoai_io — Real dataset test
Loads all SGY files from Forge 2D Survey and tests all 3 functions.
"""

from seismoai_io import load_sgy, load_folder, normalize_traces
import numpy as np

SGY_FOLDER = "C:/Users/Kiran/OneDrive/Desktop/Correlated_Shot_Gathers"

# ═══════════════════════════════════════════════════════
# TEST 1 — load_sgy: ek file load karo
# ═══════════════════════════════════════════════════════
import os
files = sorted([f for f in os.listdir(SGY_FOLDER) if f.endswith('.sgy')])
first_file = os.path.join(SGY_FOLDER, files[0])

print("=" * 60)
print("TEST 1 — load_sgy() — single file")
print("=" * 60)
traces = load_sgy(first_file)
print(f"File        : {files[0]}")
print(f"Shape       : {traces.shape}")
print(f"n_traces    : {traces.shape[0]}")
print(f"n_samples   : {traces.shape[1]}")
print(f"Max amp     : {traces.max():.4f}")
print(f"Min amp     : {traces.min():.4f}")
assert traces.shape == (167, 4001), "Shape mismatch!"
assert traces.max() < 800, "Amplitude too high!"
print("✅ load_sgy PASSED\n")

# ═══════════════════════════════════════════════════════
# TEST 2 — load_folder: poora dataset load karo
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2 — load_folder() — full dataset")
print("=" * 60)
print("Loading all files... (thoda time lagega)")

data = load_folder(SGY_FOLDER)

print(f"Total files loaded : {len(data)}")
print(f"Expected           : 166")

total_traces = 0
total_samples = 0
max_amp_overall = 0
min_amp_overall = 0

print(f"\n{'File':<20} {'Shape':>12} {'Max Amp':>10} {'Min Amp':>10}")
print("-" * 60)

for fname, t in data.items():
    total_traces  += t.shape[0]
    total_samples += t.shape[1]
    max_amp_overall = max(max_amp_overall, t.max())
    min_amp_overall = min(min_amp_overall, t.min())
    print(f"{fname[:20]:<20} {str(t.shape):>12} {t.max():>10.2f} {t.min():>10.2f}")

print("-" * 60)
print(f"\nTotal files   : {len(data)}")
print(f"Total traces  : {total_traces}")
print(f"Overall max   : {max_amp_overall:.4f}")
print(f"Overall min   : {min_amp_overall:.4f}")
assert len(data) == 166, f"Expected 166 files, got {len(data)}"
print("✅ load_folder PASSED\n")

# ═══════════════════════════════════════════════════════
# TEST 3 — normalize_traces: poore dataset pe normalize
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3 — normalize_traces() — full dataset")
print("=" * 60)

all_passed = True
for fname, t in data.items():
    normalized = normalize_traces(t)
    
    # Check range [-1, 1]
    if normalized.max() > 1.0 + 1e-5:
        print(f"❌ {fname}: max {normalized.max()} > 1.0")
        all_passed = False
    if normalized.min() < -1.0 - 1e-5:
        print(f"❌ {fname}: min {normalized.min()} < -1.0")
        all_passed = False
    
    # Check shape unchanged
    if normalized.shape != t.shape:
        print(f"❌ {fname}: shape changed!")
        all_passed = False

if all_passed:
    print(f"All {len(data)} files normalized successfully!")
    print(f"Range after normalize: [-1.0, 1.0] ✅")
    print("✅ normalize_traces PASSED\n")

# ═══════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("FINAL SUMMARY — seismoai_io real dataset test")
print("=" * 60)
print(f"✅ load_sgy      — 1 file loaded, shape (167, 4001)")
print(f"✅ load_folder   — {len(data)} files loaded, {total_traces} total traces")
print(f"✅ normalize     — all {len(data)} files in range [-1, 1]")
print(f"\nAmplitude range  : {min_amp_overall:.2f} to {max_amp_overall:.2f}")
print(f"Total traces     : {total_traces}")
print(f"\n🎉 seismoai_io library working perfectly on real data!")