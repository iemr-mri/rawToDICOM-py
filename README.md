# rawToDICOM-py

Python pipeline for processing raw cardiac MRI data from a Bruker ParaVision scanner into DICOM files. Translated from a MATLAB legacy pipeline with deliberate design improvements.

---

## Usage

```bash
python -m rawtoDICOM \
    --raw-root /data/raw \
    --sorted-root /data/sorted \
    --dicom-root /data/dicom \
    --project AGORA \
    --cohort cohort1
```

The pipeline automatically:
- Sorts raw scans that have not been sorted yet (skips existing destinations).
- Detects self-gating (SG) vs. standard CINE scans per scan directory.
- Converts sorted scans to DICOM where output does not already exist.

### Directory layout

**Raw input** (read-only):
```
raw-root/
└── AGORA/
    └── cohort1/
        └── AGORA2_F1_.../
            ├── 1/          # numeric Bruker scan dirs
            ├── 2/
            └── ...
```

**Sorted output** — scans renamed using `ACQ_scan_name`:
```
sorted-root/
└── AGORA/
    └── CINE/
        └── cohort1/
            └── AGORA2_F1_.../
                ├── LAX4/   # scan name from acqp, not the numeric folder
                ├── SAX1/
                └── ...
```

**DICOM output** — all slices flat in one subject folder, named by scan and slice:
```
dicom-root/
└── AGORA/
    └── cohort1/
        └── CINE_DICOM/
            └── AGORA2_F1_.../
                ├── LAX4_slice_001.dcm
                ├── SAX1_slice_001.dcm
                ├── SAX1_slice_002.dcm
                └── ...
```

### Flags

| Flag | Effect |
|---|---|
| `--skip-sort` | Skip the sort step (data already sorted). |
| `--force-dicom` | Overwrite existing DICOM files. |
| `--force-recon` | Force CS reconstruction even if output exists. |
| `--species rat\|mouse` | Species for SG physiological frequency bands (default: `rat`). |

---

## Stack

- Python 3.10
- MATLAB 2025b legacy reference code on https://github.com/iemr-mri/rawToDICOM

### Dependencies

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",        # FFT, signal filters, curve fitting, interpolation
    "pydicom>=2.4",       # DICOM read/write
    "scikit-learn>=1.4",  # PCA
    "matplotlib>=3.8",    # animated GIF output (slow tests)
]
```

---

## Package structure

```
src/
└── rawtoDICOM/
    ├── __init__.py
    ├── cli.py                         # entry point (replaces rawToDICOM.m)
    ├── config.py                      # PipelineConfig frozen dataclass
    ├── bruker/
    │   ├── params.py                  # JCAMP-DX parameter file parser
    │   ├── raw.py                     # binary k-space reader (PV360+ and pre-PV360)
    │   ├── reader.py                  # load_scan factory, is_sg_scan, scan_plane
    │   └── scan.py                    # BrukerScan frozen dataclass
    ├── reconstruction/
    │   ├── kspace.py                  # sort_kspace, zero_fill_kspace, fft2c, ifft2c
    │   ├── compressed_sensing.py      # reconstruct_cs() — shared by CINE and SG
    │   ├── coil_combination.py        # combine_coils()
    │   └── self_gating/
    │       ├── reader.py              # read_sg_data() — separates midlines from k-space
    │       ├── navigator.py           # run_pca, interpolate_timeline, clean_curves
    │       ├── peak_finder.py         # find_cardiac_peaks, find_breath_starts
    │       ├── shuffler.py            # shuffle_data, ShuffleConfig
    │       └── synchronizer.py        # synchronize_slices
    ├── dicom/
    │   ├── affine.py                  # from_matvec/to_matvec/rotate_affine/flip_affine/flip_voxel_axis_affine/unwrap_to_scanner_xyz + bruker_to_lps, orient_correction_brkraw
    │   ├── geometry.py                # apply_corrections, shuffle_slices
    │   └── writer.py                  # write_dicom_series
    └── pipeline/
        ├── sort.py                    # sort_raw_data (replaces sortRawData.m)
        ├── cine.py                    # process_cine_scan (replaces createDICOMCine.m)
        └── self_gating.py             # process_sg_scan (replaces SGmodule + SGcine)
```

---

## Key architecture decisions

| MATLAB issue | Python approach |
|---|---|
| Two CS reconstruction functions with duplicated algorithm | One `reconstruct_cs()`, parameterized |
| `pm` struct passed everywhere and mutated | Immutable `PipelineConfig` frozen dataclass |
| Intermediate `imageData.mat` saves between pipeline steps | Arrays flow directly between steps |
| `combineCoils.m` uses forward FFT on k-space instead of IFFT | Correct IFFT via normalized `ifft2c` |
| CINE pipeline has no spatial interpolation | 2× k-space zero-padding before IFFT doubles the reconstruction matrix; `PixelSpacing` and phase-offset correction use the actual (padded) image dimensions |
| SG pipeline added later with duplicated reconstruction logic | SG reuses `reconstruct_cs()` and `combine_coils()` |
| Separate CLI paths for CINE vs SG scan processing | Single pipeline auto-detects via `is_sg_scan()` |

---

## Module notes

### bruker/

- `BrukerScan` — frozen dataclass with `acqp`, `method`, `visu_pars`, `data`, `scan_dir`. No mutation after construction.
- `parse_bruker_params()` — full JCAMP-DX parser: scalars, numeric/string/struct arrays, dynamic enums, `@N*(val)` RLE expansion.
- `read_raw_data()` — routes on `ACQ_ScanPipeJobSettings`: PV360+ reads `rawdata.job*`; pre-PV360 reads `fid` with block-size padding. Both produce `[coils, x_points, acquisitions]` complex arrays.
- `is_sg_scan()` — returns `True` when `MidlineRate` is present in `scan.method` (written only by the `segFLASH_CS_SGv3` sequence).
- `scan_plane()` — returns `"SAX"`, `"LAX"`, or `"other"` from `ACQ_scan_name`.

### reconstruction/

- `sort_kspace()` — reshapes `[coils, x, acq]` → `[x, y, slices, frames, flow_enc_dir, coils]`. Three cases: fully sampled (reshape+permute), CS undersampled (scatter via `CSPhaseEncList`), partial echo (leading zero-fill).
- `reconstruct_cs()` — iterative CS: per (slice, coil): IFFT2c → temporal FFT → soft-threshold → temporal IFFT → restore acquired lines → repeat until convergence.
- `combine_coils()` — sum-of-squares: `sqrt(sum(|ifft2c(kspace)|²))` over coil axis.
- `zero_fill_kspace()` — 2× spatial zero-padding by embedding acquired k-space in the centre of a doubled matrix.

### reconstruction/self_gating/

- `read_sg_data()` — separates every `MidlineRate`-th frame (navigator midlines) from regular k-space lines.
- `run_pca()` — z-scores real+imaginary parts across coils, runs sklearn PCA, returns 10 component time-series.
- `interpolate_timeline()` — places sparse midline timestamps on a fine grid (TR/100 ms resolution).
- `clean_curves()` — selects best cardiac and breathing PCs by PSD energy within species-specific bands (rat: cardiac 4–8 Hz, breathing 0.5–1.5 Hz; mouse: 4–10 Hz, 0.5–2.0 Hz), then Butterworth-filters each.
- `find_cardiac_peaks()` — rise/fall window detector; takes argmax within each detected window.
- `find_breath_starts()` — compares peak vs. valley widths, returns the narrower set as cycle boundaries.
- `shuffle_data()` — bins acquisitions into cardiac frames by time since previous peak; averages duplicate k-space lines; navigator midlines averaged into center ky line.
- `synchronize_slices()` — detects diastole in center SAX slice (fewest air pixels), shifts to frame 0, aligns all other slices outward by minimising inter-slice RMS via circular shifting. Fully automated (no interactive ROI).

### dicom/

- `affine.py` — faithful translation of [BrkRaw resolver/affine.py](https://github.com/BrkRaw/brkraw): `from_matvec`, `to_matvec`, `rotate_affine`, `flip_affine`, `flip_voxel_axis_affine`, `unwrap_to_scanner_xyz`, plus `bruker_to_lps` and `orient_correction_brkraw`.
- `apply_corrections()` — phase-offset pixel shift per slice + int16 normalisation (30 000-count ceiling). Pixel offset is computed from the actual image y-size so it remains accurate after zero-padding.
- `shuffle_slices()` — reorders from Bruker's interleaved acquisition order to anatomical order. For multi-pack acquisitions, shuffles independently within each pack (each pack is interleaved separately by the scanner).
- `bruker_to_lps()` — builds `ImagePositionPatient` / `ImageOrientationPatient` for all slices. See below.
- `write_dicom_series()` — writes one multi-frame DICOM file per slice, named `{scan_label}_slice_{n:03d}.dcm`, mapping `visu_pars` fields to standard tags.

#### Image orientation

Getting co-registered LAX+SAX geometry right in a DICOM viewer requires both the pixel array and the position/orientation metadata to be consistent. The pipeline uses a BrkRaw-style affine approach: correct the metadata, then apply only the matching pixel flips.

**`bruker_to_lps` + `orient_correction_brkraw`** — modelled on [BrkRaw](https://github.com/BrkRaw/brkraw)'s `resolve()`. For each slice pack:

1. **Multi-pack layout** — reads `PVM_NSPacks` / `PVM_SPackArrNSlices` / `PVM_SPackArrSliceDistance` / `PVM_SPackArrSliceGap` to handle acquisitions with more than one slice package. Each pack gets its own 4×4 affine.

2. **Affine construction** — builds the affine from `VisuCoreOrientation` (rotation) and `VisuCoreExtent` / `VisuCoreSize` (voxel resolution). Origin is chosen by minimum-projection: all per-slice positions are projected onto the slice normal and the slice with the smallest projection becomes the affine origin, giving a consistent anatomical ordering regardless of acquisition order.

3. **Phase flip** (`ACQ_scaling_phase < 0`) — `flip_voxel_axis_affine(axis=1)`: negates the column-direction cosines in `ImageOrientationPatient` and shifts the origin by `(n_phase − 1) × voxel_phase`. Matched by `np.flip(axis=1)` in `orient_correction_brkraw`.

4. **Coronal flip** (`PVM_SPackArrSliceOrient == "coronal"`) — `flip_voxel_axis_affine(axis=2)` per coronal pack: reverses the slice-position ordering within that pack. Matched by slice reversal within each coronal pack in `orient_correction_brkraw`.

5. **Subject-type / position transform** (`unwrap_to_scanner_xyz`) — converts Bruker's internal subject-relative frame to scanner LPS. This is a pure coordinate-frame conversion (premultiply by flip/rotation matrix); it does not change which voxel is "first", so no pixel reordering is applied for this step.
   - **Quadruped** (rodents — this pipeline): `flip_x` converts LSA+ → RSA+, plus a Z-rotation for Supine / Left / Right gravity orientations.
   - **Biped** (humans): `flip_y` converts LPS+ → LAS+, plus a Z-rotation for Prone / Left / Right.
   - Foot-first entry applies a 180° Y-rotation before either of the above.

`bruker_to_lps()` returns `(positions, orientations, pack_slices, pack_is_coronal)`. `pack_slices` is threaded into `shuffle_slices()` for per-pack interleave reordering; both values are passed to `orient_correction_brkraw()` which applies the matching pixel flips so that voxel (0,0,0) maps to the updated affine origin. `orient_correction_brkraw()` is applied after `shuffle_slices()` in `write_dicom_series()`.

---

## Commands

```bash
pytest                # run tests
mypy src/             # type check
ruff check .          # lint
ruff format .         # format
pytest -m slow        # generate animated GIFs to tests/examples/
```

---

## Tests

All tests run against real Bruker ParaVision data in `tests/raw-data/AGORA/cohort1/` (12 scans, cohort1).

| File | Count | Covers |
|---|---|---|
| `test_reader.py` | 16 | JCAMP-DX parsing, all scan dirs load cleanly |
| `test_scan.py` | 14 | `BrukerScan` contract: `data[0].shape == [coils, x, acq]` |
| `test_kspace.py` | 12 | FFT round-trip, normalization, `sort_kspace` for all three cases |
| `test_compressed_sensing.py` | 11 | Convergence on synthetic data, output shape |
| `test_coil_combination.py` | 7 | Coil axis dropped, output non-negative real |
| `test_self_gating.py` | 25 | SG identification, midlines, PCA, freq detection, peaks, synchronizer |
| `test_pipeline_integration.py` | 12 | Shape assertions per stage, DICOM tag verification, skip-if-exists logic; 2 slow tests generate GIFs |

**SG scan fixtures:**
- `AGORA2_f2` scans 11–24 (`SG Cine SAX`) — SAX self-gated stack; scan 18 is the primary SG fixture.
- `AGORA2_F3` scans 14–28, 29, 31 (`segFLASH_CS_SGv3_slow`, `segFLASH_CS_SGv3_LAX4`) — LAX self-gated.

**Slow test outputs** (`pytest -m slow`, saved to `tests/examples/`):
- `figure1_reconstruction_stages.gif` — k-space before/after CS + final coil-combined images cycling through cardiac frames (scan 15, middle SAX slice).
- `figure2_coreg_lax_sax.gif` — LAX4 frame with yellow SAX co-registration line alongside current SAX slice, cycling through slices (scans 10 + 15).
