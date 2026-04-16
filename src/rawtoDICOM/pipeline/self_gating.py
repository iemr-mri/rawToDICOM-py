"""Self-gating pipeline orchestration.

Translates SGmodule.m and SGcine.m.

Pipeline per scan:
  1. read_sg_data        — separate navigator midlines from k-space lines
  2. run_pca             — PCA on navigator midlines
  3. interpolate_timeline — place sparse timestamps on fine grid
  4. clean_curves        — bandpass-select cardiac and breathing PCs
  5. find_cardiac_peaks  — detect cardiac R-peaks in cleaned cardiac curve
  6. find_breath_starts  — detect breath-cycle boundaries
  7. shuffle_data        — bin acquisitions into cardiac frames
  8. reconstruct_cs      — fill undersampled k-space via CS
  9. zero_fill_kspace    — 2× spatial zero-padding
 10. combine_coils       — sum-of-squares coil combination
 11. synchronize_slices  — (SAX stacks only) align slices to diastole
 12. write_dicom_series  — geometry corrections + DICOM write
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rawtoDICOM.bruker.reader import load_scan, scan_plane
from rawtoDICOM.dicom.writer import write_dicom_series
from rawtoDICOM.reconstruction.coil_combination import combine_coils
from rawtoDICOM.reconstruction.compressed_sensing import reconstruct_cs
from rawtoDICOM.reconstruction.kspace import zero_fill_kspace
from rawtoDICOM.reconstruction.self_gating.navigator import (
    clean_curves,
    interpolate_timeline,
    run_pca,
)
from rawtoDICOM.reconstruction.self_gating.peak_finder import (
    find_breath_starts,
    find_cardiac_peaks,
)
from rawtoDICOM.reconstruction.self_gating.reader import read_sg_data
from rawtoDICOM.reconstruction.self_gating.shuffler import ShuffleConfig, shuffle_data
from rawtoDICOM.reconstruction.self_gating.synchronizer import synchronize_slices


def process_sg_scan(
    scan_dir: Path,
    output_dir: Path,
    *,
    species: str = "rat",
    shuffle_config: ShuffleConfig = ShuffleConfig(),
    force_dicom: bool = False,
) -> list[Path]:
    """Run the full SG reconstruction pipeline for one scan directory.

    Translates SGmodule.m + SGcine.m.

    Args:
        scan_dir:       Path to a single Bruker numeric scan directory.
        output_dir:     Destination directory for DICOM files.
        species:        ``"rat"`` or ``"mouse"`` — sets physiological frequency
                        bands used by ``clean_curves``.
        shuffle_config: Cardiac binning parameters (default ShuffleConfig()).
        force_dicom:    When True, existing DICOM files are overwritten.

    Returns:
        List of written DICOM file paths, one per slice.
    """
    output_dir = Path(output_dir)

    if not force_dicom and output_dir.exists() and any(output_dir.glob("*.dcm")):
        return sorted(output_dir.glob("*.dcm"))

    scan = load_scan(scan_dir)

    # --- Step 1: separate midlines from k-space --------------------------------
    sg_raw = read_sg_data(scan)

    # --- Steps 2–4: navigator signal processing --------------------------------
    scores, explained = run_pca(sg_raw.midlines)

    method = scan.method
    tr_ms = float(method["FrameRepTime"])
    midline_rate = int(method["MidlineRate"])

    interpolated, timeline_ms = interpolate_timeline(scores, midline_rate, tr_ms)

    cardiac_curve, breath_curve, cardiac_freq_hz, _breath_freq_hz = clean_curves(
        interpolated, explained, midline_rate, tr_ms, species=species
    )

    # --- Steps 5–6: peak detection ---------------------------------------------
    temporal_resolution_ms = tr_ms / 100.0
    cardiac_peaks = find_cardiac_peaks(cardiac_curve, cardiac_freq_hz, temporal_resolution_ms)
    breath_starts = find_breath_starts(breath_curve)

    # --- Step 7: bin acquisitions into cardiac frames --------------------------
    # shuffled: [x, actual_y, new_frame_count, coils]
    shuffled = shuffle_data(
        sg_raw, cardiac_peaks, breath_starts, timeline_ms, config=shuffle_config
    )

    # Expand to [x, y, 1_slice, frames, coils] for reconstruct_cs
    kspace_for_cs = shuffled[:, :, np.newaxis, :, :]  # [x, y, 1, frames, coils]

    # --- Step 8: CS reconstruction --------------------------------------------
    kspace_reconstructed = reconstruct_cs(kspace_for_cs)  # [x, y, 1, frames, coils]

    # --- Step 9: zero-fill ----------------------------------------------------
    kspace_padded = zero_fill_kspace(kspace_reconstructed)  # [2x, 2y, 1, frames, coils]

    # --- Step 10: coil combination --------------------------------------------
    # combine_coils expects [x, y, slices, frames, coils]
    images = combine_coils(kspace_padded)  # [x, y, 1, frames]

    # --- Step 11: synchronize slices (SAX only) --------------------------------
    plane = scan_plane(scan)
    if plane == "SAX" and images.shape[2] > 1:
        images = synchronize_slices(images)

    # --- Step 12: write DICOM -------------------------------------------------
    return write_dicom_series(images, scan, output_dir)
