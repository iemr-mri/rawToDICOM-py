"""Self-gating reconstruction sub-package.

Translates the MATLAB self-gating pipeline (SGcine.m and its helpers) into
a clean sequence of composable functions.

Public API
----------
read_sg_data        — separate navigator midlines from k-space (CSSGcineReader.m)
run_pca             — PCA on navigator midlines (PCArunner.m)
interpolate_timeline — resample sparse midlines onto fine temporal grid (timeCorrecter.m)
clean_curves        — bandpass-select cardiac and breathing PCs (curveCleaner.m)
find_cardiac_peaks  — rise/fall window peak detection (PCApeakFinder.m)
find_breath_starts  — breath-phase boundary detection (PCApeakFinder.m)
shuffle_data        — bin acquisitions into cardiac frames (dataShuffler.m)
synchronize_slices  — align SAX slices to diastole (sliceSynchronizer.m)
"""

from rawtoDICOM.reconstruction.self_gating.navigator import (
    clean_curves,
    interpolate_timeline,
    run_pca,
)
from rawtoDICOM.reconstruction.self_gating.peak_finder import (
    find_breath_starts,
    find_cardiac_peaks,
)
from rawtoDICOM.reconstruction.self_gating.reader import SGRawData, read_sg_data
from rawtoDICOM.reconstruction.self_gating.shuffler import ShuffleConfig, shuffle_data
from rawtoDICOM.reconstruction.self_gating.synchronizer import synchronize_slices

__all__ = [
    "SGRawData",
    "ShuffleConfig",
    "clean_curves",
    "find_breath_starts",
    "find_cardiac_peaks",
    "interpolate_timeline",
    "read_sg_data",
    "run_pca",
    "shuffle_data",
    "synchronize_slices",
]
