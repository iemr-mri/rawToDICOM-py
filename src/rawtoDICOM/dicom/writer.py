"""DICOM file writer.

Translates convertToDICOM.m.

Receives a coil-combined magnitude image stack, applies geometry corrections,
and writes one multi-frame DICOM file per slice using pydicom.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from rawtoDICOM.bruker.scan import BrukerScan
from rawtoDICOM.dicom.affine import bruker_to_lps, orient_correction_brkraw
from rawtoDICOM.dicom.geometry import apply_corrections, shuffle_slices

# DICOM SOP class UID for MR Image Storage
_MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"


def write_dicom_series(
    images: npt.NDArray[np.floating],
    scan: BrukerScan,
    output_dir: Path,
    heart_rate: float | None = None,
    scan_label: str = "slice",
    scan_index: int | None = None,
) -> list[Path]:
    """Apply geometry corrections and write one DICOM file per slice.

    Translates convertToDICOM.m.

    Pipeline per call:
      1. apply_corrections        — phase-offset circshift + int16 normalisation.
      2. bruker_to_lps            — position/orientation metadata + pack layout.
      3. shuffle_slices           — Bruker interleaved → anatomical order, per pack.
      4. orient_correction_brkraw — phase/coronal/subject-position pixel flips.
      5. _write_slice             — pydicom Dataset construction and save_as.

    Args:
        images:      Float magnitude array [x, y, slices, frames].
        scan:        BrukerScan supplying method, acqp, and visu_pars metadata.
        output_dir:  Directory where .dcm files are written (created if absent).
        heart_rate:  Heart rate in bpm.  When None, estimated from ACQ_repetition_time.
        scan_label:  Prefix for output filenames, e.g. ``"LAX4"`` → ``LAX4_slice_001.dcm``.
        scan_index:  When set, overrides the per-file ``slice_idx`` with a fixed number.
                     Used for SAX stacks where each call writes one slice and the caller
                     tracks the slice position across scans (e.g. ``SAX_slice_003.dcm``).

    Returns:
        List of written file paths, one per slice.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corrected = apply_corrections(images, scan)  # [x, y, slices, frames], int16

    positions_lps, orientations_lps, pack_slices, pack_is_coronal = bruker_to_lps(scan)

    n_slices = corrected.shape[2]
    if n_slices > 1:
        corrected = shuffle_slices(corrected, pack_slices).astype(np.int16)

    corrected = orient_correction_brkraw(
        corrected, scan, pack_slices, pack_is_coronal
    ).astype(np.int16)

    # --- Metadata extracted once ---
    m = scan.method
    vp = scan.visu_pars

    extent = np.asarray(vp.get("VisuCoreExtent", [1.0, 1.0])).ravel()

    patient_id = str(vp.get("VisuSubjectId", "unknown"))
    protocol = str(vp.get("VisuAcquisitionProtocol", ""))
    slice_thick = float(np.asarray(m.get("PVM_SliceThick", 1.0)).ravel()[0])
    slice_offsets = np.asarray(m.get("PVM_EffSliceOffset", np.zeros(n_slices))).ravel()

    acq_tr_ms = float(np.asarray(scan.acqp.get("ACQ_repetition_time", 1000.0)).ravel()[0])
    estimated_hr = 60.0 / (acq_tr_ms / 1000.0)
    hr_bpm = heart_rate if heart_rate is not None else estimated_hr

    # Derive stable UIDs from the subject ID so all scans for the same subject
    # share one StudyInstanceUID and SeriesInstanceUID — makes LAX + SAX load
    # together in DICOM viewers.
    study_uid = generate_uid(entropy_srcs=[patient_id, "study"])
    series_uid = generate_uid(entropy_srcs=[patient_id, str(scan.scan_dir.name)])
    written: list[Path] = []

    for slice_idx in range(n_slices):
        slice_data = corrected[:, :, slice_idx, :]  # [x, y, frames]
        pixel_spacing = [
            float(extent[0]) / slice_data.shape[0],
            float(extent[1]) / slice_data.shape[1],
        ]

        position = positions_lps[min(slice_idx, len(positions_lps) - 1)].tolist()
        orientation = orientations_lps[min(slice_idx, len(orientations_lps) - 1)].tolist()
        slice_loc = float(slice_offsets[min(slice_idx, len(slice_offsets) - 1)])

        slice_number = scan_index if scan_index is not None else slice_idx + 1
        if n_slices == 1 and scan_index is None:
            file_stem = scan_label
        else:
            file_stem = f"{scan_label}_slice_{slice_number:02d}"
        out_path = output_dir / f"{file_stem}.dcm"

        ds = _build_dataset(
            slice_data,
            study_uid=study_uid,
            series_uid=series_uid,
            patient_id=patient_id,
            protocol=protocol,
            slice_thick=slice_thick,
            slice_location=slice_loc,
            position=position,
            orientation=orientation,
            pixel_spacing=pixel_spacing,
            heart_rate=hr_bpm,
            instance_number=slice_number,
        )
        ds.save_as(str(out_path), enforce_file_format=True)
        written.append(out_path)

    return written


def _build_dataset(
    slice_images: npt.NDArray[np.int16],
    *,
    study_uid: str,
    series_uid: str,
    patient_id: str,
    protocol: str,
    slice_thick: float,
    slice_location: float,
    position: list[float],
    orientation: list[float],
    pixel_spacing: list[float],
    heart_rate: float,
    instance_number: int = 1,
    series_number: int = 1,
) -> FileDataset:
    """Construct a pydicom FileDataset for one slice with all frames.

    Args:
        slice_images: int16 array [x, y, frames] after orientation correction.

    Returns:
        A FileDataset ready for save_as().
    """
    cols, rows, n_frames = slice_images.shape  # x=cols, y=rows in DICOM convention

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(_MR_IMAGE_STORAGE)
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = UID(ExplicitVRLittleEndian)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )
    ds.SpecificCharacterSet = "ISO_IR 6"
    ds.SOPClassUID = _MR_IMAGE_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.StudyID = "1"
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number

    # Dates and times
    ds.StudyDate = date_str
    ds.SeriesDate = date_str
    ds.AcquisitionDate = date_str
    ds.StudyTime = time_str
    ds.SeriesTime = time_str
    ds.AcquisitionTime = time_str

    # Patient
    ds.PatientID = patient_id
    ds.PatientName = patient_id

    # Modality and sequence
    ds.Modality = "MR"
    ds.ImageType = "ORIGINAL\\PRIMARY\\OTHER"
    ds.ScanningSequence = "RM\\GR"
    ds.SequenceVariant = "SP"
    ds.MRAcquisitionType = "2D"
    ds.InPlanePhaseEncodingDirection = "ROW"
    ds.ProtocolName = protocol
    ds.AnatomicalOrientation = "QUADRUPED"

    # Geometry
    ds.SliceThickness = slice_thick
    ds.SliceLocation = slice_location
    ds.PixelSpacing = [round(p, 6) for p in pixel_spacing]
    ds.ImagePositionPatient = [round(v, 6) for v in position]
    ds.ImageOrientationPatient = [round(v, 6) for v in orientation]
    ds.AcquisitionMatrix = [0, cols, rows, 0]

    # Heart rate
    ds.HeartRate = int(round(heart_rate))

    # Multi-frame
    ds.NumberOfFrames = n_frames

    # Pixel data attributes
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int16

    # Pixel data: DICOM order is [frames, rows, cols]
    pixel_array = np.transpose(slice_images, (2, 1, 0))  # [frames, y, x]
    ds.PixelData = pixel_array.tobytes()

    return ds
