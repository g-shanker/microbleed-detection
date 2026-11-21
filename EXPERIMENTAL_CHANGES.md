---
title: Experimental changes
description: Experimental features and diagnostics added on top of the pipeline, and how to remove each one cleanly.
---

## Overview

This file lists experimental additions that sit on top of the original
pipeline. Each section names every file and symbol involved, so you can remove
a feature without searching for leftovers. After removing anything, run
`uv sync` to refresh the console scripts, then the checks in the README.

Two changes are intentionally permanent and not listed here: the brain-edge
exclusion and volume-wide normalization in
`src/microbleednet/core/transforms/frst.py`.

## Point-annotation dilation

Dilates each annotated point into a ball of `point_annotation_radius_mm`
during preprocessing.

* Delete `src/microbleednet/core/transforms/point_annotations.py`.
* Delete `tests/core/transforms/test_point_annotations.py`.
* In `src/microbleednet/orchestration/pipes/preprocess.py`, remove the
  `point_annotations` import, the `if` block under the
  `# point annotation dilation` comment, the `point_annotation_radius_mm`
  parameter of `preprocess_subject`, and the `point_annotation_radius_mm=`
  arguments in `execute`.
* In `src/microbleednet/orchestration/configs.py`, remove the
  `point_annotation_radius_mm` field of `PreprocessConfig` and the resume check
  that compares it with the manifest.
* In `src/microbleednet/orchestration/manifests.py`, remove the
  `point_annotation_radius_mm` field of `PreprocessedDatasetManifest`.
* In `tests/orchestration/pipes/test_preprocess.py`, remove
  `_fake_point_preprocess` and the three tests that use it.
* Remove the `point_annotation_radius_mm` setting from
  `configs/preprocess.toml` and its paragraph from the README.

## Lesion report

The `microbleednet-check-lesions` command, which describes every annotated
lesion of an evaluation run.

* Remove the inpainting report first, because it imports `local_contrast` from
  this module.
* Delete `src/microbleednet/lesion_report.py` and `tests/test_lesion_report.py`.
* Remove the `microbleednet-check-lesions` line from `[project.scripts]` in
  `pyproject.toml`.

## Inpainting dilation report

The `microbleednet-check-inpainting` command, which measures vessel-mask
dilation settings before inpainting and recommends an iteration count.

* Delete `src/microbleednet/inpainting_report.py` and
  `tests/test_inpainting_report.py`.
* Remove the `microbleednet-check-inpainting` line from `[project.scripts]` in
  `pyproject.toml`.
* Optional: in `src/microbleednet/core/engines/processor.py`, fold
  `prepare_volume` back into `preprocess`. The split does not change pipeline
  behavior, so keeping it is harmless.

## FRST variant comparison

The `microbleednet-check-frst` command, which scores FRST variants by how well
FRST alone separates annotated microbleeds from background peaks. The pipeline
FRST in `frst.py` is untouched; the variants live in their own module.

* Delete `src/microbleednet/frst_report.py` and `tests/test_frst_report.py`.
* Delete `src/microbleednet/core/transforms/frst_variants.py` and
  `tests/core/transforms/test_frst_variants.py`.
* Remove the `microbleednet-check-frst` line from `[project.scripts]` in
  `pyproject.toml`.
* The report imports `format_duration` from the inpainting report and
  `local_distances` and `max_within_radius` from the lesion report, so remove
  it before or together with those two.

## Report checkpoints

`src/microbleednet/report_checkpoints.py` saves each finished subject of the
inpainting and FRST reports, so a crashed run can resume with `--checkpoint`.
It is used only by those two reports: delete it and
`tests/test_report_checkpoints.py` once both reports are removed.
