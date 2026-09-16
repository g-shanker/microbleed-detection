---
title: MicrobleedNet Quality-of-Life Improvements
description: A prioritized list of small readability and maintainability improvements for MicrobleedNet.
author: MicrobleedNet contributors
ms.date: 2026-09-16
ms.topic: reference
keywords:
  - readability
  - maintainability
  - Python
  - MicrobleedNet
---

## Purpose

This backlog collects small changes that would make the repository easier to read and maintain without changing its pipeline behavior. The items are ordered roughly by value and effort.

## High Priority

### Replace unfinished CLI help text

* Files: [commands.py](microbleed-detection/src/microbleednet/cli/commands.py) and [entrypoint.py](microbleed-detection/src/microbleednet/cli/entrypoint.py)
* Current issue: The shared command help and the application help still say `TODO: write a help message`.
* Suggested change: Give each `CommandSpec` a concise description of its input and output, and replace the application-level placeholder with a useful summary of the pipeline.
* Why it helps: `--help` is the first interface new users see, and accurate help reduces the need to inspect source code.
* Validation: Exercise `microbleednet --help`, `microbleednet describe`, and each command's help output.

### Fix the README prerequisite typo

* File: [README.md](microbleed-detection/README.md)
* Current issue: The prerequisite says `CUDA is optional/`.
* Suggested change: Replace the trailing slash with a period and state whether CPU execution is supported.
* Why it helps: This is a visible documentation defect in the setup path.
* Validation: Review the rendered Markdown and run the documented installation commands where practical.

### Centralize repeated dataset-manifest validation

* File: [configs.py](microbleed-detection/src/microbleednet/orchestration/configs.py)
* Current issue: `SplitConfig`, `TrainConfig`, and related configuration models repeat checks for an existing dataset directory and a required manifest file.
* Suggested change: Add a small private validation helper that accepts the directory, manifest path, and human-readable manifest name, then reuse it from the validators.
* Why it helps: Error wording and validation behavior stay consistent, while each model's validator becomes easier to scan.
* Caution: Keep the helper narrow. The configuration models should still own which manifest they require.
* Validation: Run the configuration tests, especially missing-directory and missing-manifest cases.

## Medium Priority

### Use modern union syntax consistently

* File: [index_data.py](microbleed-detection/src/microbleednet/orchestration/pipes/index_data.py)
* Current issue: `Optional[str]` is used while the rest of the project uses Python 3.13 union syntax such as `str | None`.
* Suggested change: Replace `Optional[str]` with `str | None` and remove the unused `Optional` import.
* Why it helps: One typing style makes signatures easier to scan and matches the project's Python version requirement.
* Validation: Run Ruff and Pyright.

### Name the atomic helper around its lifecycle

* File: [io.py](microbleed-detection/src/microbleednet/core/io.py)
* Current issue: `atomic_path` is concise, but its context-manager contract is not obvious from the name alone.
* Suggested change: Add a short docstring stating that it creates a same-directory temporary path, replaces the destination on successful exit, and removes the temporary file on failure.
* Why it helps: The helper owns the most important durability behavior in the file, so its contract should be visible where it is defined.
* Validation: Keep the existing I/O tests and add or retain tests for replacement and cleanup behavior.

### Add explicit docstrings to simple public I/O functions

* File: [io.py](microbleed-detection/src/microbleednet/core/io.py)
* Current issue: `save_volume`, `save_array`, `save_checkpoint`, and several conversion helpers rely on their names and have no local explanation of their input and output contracts.
* Suggested change: Add short docstrings, especially where suffix handling or atomic replacement matters.
* Why it helps: These functions are the repository's persistence boundary, so their behavior should be understandable without reading implementation details.
* Validation: No behavior change is expected; run the I/O tests and Pyright.

### Replace repeated stage-name literals with a shared type or constants

* Files: [configs.py](microbleed-detection/src/microbleednet/orchestration/configs.py), [train.py](microbleed-detection/src/microbleednet/orchestration/pipes/train.py), [infer.py](microbleed-detection/src/microbleednet/orchestration/pipes/infer.py), and related tests
* Current issue: Strings such as `detector`, `teacher`, and `student` are repeated across layout paths, validation, training, inference, and tests.
* Suggested change: Introduce a small `Literal` type or constants module for stage names, using it only where it improves validation or removes duplicated spelling.
* Why it helps: It reduces typo risk and makes the set of supported stages discoverable.
* Caution: Do not force every string field to use a shared abstraction if that makes serialized manifests less clear.
* Validation: Run the full orchestration test suite and inspect serialized path names.

## Lower Priority

### Make layout method names reflect whether they return a directory or a file

* File: [layouts.py](microbleed-detection/src/microbleednet/orchestration/layouts.py)
* Current issue: Methods such as `preprocessed_volumes_path()` and `patch_dir_path()` return directories, while neighboring methods return files. The distinction is visible only by reading the implementation.
* Suggested change: Use a consistent naming convention such as `*_dir()` for directories and `*_path()` for files, or document the convention on the layout classes.
* Why it helps: Callers can understand the returned path without opening the layout implementation.
* Caution: Renaming has a wider call-site and test impact, so documentation may be the better first step.
* Validation: Run all orchestration tests after any rename.

### Reduce unnecessary `str(...resolve())` conversions at construction sites

* Files: orchestration pipes and [manifests.py](microbleed-detection/src/microbleednet/orchestration/manifests.py)
* Current issue: Several pipes repeatedly resolve a `Path` and immediately convert it to `str` when building manifest records.
* Suggested change: Decide whether persisted path fields should be `Path` values serialized by Pydantic or strings by contract, then apply that choice consistently through the data models.
* Why it helps: A single path representation reduces visual noise and avoids repeated conversion decisions.
* Caution: This affects manifest JSON shape and compatibility, so treat it as a deliberate schema change rather than a mechanical cleanup.
* Validation: Add round-trip tests for every manifest containing paths and document any schema migration requirement.

### Make validation helper names describe their failure scope

* File: [configs.py](microbleed-detection/src/microbleednet/orchestration/configs.py)
* Current issue: Names such as `validate_dataset` can mean directory existence, manifest existence, or cross-manifest consistency depending on the model.
* Suggested change: Prefer names such as `validate_preprocessed_manifest` or `validate_split_manifest_dataset` when a validator checks a specific artifact.
* Why it helps: The model becomes easier to navigate, especially when several validators run in sequence.
* Validation: This is a rename-only change; run configuration tests and Pyright.

### Keep documentation references synchronized with repository files

* File: [manifests.py](microbleed-detection/src/microbleednet/orchestration/manifests.py)
* Current issue: The module docstring refers to `ARCHITECTURE.md`, but that file is not present at the repository root.
* Suggested change: Either add the referenced architecture document, update the reference to an existing document, or remove the parenthetical reference.
* Why it helps: Broken documentation links make readers search for guidance that does not exist.
* Validation: Search for other references to missing documentation files.

## Suggested Order

1. Replace the CLI placeholders and fix the README typo.
2. Centralize repeated configuration validation.
3. Modernize `Optional` and add the `atomic_path` contract docstring.
4. Decide whether stage constants and layout naming justify their broader call-site changes.
5. Defer path-type and manifest-schema changes until there is a compatibility plan.
