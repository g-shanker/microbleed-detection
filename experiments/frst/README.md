# FRST experiment workspace

A small, self-contained workspace for finding a better FRST (Fast Radial
Symmetry Transform) configuration. The detector takes the FRST map as an input
channel, so its quality matters a lot, and retraining the detector for every
idea is slow. This workspace instead scores how well the FRST map **alone**
separates annotated microbleeds from background clutter, which takes seconds
per variant.

Nothing in `src/` or `tests/` imports this directory. To remove it, see
[Removing the workspace](#removing-the-workspace).

## Layout

| File | Purpose |
|------|---------|
| `variants.py` | `FrstSpec` (every tunable setting) and `apply`, a configurable FRST. `FrstSpec()` reproduces the pipeline's FRST. |
| `scoring.py` | Turns a FRST map into lesion scores and background peaks, and summarizes them into metrics. |
| `config.py` | Experiment settings and the expansion of sweeps into named variants. |
| `run.py` | Command-line runner. |
| `log.py` | Progress output, mirrored to `run.log`. |
| `config.example.toml` | Annotated example configuration. |
| `tests/` | Tests for the workspace (not part of the project's test gate). |

## Run an experiment

1. Preprocess a dataset with `microbleednet preprocess` (see the main README).
   Only subjects with a lesion mask are used, and only their unaugmented
   variant.
2. Copy the example configuration and edit `dataset_dir` and the sweep:

   ```shell
   cp experiments/frst/config.example.toml experiments/frst/config.toml
   ```

   On Windows PowerShell, use `Copy-Item` instead of `cp`.

3. List the variants without computing anything:

   ```shell
   uv run python -m experiments.frst.run --config experiments/frst/config.toml --dry-run
   ```

4. Run it from the repository root:

   ```shell
   uv run python -m experiments.frst.run --config experiments/frst/config.toml
   ```

`config.toml` and `results/` are git-ignored by `experiments/frst/.gitignore`.

Each variant costs roughly one FRST per subject (about a second for a
256 x 256 x 60 volume for the pipeline settings, more for `mode = "volume"`),
so start with `max_subjects = 10` to 20 and widen the sweep afterwards.

### Progress and logs

The run reports every step as it happens, and mirrors everything to
`results/run.log` (appended to on reruns, each run starting with a
`=== run started … ===` header). A trimmed example:

```text
[18:37:31] dataset: /data/preprocessed
[18:37:31] variants: 22; subjects: 20
[18:37:31] [1/20] subject-001
[18:37:32]   loaded in 0.4s: shape (190, 220, 60), spacing (0.5, 0.5, 2.0)
[18:37:34]   [1/22] pipeline: 3 lesions, 143 background peaks, 1.9s
[18:37:36]   [2/22] radii=[2, 3, 4]: 3 lesions, 151 background peaks, 1.5s
[18:38:40]   subject done in 69s; elapsed 1m09s, about 21m left
```

* Each variant reports how long it took and how many lesions and background
  peaks it scored on that subject. A variant with no peaks or no lesions
  stands out at once.
* After each subject, the log estimates the time left from the average so far.
* The final table and the path of the outputs are logged too.
* If the run fails, the traceback is written to `run.log` before the error is
  raised, so it survives a closed terminal.
* `--dry-run` writes no log.

## What can be varied

Every field of `FrstSpec` in [`variants.py`](variants.py) can appear under
`[base]`, `[sweep]` or `[[variants]]`. The defaults equal the pipeline.

| Field | Meaning |
|-------|---------|
| `mode` | `slice` votes independently per axial slice (the pipeline). `volume` votes in 3D using the voxel spacing from the image header. |
| `radii` | Voting radii in in-plane voxels. The per-radius maps are averaged. |
| `alpha` | Exponent applied to the orientation projection map. |
| `blur_factor` | Gaussian sigma of each radius map, as a fraction of its radius. |
| `gradient_quantile` | Fraction of the weakest nonzero gradients (volume-wide, inside the brain) that do not vote. |
| `edge_margin` | In-plane voxels next to the brain boundary whose gradients do not vote. |
| `normalization` | Scale voting maps per slice (the pipeline) or once per volume. |
| `through_slice_blur_mm` | Gaussian blur across slices, in mm, applied to the final map. |

The configuration offers three ways to build variants. The unmodified pipeline
FRST is always evaluated first, as `pipeline`.

* `[sweep]` with `sweep_mode = "one_at_a_time"` (default) changes one field at
  a time from `[base]`. This shows which settings matter at all.
* `sweep_mode = "grid"` tries every combination of the swept values. The count
  multiplies quickly.
* `[[variants]]` lists hand-written combinations. Use it for the promising
  combinations you found with a one-at-a-time sweep.

`[base]` holds overrides that every swept variant starts from. Set it to the
best settings so far to keep refining. Variants identical to an earlier one are
dropped.

## How variants are scored

For each subject, the FRST map yields:

* one **lesion score** per annotated lesion (26-connected component of the
  mask): the maximum FRST value inside the lesion widened by `lesion_margin`
  voxels, which absorbs small annotation offsets, and
* the **background peaks**: local maxima of the FRST map inside the brain, more
  than `exclusion_radius` voxels from every lesion and at least
  `peak_min_distance` voxels apart.

Scores are compared only within a subject, so they do not depend on the
absolute scale of a variant's FRST values. Results pool the lesions of all
subjects:

* `mean_auc`: for each lesion, the share of its subject's background peaks that
  score lower (ties count half), averaged over lesions. 1.0 means every lesion
  outranks all clutter; 0.5 is chance.
* `recall@K`: share of lesions that fewer than K background peaks of their
  subject match or exceed. This approximates sensitivity when each subject can
  afford about K false candidates.

Variants are ranked by `mean_auc`, then by each `recall@K`. `results.csv` holds
the ranking together with every variant's settings, the lesion and subject
counts, and the seconds spent computing the variant. `variants.json` holds the
exact settings.

## Limitations

* The score is a proxy for detector quality. It ignores how the network uses
  the FRST channel, so confirm the best few variants by training and
  evaluating the detector.
* Only the unaugmented variant of each subject is scored, using the saved,
  already-inpainted volume.
* With few lesions the metrics are noisy. Compare variants on the same subjects
  (the same `seed` and `max_subjects`) and treat small differences as ties.
* `mode = "volume"` assumes in-plane spacing is the smaller of the first two
  voxel sizes and takes the slice spacing from the header.

## Adopting a variant

The pipeline FRST lives in
[`src/microbleednet/core/transforms/frst.py`](../../src/microbleednet/core/transforms/frst.py)
with its constants in `src/microbleednet/constants.py`. Port the winning
settings there, update the FRST tests, then re-run `preprocess` (FRST maps are
stored on disk) and retrain the models. A dataset preprocessed with a different
FRST will not match a model trained with another one.

## Run the workspace tests

The project's default test run only covers `tests/`. Run these explicitly and
skip the coverage gate, which applies to `src/`:

```shell
uv run pytest experiments/frst/tests --no-cov
```

## Removing the workspace

Delete the `experiments/` directory. Nothing else references it, and it adds no
dependencies, configuration or files outside that directory.
