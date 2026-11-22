# Inpainting experiment workspace

Compares **vessel detection strategies** and **in-plane dilation iteration
counts** for the vessel-inpainting step of preprocessing. The pipeline detects
vessels, dilates the mask in 2D, one slice at a time, by
`VESSEL_DILATION_ITERATIONS` (in
[`constants.py`](../../src/microbleednet/constants.py)), and replaces the masked
voxels by neighbourhood means. Too little masking leaves vessel rims behind,
which become false FRST peaks. Too much erases real microbleeds that touch or
sit near a vessel.

Nothing in `src/` or `tests/` imports this directory. It does import the
scoring module of the [FRST workspace](../frst/README.md), so remove the whole
`experiments/` directory rather than only `experiments/frst/`. See
[Removing the workspace](#removing-the-workspace).

## Strategies

| Strategy | What it does |
|----------|--------------|
| `axial` | The pipeline's detector, unchanged: for every axial slice, Frangi vesselness and a structure-tensor linearity measure are clustered into two groups with KMeans (the smaller group is vessels), then compact round regions are discarded because they look like microbleeds. |
| `frangi3d` | The same recipe on the whole volume instead of one slice at a time, with the pipeline's Frangi parameters passed exactly as the pipeline passes them. See [How the 3D strategy differs](#how-the-3d-strategy-differs). |
| `frangi3d_range` | `frangi3d` with one change: `FRANGI_SIGMAS = (0.5, 1.2, 0.2)` is read as `(start, stop, step)`, as the original paper specifies the scales, so Frangi runs at the four scales 0.5, 0.7, 0.9 and 1.1 instead of at the three scales 0.5, 1.2 and 0.2. |

All strategies feed the same dilation (always in-plane, as in the pipeline), inpainting and
metrics, so any difference comes from which voxels are called vessels.

### How the 3D strategy differs

Only what has to change in 3D does:

* **Frangi parameters are unchanged.** `frangi3d` calls scikit-image's `frangi`
  on the 3D volume with the pipeline's `FRANGI_SIGMAS`, `FRANGI_ALPHA`,
  `FRANGI_BETA` and `FRANGI_BLACK_RIDGES`. Note two properties of those values:
  `FRANGI_SIGMAS` is passed as `sigmas`, exactly as the pipeline does, so the
  tuple `(0.5, 1.2, 0.2)` becomes three scales. If it was meant as a range, as
  the original paper's scales suggest, `frangi3d_range` is that reading. And
  `FRANGI_ALPHA` has no effect in 2D but does in 3D (it penalizes plate-like
  structures), so the same numbers do not give the same behaviour.
* **Rescaling.** With the pipeline's `alpha`, 3D responses are around 1e-3. The
  pipeline clusters the Frangi output next to a linearity feature scaled to 0 to
  1, so unscaled it would not influence the clusters at all. The 3D vesselness
  is divided by its 99.9th percentile in the brain and clipped to 1
  (`VESSELNESS_PERCENTILE` in `detectors.py`). The maximum cannot be used,
  because degenerate voxels inside Frangi can reach 1 on their own.
* **Linearity.** The structure tensor is 3D, and the measure is `|l2 - l3| / 2`
  of the sorted eigenvalues. The 2D `|l1 - l2| / 2` is large for lines and zero
  for blobs; in 3D a tube has two strong gradient directions, so the gap between
  the two smaller eigenvalues is the analogue that is large for tubes and zero
  for blobs and plates.
* **No shape filter.** The pipeline's eccentricity and solidity filter discards
  round regions. A vessel running through the slice plane looks round in a
  single slice, which is exactly what a 3D detector is meant to catch, and
  `regionprops` has no 3D eccentricity anyway.
* **Anisotropic voxels.** The sigmas are in voxels, like in the pipeline. With
  slices several times thicker than the in-plane resolution, 3D filters are
  physically elongated along z. This experiment does not resample.

## Layout

| File | Purpose |
|------|---------|
| `detectors.py` | The vessel detection strategies. |
| `prepare.py` | Runs the real preprocessing with vessel inpainting switched off, and caches the volume, lesion and brain masks, geometry and every strategy's undilated vessel mask. |
| `measure.py` | Dilates, inpaints and measures one subject for one strategy and iteration count. |
| `export.py` | Writes masks and volumes as NIfTI files. |
| `summary.py` | Pools subjects into one row per iteration count and picks a recommendation. |
| `config.py` | Experiment settings. |
| `run.py` | Command-line runner. |
| `analyze.py` | Scores the subjects a run has cached so far, to read partial results. |
| `log.py` | Progress output, mirrored to `run.log`. |
| `config.example.toml` | Annotated example configuration. |
| `tests/` | Tests for the workspace (not part of the project's test gate). |

## Run an experiment

Unlike the FRST workspace, this one starts from the **raw** indexed dataset.
Vessel inpainting happens in the middle of preprocessing, so the saved
preprocessed volumes (already inpainted) cannot be reused. Each subject is run
through the pipeline's own preprocessing with only the inpainting step
captured, so this needs FSL exactly like `microbleednet preprocess` does.
Preprocessing and vessel detection are cached, so only the first run pays for
them.

1. Index your data with `microbleednet index-data` (see the main README). Only
   subjects with a lesion mask are used.
2. Copy the example configuration and edit `dataset_dir`,
   `bias_field_correction` (required), `strategies` and `iterations`:

   ```shell
   cp experiments/inpainting/config.example.toml experiments/inpainting/config.toml
   ```

   On Windows PowerShell, use `Copy-Item` instead of `cp`.

3. Check which subjects, strategies and iteration counts will be used:

   ```shell
   uv run python -m experiments.inpainting.run --config experiments/inpainting/config.toml --dry-run
   ```

4. Run it from the repository root:

   ```shell
   uv run python -m experiments.inpainting.run --config experiments/inpainting/config.toml
   ```

`config.toml` and `results/` are git-ignored by
`experiments/inpainting/.gitignore`. Results land in `results/results.csv`, the
progress log in `results/run.log` and the per-subject cache in
`results/cache/<bias_field_correction>/`. Delete the
cache after changing anything that affects preprocessing, such as the pipeline
code (the cache is already separate per bias-field method). Adding a strategy to
an existing cache only computes that strategy.

### How long it takes

Every strategy and iteration count reruns the strategy's detector once on the
inpainted volume, which gives `residual_vessels`. On a 190 x 220 x 60 volume, one
detection takes about 20 s for each 3D strategy (the range version has one more
scale, so slightly longer) and, parallelized across slices like the pipeline, a
few seconds for `axial`. With 3 strategies, 7 iteration counts and 20 subjects
that adds up to well over an hour. Start with `max_subjects = 5` and fewer
iteration counts to check the setup, and drop strategies you do not need from
`strategies`.

## Saved masks

With `save_masks = true` (the default) the run writes NIfTI files, with the
geometry of the preprocessed volume so that they overlay in any viewer, to
`results/masks/<subject_id>/`:

| File | Contents |
|------|----------|
| `volume.nii.gz` | The volume that goes into vessel inpainting (after normalization, inversion and cropping). |
| `lesions.nii.gz` | The annotated lesions. |
| `brain.nii.gz` | The brain mask. |
| `<strategy>_n<N>.nii.gz` | The dilated vessel mask that was inpainted, for each strategy and iteration count, for example `frangi3d_n2.nii.gz`. |
| `<strategy>_n<N>_inpainted.nii.gz` | The inpainted volume, only with `save_inpainted = true`. It is float data, so it is large. |

To diagnose a failure, open `volume.nii.gz`, then overlay `lesions.nii.gz` and a
vessel mask. Compare `axial_n0` with `frangi3d_n0` to see what each detector
calls a vessel, and `*_n0` with `*_n3` to see what dilation adds. A lesion that
sits inside a mask is one that inpainting erased.

## Progress and logs

The run reports every step as it happens, and mirrors everything to
`results/run.log` (appended to on reruns, each run starting with a
`=== run started … ===` header). A trimmed example:

```text
[18:37:31] strategies: ['axial', 'frangi3d']; iterations: [0, 2]
[18:37:31] subjects: 1; masks saved: True
[18:37:31] [1/1] subject-001
[18:37:31]   preprocessing (not cached; runs FSL)
[18:37:31]   preprocessing done in 142.0s
[18:37:31]   detecting vessels: frangi3d
[18:37:31]   detected frangi3d in 19.8s (201943 vessel voxels)
[18:37:31]   cache written: results/cache/n4/subject-001.npz
[18:37:31]   frangi3d n=2: lesion voxels erased 1/12, lesions hit 1/4, 4.42% of brain inpainted, 38.1s
[18:37:31]   subject done in 612s; elapsed 10m12s, about 3h14m left
```

* Each step says whether it was **computed or taken from the cache**
  (`preprocessing: cached`, `detection axial: cached`) and how long it took.
* Every strategy and iteration count reports, for that subject, how many
  annotated lesion voxels and lesions the mask erased. A single bad subject is
  visible without opening the CSV.
* After each subject, the log estimates the time left from the average so far.
  It is rough, since the first subjects pay for preprocessing.
* The final tables, the recommendations and the paths of the outputs are logged
  too.
* If the run fails, the traceback is written to `run.log` before the error is
  raised, so it survives a closed terminal.
* `--dry-run` writes no log.

## Looking at partial results

A full run on many subjects takes hours, and `results.csv` is only written at the
end. You do not have to wait. Every subject's volume, lesions and vessel masks
are cached as soon as it is prepared, and `analyze` scores just those subjects
with the same metrics and prints the same tables:

```shell
uv run python -m experiments.inpainting.analyze --config experiments/inpainting/config.toml
```

* It works while the full run is still going, or after it was stopped. It reads
  `results/cache/<bias_field_correction>/` and never preprocesses or detects
  anything, so it needs no FSL. Files that the run is writing at that moment, or
  that lack a configured strategy, are skipped and reported.
* Results are written to `results/partial_results.csv`. The run's own
  `results.csv` and `run.log` are not touched.
* It skips `residual_vessels`, because that metric reruns every detector on
  every inpainted volume and dominates the measurement time. Everything else is
  computed exactly as in the full run (the numbers match for the same
  subjects). Add `--residual` to include it.
* `--max-subjects N` uses only the first N cached subjects, in subject-ID order,
  for a quicker answer.
* The output is labelled provisional and states the number of subjects and
  lesions it is based on. With few lesions, `mean_auc` and `recall@K` mostly
  reflect individual cases. `lesion_loss`, `lesions_hit` and `vessel_fraction`
  are steadier, so use those to rule out dilation counts that erase lesions, and
  wait for more subjects before choosing between strategies on `mean_auc`.

## What is measured

For every strategy, iteration count `N` (0 means no dilation) and subject, the
strategy's undilated vessel mask is dilated in-plane `N` times, one slice at a
time, exactly as the pipeline does. The volume is inpainted with that mask.
Then:

| Metric | Meaning | Better |
|--------|---------|--------|
| `lesion_loss` | Share of annotated lesion voxels inside the dilated mask, i.e. erased by inpainting. | lower |
| `lesions_hit` | Share of lesions with at least one voxel erased. | lower |
| `vessel_fraction` | Mean share of brain voxels that get inpainted. | context |
| `residual_vessels` | The same strategy's detector is run again on the inpainted volume. This is the vessel area it still finds, relative to the area it found before inpainting. | lower |
| `mean_auc`, `recall@K` | FRST-only lesion separation on the inpainted volume, with the pipeline's FRST: the same metrics, and the same `lesion_margin`, `exclusion_radius`, `peak_min_distance` and `recall_at` settings, as the [FRST workspace](../frst/README.md#how-variants-are-scored). | higher |

The first two measure the harm of masking, `residual_vessels` and the FRST
metrics measure the benefit.

`results.csv` has one row per strategy and iteration count.

### Recommendation

The runner marks one row per strategy as `recommended`. Only iteration counts
with `lesion_loss <= max_lesion_loss` qualify. Among them, the best `mean_auc`
sets the bar, and the smallest count whose `mean_auc` is within `auc_tolerance`
of it wins. If no count qualifies, the runner says so and recommends nothing for
that strategy. The recommendation is made within each strategy, so compare
strategies by reading the table: a strategy that reaches the same `mean_auc`
with a lower `lesion_loss` is the better one. This is a heuristic; read the full
table before changing the pipeline.

## Limitations

* `residual_vessels` reruns a detector that always finds some "vessel" cluster
  (both detectors cluster into two groups), so it never reaches zero, and it is
  only comparable within a strategy. It cannot show a vessel a detector misses
  in the first place; the FRST metrics and the saved masks can.
* Annotated lesions are the only ground truth. Vessels are judged by the
  strategy's own detector, so any miss it has is invisible to
  `residual_vessels`.
* `lesion_loss` counts annotated voxels, so single-voxel point annotations make
  it coarse. Prefer `lesions_hit` in that case.
* The 3D strategy is an adaptation (see above), not a published method, and its
  clustering has no shape filter, so it can mask compact regions that the axial
  detector would spare. Check `lesion_loss` and the saved masks.
* Only the unaugmented variant of each subject is evaluated, and FRST uses the
  pipeline's current settings. Re-run after you change the FRST.
* The dilation uses the pipeline's cross-shaped footprint, so `N` iterations
  reach `N` voxels along the rows and columns, and fewer diagonally.

## Adopting a result

Set `VESSEL_DILATION_ITERATIONS` in
[`src/microbleednet/constants.py`](../../src/microbleednet/constants.py),
re-run `preprocess` (inpainted volumes and FRST maps are stored on disk) and
retrain. Adopting the `frangi3d` strategy additionally needs a 3D vessel
detector in
[`inpaint_vessels.py`](../../src/microbleednet/core/transforms/inpaint_vessels.py),
which does not exist yet; the implementation in `detectors.py` is the starting
point.

## Run the workspace tests

The project's default test run only covers `tests/`. Run these explicitly and
skip the coverage gate, which applies to `src/`:

```shell
uv run pytest experiments/inpainting/tests --no-cov
```

## Removing the workspace

Delete the `experiments/` directory. Nothing else references it, and it adds no
dependencies, configuration or files outside that directory.
