# Experiments

Self-contained workspaces for tuning the preprocessing. Nothing in `src/` or
`tests/` imports them, and they add no dependencies or configuration outside
this directory.

| Workspace | Question |
|-----------|----------|
| [`frst/`](frst/README.md) | Which FRST settings best separate microbleeds from clutter? |
| [`inpainting/`](inpainting/README.md) | Which vessel detection strategy (axial or 3D Frangi) and how many in-plane dilation iterations should be used before inpainting? |

The inpainting workspace imports the scoring module of the FRST workspace, so
the workspaces are removed together: delete this whole `experiments/` directory.
Each workspace's tests run separately from the project's test gate, for example
`uv run pytest experiments --no-cov`.
