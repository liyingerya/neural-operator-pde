# Reproducibility and artifact prerequisites

The public checkout is sufficient to inspect all scientific source, run tests,
generate a new development dataset, train a new baseline, and rebuild the final
presentation. It does **not** include the exact historical weights and run files
needed by the deliberately hash-locked Stage 4–6 replay commands. This distinction
is part of the reproducibility contract, not an implicit download step.

## Install and quick verification

From the repository root, using Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,evaluation]'
python -m pytest -q
python -m examples.gaussian_demo --output /tmp/gaussian_demo.npz
```

The full extra installs NumPy, PyTorch, Matplotlib, and pytest through declared
dependencies. Solver/data-only use can install `.[test]`; ML-dependent tests then
skip if PyTorch is unavailable. Historical measured runs used Python 3.10.8,
NumPy 2.2.6, PyTorch 2.14.0, and Matplotlib 3.10.9 on Apple M1 CPU. Dependency
ranges allow installation elsewhere; they are not a bitwise-reproduction promise.

The demo is inexpensive and writes a new NumPy file to the supplied path. Use a
fresh output filename if retaining a previous demo. It is not a new experiment
in the final synthesis, and its newly printed values are not used in final figures.

## Public summaries and final figures: no local runs needed

```bash
python -m examples.build_final_report
```

This reads only tracked Stage 4/6 summaries, the Stage 3 persistence result, and
the frozen solver-test reference. It writes `docs/results/final_summary.json`,
`docs/results/final_model_comparison.csv`, and five `docs/figures/*.png` files.
It does not import a training pipeline, load checkpoints, run inference, or fit
normalization. Its font cache lives under ignored `runs/stage7/`.

To inspect a rebuild without replacing the curated final files:

```bash
python -m examples.build_final_report --output-root runs/stage7/public_rebuild
```

Historical reports and summaries are never overwritten. Figure pixels may vary
across Matplotlib/font versions; underlying table values and source hashes are
checked independently. The numerical-verification figure is a map of existing
checks, not fabricated or newly measured convergence data.

## From-scratch numerical data and a new baseline

Run in a fresh checkout or with fresh output paths:

```bash
python -m examples.generate_dev_dataset
python -m examples.train_fno
```

The first command creates `data/dev/trajectories.npz` with seed 2026 and the
64-trajectory development configuration. The second uses the fixed 48/8/8 split,
training-only normalization, an eight-pair hard gate, and the historical
100-epoch one-step schedule. It writes `runs/stage3_baseline/` and refuses an
existing run directory. Dataset generation also refuses an existing archive.

Arrays are reproducible in a matching numerical environment; compressed archive
bytes and trained checkpoint bytes are not promised across environments. A newly
trained model is a new run, even if configured like historical A. Do not rename
or alter hashes to make it appear to be the approved checkpoint.

## Exact historical replay: required local artifacts

| Workflow | Prerequisites |
| --- | --- |
| Stage 4 | Original development archive; complete approved `runs/stage3_baseline/` with epoch-31 checkpoint, manifest, and recorded source/normalization provenance |
| Stage 5 training | Same development archive and approved Stage 3 run |
| Stage 5 evaluation | Completed Stage 5 selection/checkpoints; existing Stage 4 OOD/resolution archives and manifests; saved Stage 4 JSON artifacts referenced by the frozen public summary |
| Stage 6 training | Approved Stage 3 run; Stage 5 protocol and matching initialization/source provenance; development archive |
| Stage 6 evaluation/reporting | Completed D run; frozen selected B/C checkpoints, Stage 5 histories/selection; existing Stage 4 datasets and manifests; frozen public summaries |

There is no public artifact download configured. The approved A checkpoint hash
is `300f7c54920c3d5d867fca1e059f7cd1024bc46499fc3c2a15a4b63c8b322f87`.
Loaders stop if bytes, epoch, source, split, or normalization do not match.
Sequentially running fresh training does not guarantee these exact bytes; exact
historical replay requires the original local artifact bundle. The final public
figures and tables remain fully reproducible without that bundle.

## Ordered historical workflow commands

These are the recorded workflows, **not** commands executed again during final
presentation. Use them only when the prerequisites above are satisfied. Default
paths are deliberately shared between dependent stages. Do not rerun commands
against an occupied archive/run path; use a fresh workspace or supported output
arguments. Custom output arguments are not automatically propagated to downstream
commands with fixed defaults.

Stage 4:

```bash
python -m examples.evaluate_stage4 --phase id
python -m examples.generate_stage4_data
python -m examples.evaluate_stage4 --phase ood
python -m examples.benchmark_stage4
python -m examples.generate_stage4_data --resolution
python -m examples.evaluate_stage4 --phase resolution
python -m examples.plot_stage4
python -m examples.summarize_stage4 --output runs/stage4/public_rebuild/stage4_summary.json
```

Stage 5:

```bash
python -m examples.train_stage5
python -m examples.evaluate_stage5
python -m examples.report_stage5 --public runs/stage5/public_rebuild/stage5_summary.json
```

Stage 6:

```bash
python -m examples.train_stage6
python -m examples.evaluate_stage6
python -m examples.report_stage6 --public runs/stage6/public_rebuild/stage6_summary.json
```

Historical reporting scripts use exclusive output creation. Their public
summaries already exist in Git, so the fresh ignored `--output`/`--public` paths
above avoid collisions. Stage 5/6 training cannot overwrite runs or bypass failed
gates. No optional E model is part of the completed study.

Stage 4 timing/hardware records can differ on another host even when numerical
predictions match. Later exact-replay checks against those frozen artifact hashes
therefore require the historical records as well. Rebuilding source code is not
synonymous with reproducing measured wall-clock numbers.

## Frozen source references and audits

See [provenance](stage7_provenance.md) for revisions, source hashes, and the final
audit scope, and [the final report](final_project_report.md) for metric definitions.
The complete regression suite and final public-source checks can be run without
any dataset or model checkpoint. New scientific training was not required to
validate entry-point imports and argument parsing during finalization.
