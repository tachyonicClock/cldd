# Continual Learners as a Drift Detection dataset

We investigate error-rate-based drift detection as a mechanism for identifying task
boundaries in continual learning. In continual learning, drift occurs at the boundary
between tasks. In domain incremental learning, new instances arise from a distinct
feature distribution, whereas in class incremental learning, entirely new classes
appear. It is often assumed that a learner has perfect knowledge of task boundaries. For
many applications, task boundaries are unknown and gradual. Unfortunately, many
continual learning algorithms rely on task boundaries to function correctly, for
example, updating their memory, freezing parameters, or performing other actions at the
boundary. In this chapter, we conduct a controlled case study to evaluate drift
detection as a component of continual learners.

The dataset is available on Zenodo: [doi.org/10.5281/zenodo.24124381](https://doi.org/10.5281/zenodo.24124381) (v2.0.0).

## Reproducing Results

To get started install [uv](https://docs.astral.sh/uv/) an extremely fast python package
manager. 

Run a single experiment:
```sh
uv run main.py -c config/base.yml -c config/strategy/EWC.yml -c config/detector/oracle.yml run
```

Plot results with tensorboard:
```sh
uv run tensorboard --logdir $LOGDIR/cldd
```

Run experiment workflow `dodo.py` to generate benchmark and dataset:
```sh
uv run doit
```
Will generate a `$LOGDIR/cldd/<dataset>/evaluate/data_frame.csv` and `$LOGDIR/cldd/<dataset>/dd_run/data_frame.csv`, where `<dataset>` is `rfmnist` or `covertype`. Each dataset directory also has its own `console/` logs and `optuna.journal.log`.

Parts of the workflow can be run independently
```
error_stream             Phase 1 output: create per-seed error streams using tuned strategy HP.
tune_strategy            Phase 1: tune strategy HP using the oracle detector for each boundary.
tune_hp_detector         Phase 2: tune detector HP from all error streams of a strategy/boundary.
tune_detector_strategy   Tune detector and strategy HPs together for each boundary.
evaluate                 Phase 3: evaluate tuned configurations on held-out evaluation seeds.
dd_run                   Replay tuned detector configs on oracle evaluate error streams.
collect_evaluate         Collect all evaluate outputs into one CSV for analysis.
dd_collect               Collect all dd_run outputs into one CSV for analysis.
CLDD_A                   Generate continual learners as a drift detection dataset.
CLDD_B                   Generate continual learners as a drift detection dataset.
```

Analysis with the scripts in `analysis/` can be run with:
```sh
uv run analysis/plots.py
```

### Controlled triggers

The doit task `synthetic` re-runs each tuned strategy (not FT) on RotatedFashionMNIST with a
`schedule` detector (`ScheduleArgs` in `src/drift_detector.py`) that ignores the error
stream. Variants, per strategy, boundary and evaluation seed:

| Variant | Alarms |
|---|---|
| `periodic15k`, `periodic30k`, `periodic60k`, `periodic120k` | every 15k, 30k, 60k or 120k instances, from a random phase |
| `random_adwin` | uniformly random, as many as ADWIN raises (mean over the evaluation seeds, rounded) |
| `oracle_off0`, `oracle_off-6k`, `oracle_off+6k`, `oracle_off-18k`, `oracle_off+18k` | the true boundaries, shifted by that many instances |
| `oracle_miss1` to `oracle_miss4` | the true boundaries with that many dropped |
| `oracle_extra3`, `oracle_extra10`, `oracle_extra20` | the true boundaries plus that many random alarms |
| `silent_ftlr` | none, with the learning rate tuned for FT |
| `ADWIN_more`, `oracle_more` | ADWIN and the oracle on ten more seeds (`EXTRA_EVALUATION_SEEDS` in `dodo.py`) |

Runs go to `$LOGDIR/cldd/rfmnist/synthetic/<boundary>/<strategy>/<variant>/<trial>`.
The task needs the phase 1 and 2 results, and the evaluate runs of ADWIN. It is part of
the default `uv run doit`; run only this part, which finished runs are skipped in, with:
```sh
uv run doit run -n 6 synthetic
```

## Result Files

The workflow writes these CSV files to `$LOGDIR/cldd/<dataset>`, where `<dataset>` is `rfmnist` or `covertype`.
Copies are in `data/<dataset>/`. The scripts in `analysis/` read the ones in `$LOGDIR/cldd`. Each row is one
run, and the files share these columns:

| Column | Meaning |
| --- | --- |
| `strategy` | Continual learning strategy (EWC, FT, LWF, MAS, RWalk, SI). |
| `detector` | Drift detector in the config (`oracle` knows the true boundaries). |
| `detector_label` | Detector name used in plots. `ADWIN_JOINT` is ADWIN with strategy and detector HP tuned together. |
| `boundary` | Drift type: `abrupt`, `gradual` or `slow`. |
| `seed` | Random seed of the run. |
| `dd.precision`, `dd.recall`, `dd.f1` | Detection scores from CapyMOA. |
| `dd.mdt` | Mean detection time, in stream instances. |
| `dd.far` | False alarm rate. |
| `dd.my_tp`, `dd.my_fp`, `dd.my_fn` | Detection counts from `src/metrics.py`. Each true boundary has one window, from `max_early` before to `max_delay` after it. A detection inside a window is a hit, extra detections are false positives, and a window with none is a miss. |
| `dd.my_f1` | F1 score of those counts. Used as F1 in the paper. |

`evaluate/data_frame.csv` (1,380 rows for `rfmnist`, 1,530 for `covertype`) holds the final runs of each strategy
with a detector on held-out seeds. It has these extra columns:

| Column | Meaning |
| --- | --- |
| `ocl.accuracy_final` | Accuracy on all tasks at the end of the stream (0 to 1). |
| `ocl.accuracy_seen_avg` | Mean accuracy on the tasks seen so far (0 to 1). |
| `ocl.forward_transfer`, `ocl.backward_transfer` | Forward and backward transfer. |
| `ttt.accuracy` | Test-then-train accuracy over the stream (percent). |

`dd_run/data_frame.csv` (900 rows per dataset) holds only the drift scores. Each tuned
detector is replayed on the error stream of the oracle runs from `evaluate`,
so the detector does not change the learner.

`replay_cldd_b/data_frame.csv` (3,000 rows per dataset) holds the drift scores of each tuned detector
replayed on the CLDD-B error streams of the four other detectors (named in the `recorder` column).
It is written by `script/replay_cldd_b.py`.

NMDT in the tables is `dd.mdt` divided by a length set by `boundary`
(see `compute_nmdt` in `analysis/plots.py`). It is 1.0 when nothing is detected.
