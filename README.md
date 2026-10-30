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

The dataset is available [here](https://pub-69d9a50abcb04461bc0b8f04d7a47656.r2.dev/cldd_dataset.zip) for anonymous peer-review and Zenodo.

## Reproducing Results

To get started install [uv](https://docs.astral.sh/uv/) an extremely fast python package
manager. 

Run a single experiment:
```sh
uv run main.py -c config/base.yml -c config/strategy/EWC.yml -c config/detector/oracle.yml run
```

Plot results with tensorboard:
```sh
uv run tensorboard --logdir $LOGDIR
```

Run experiment workflow `dodo.py` to generate benchmark and dataset:
```sh
uv run doit
```
Will generate a `$LOGDIR/evaluate/data_frame.csv` and `$LOGDIR/dd_run/data_frame.csv`.

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

## Result Files

The workflow writes two CSV files to `$LOGDIR`. Copies are in `data/`. The scripts in
`analysis/` read the ones in `$LOGDIR`. Each row is one run, and both files share these columns:

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

`evaluate/data_frame.csv` (1080 rows) holds the final runs of each strategy
with a detector on held-out seeds. It has these extra columns:

| Column | Meaning |
| --- | --- |
| `ocl.accuracy_final` | Accuracy on all tasks at the end of the stream (0 to 1). |
| `ocl.accuracy_seen_avg` | Mean accuracy on the tasks seen so far (0 to 1). |
| `ocl.forward_transfer`, `ocl.backward_transfer` | Forward and backward transfer. |
| `ttt.accuracy` | Test-then-train accuracy over the stream (percent). |

`dd_run/data_frame.csv` (900 rows) holds only the drift scores. Each tuned
detector is replayed on the error stream of the oracle runs from `evaluate`,
so the detector does not change the learner.

NMDT in the tables is `dd.mdt` divided by a length set by `boundary`
(see `compute_nmdt` in `analysis/plots.py`). It is 1.0 when nothing is detected.
