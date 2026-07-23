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

The dataset is available on [Zenodo](https://zenodo.org/records/21232615).

## Reproducing Results

To get started install [uv](https://docs.astral.sh/uv/) an extremely fast python package
manager. 

Run a single experiment:
```
uv run main.py -c config/base.yml -c config/strategy/EWC.yml -c config/detector/oracle.yml run
```

Plot results with tensorboard:
```
uv run tensorboard --logdir logs
```

Run experiment workflow `dodo.py` to generate benchmark and dataset:
```sh
uv run doit
```
Will generate a `logs/evaluate/data_frame.csv` and `logs/dd_run/data_frame.csv`.

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
uv run analysis/plot.py
```