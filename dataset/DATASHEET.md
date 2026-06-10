# CLDD

CLDD (Continual Learners as a Drift Detection dataset) is a collection of error streams taken
from online continual learning algorithms used to benchmark and develop future drift
detection algorithms.

## Variants

CLDD is provided in two variants, CLDD-A and CLDD-B, which differ in the types of drift
detectors used to control the continual learning algorithms. The two variants may be
concatenated together if desired.

### CLDD-A (N=180) `CLDD-A.parquet`

CLDD-A contains the `oracle` drift detector which perflectly predicts drifts controling
a continual learning algorithm.

The standard evaluation scheme splits the dataset based on seeds into a training set
(seeds 0–4) and a test set (seeds 5–9). Using five training error-streams, each detector
is tuned for a specific strategy-boundary pair.

### CLDD-B (N=420) `CLDD-B.parquet`

CLDD-B contains non-oracle drift detectors that imperfectly controlling a continual
learning algorithm. CLDD-B is potentially more challenging because their is some
ambiguity about if a drift is caused by the continual learner adapting to a detected
drift or the underlying drift. Only 5 seeds exists for CLDD-B configurations.

CLDD-B is provided without a prescribed evaluation methodology. Users may adopt various
evaluation schemes depending on their research objectives, such as holding out specific
seeds, boundary types, strategy types, or detector types.

## Columns

*   `boundary` (String)

    Controls how gradual the transition between tasks.

*   `strategy` (String)
    
    The continual learning strategy generating the error stream.

*   `detector` (String)

    The drift detector informing the continual learning strategy of changes in data
    distribution. The `oracle` detector is a special case where the detector is
    artifically perfect.

*  `seed` (int32)

    The seed identifier. Multiple seeds exist for each combination of boundary,
    strategy, and detector.

*   `ce_stream` (list, float32)

    Test-then-train per-instance cross entropy. Roughly 300,000 values.

*   `error_stream` (list, boolean)

    Test-then-train per-instance errors. Roughly 300,000 values.

*   `trues` (list, int32)

    The ground truth center to each drift.

*   `preds` (list, int32)

    The detector's predicted drifts.


### Types of Boundaries

*  `abrupt`

    Transition between tasks is instantaneous.
    
*  `gradual`

*  `slow`


### Types of Detectors

*  `oracle`


## Underlying Online Continual Learning Scenario


