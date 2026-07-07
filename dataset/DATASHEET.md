# CLDD

CLDD (Continual Learners as a Drift Detection dataset) is a collection of error streams taken
from online continual learning algorithms used to benchmark and develop future drift
detection algorithms.

Source code to reproduce this dataset is made available at https://github.com/tachyonicClock/cldd.

## Variants

CLDD is provided in two variants, CLDD-A and CLDD-B, which differ in the types of drift
detectors used to control the continual learning algorithms. The two variants may be
concatenated together if desired.

### CLDD-A `CLDD_A.parquet`

CLDD-A contains the `oracle` drift detector which perflectly predicts drifts controling
a continual learning algorithm.

The standard evaluation scheme splits the dataset based on seeds into a training set
(seeds 0–4) and a test set (seeds 5–9). Using five training error-streams, each detector
is tuned for a specific strategy-boundary pair.

CLDD-A is a full factorial of 3 boundaries × 6 strategies × 1 detector × 10 seeds = 180 configurations.

| `boundary`            | `strategy`                      | `detector` | count each |
|:----------------------|:--------------------------------|:-----------|:----------:|
| abrupt, gradual, slow | EWC, FT, LWF, MAS, RWalk, SI    | oracle     |     10     |


### CLDD-B `CLDD_B.parquet`

CLDD-B contains non-oracle drift detectors that imperfectly controlling a continual
learning algorithm. CLDD-B is potentially more challenging because their is some
ambiguity about if a drift is caused by the continual learner adapting to a detected
drift or the underlying drift. Only 5 seeds exists for CLDD-B configurations.

CLDD-B is provided without a prescribed evaluation methodology. Users may adopt various
evaluation schemes depending on their research objectives, such as holding out specific
seeds, boundary types, strategy types, or detector types.

CLDD-B is a full factorial of 3 boundaries × 5 strategies × 5 detectors × 10 seeds = 750 configurations.

| `boundary`             | `strategy`                    | `detector`                    | count each |
|:---------------------|:----------------------------|:----------------------------------|:----------:|
| abrupt, gradual, slow | EWC, LWF, MAS, RWalk, SI   | ADWIN, DDM, PH, SEED, STEPD       |     10     |

## Usage

Data is stored in the parquet format and can be loaded with any compatible library.

```python
# Using pandas https://pandas.pydata.org/
import pandas as pd
cldd_a = pd.read_parquet("CLDD_A.parquet")
```

```python
# Using pyarrow https://arrow.apache.org/docs/python/index.html
import pyarrow.parquet as pq
cldd_a = pq.read_table("CLDD_A.parquet")
cldd_a
```

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


### Types of `boundary`

Controls how gradual the transition between tasks is, simulated by cross-fading between task streams.

*  `abrupt`

    An instantaneous transition between tasks ($w=0$).

*  `gradual`

    A gradual change between tasks ($w=0.1$).

*  `slow`

    A slower gradual change between tasks ($w=0.2$).

### Types of `strategy`

The online continual learning strategy that generates the error stream.

*  `FT`

    **Fine-tuning**; uses no continual learning strategy and does not use task boundaries.

*  `EWC`

    **Elastic Weight Consolidation**; a weight regularisation strategy that penalises changes to important parameters.

*  `SI`

    **Synaptic Intelligence**; a weight regularisation strategy that uses importance-weighted constraints on weight changes, similar to EWC.

*  `LWF`

    **Learning Without Forgetting**; a functional regularisation strategy that regularises the model to behave like a checkpoint to avoid forgetting.

*  `MAS`

    **Memory Aware Synapses**; a weight regularisation strategy that measures per-parameter importance as sensitivity of model outputs to parameter change.

*  `RWalk`

    **Riemannian Walk**; a generalisation of EWC++ and Synaptic Intelligence based on KL-divergence.

### Types of `detector`

The drift detector informing the continual learning strategy of task boundaries.

*  `oracle`

    A special case where the detector perfectly predicts drifts (used in CLDD-A).

*  `ADWIN`

    **Adaptive Windowing**.
    

*  `DDM`

    **Drift Detection Method**.

*  `PH`

    **Page-Hinkley test**.

*  `SEED`

    **Streaming Ensemble Entropy-based Drift detection**.

*  `STEPD`

    **Statistical Test of Equal Proportions Detector**.



