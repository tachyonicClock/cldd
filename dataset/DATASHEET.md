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

    **Elastic Weight Consolidation**

    Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.
    A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath,
    C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural
    networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
    https://doi.org/10.1073/pnas.1611835114


*  `SI`

    **Synaptic Intelligence**

    Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic
    Intelligence. International Conference on Machine Learning, 3987–3995.


*  `LWF`

    **Learning Without Forgetting**

    Li, Z., & Hoiem, D. (2016). Learning without forgetting. CoRR, abs/1606.09282.
    http://arxiv.org/abs/1606.09282

*  `MAS`

    **Memory Aware Synapses**

    Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T. (2017).
    Memory aware synapses: Learning what (not) to forget. CoRR, abs/1711.09601.
    http://arxiv.org/abs/1711.09601
    

*  `RWalk`

    **Riemannian Walk**

    Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian Walk
    for Incremental Learning: Understanding Forgetting and Intransigence. In V. Ferrari,
    M. Hebert, C. Sminchisescu, & Y. Weiss (Eds.), Proceedings of the European
    Conference on Computer Vision (ECCV) (Vol. 11215, pp. 556–572). Springer
    International Publishing. https://doi.org/10.1007/978-3-030-01252-6_33


### Types of `detector`

The drift detector informing the continual learning strategy of task boundaries.

*  `oracle`

    A special case where the detector perfectly predicts drifts (used in CLDD-A).

*  `ADWIN`

    **Adaptive Windowing**.

    Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive
    windowing." Proceedings of the 2007 SIAM international conference on data mining.
    Society for Industrial and Applied Mathematics, 2007.

*  `DDM`

    **Drift Detection Method**.

    Gama, Joao, et al. "Learning with drift detection." Advances in Artificial
    Intelligence–SBIA 2004: 17th Brazilian Symposium on Artificial Intelligence, Sao
    Luis, Maranhao, Brazil, September 29-Ocotber 1, 2004.

*  `PH`

    **Page-Hinkley test**.

    Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954), 100-115.

*  `SEED`

    **Streaming Ensemble Entropy-based Drift detection**.

    Huang, David Tse Jung, et al. "Detecting volatility shift in data streams." 2014
    IEEE International Conference on Data Mining. IEEE, 2014.

*  `STEPD`

    **Statistical Test of Equal Proportions Detector**.

    Nishida, Kyosuke, and Koichiro Yamauchi. "Detecting concept drift using
    statistical testing." International conference on discovery science. Berlin,
    Heidelberg: Springer Berlin Heidelberg, 2007.
