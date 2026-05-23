
Our goal is to evaluate error-rate-based drift detection algorithms for detecting
gradual task boundaries from online windowed accuracy.

Manipulated Variables:
 * $\mathcal{D}$ drift detector (4):
    * ADWIN   (Adaptive Windowing)
    * DDM     (Drift Detection Method)
    * PH      (Page-Hinkley)
    * ORACLE  (Task boundary oracle)
 * $\mathcal{L}$ strategy (9):
    * use drift detector (5):
        * EWC     (Elastic Weight Consolidation)
            * On detection update anchor model.
        * SI      (Synaptic Intelligence)
            * On detection update anchor model.
        * LWF     (Learning without Forgetting)
            * On detection update teacher model.
        * DER     (dark experience replay)
            * On detection update teacher model.
        * PN      (PackNet)
            * On detection perform pruning.
    * drift detector free (3)
        * RAR     (repeated augmented rehearsal)
        * ER      (experience replay)
        * FT      (fine-tuning)
 * $\mathcal{B}$ gradual boundary width (3):
    * abrupt  (abrupt change, 0.0 of task width)
    * gradual (gradual change, 0.5 of task width)
    * slow    (slow change, 1.0 of task width)

Dependent Variables:
* drift detection evaluation metrics
* continual learning metrics 

Nuisance Variables (Hyper-parameter Selection Phase):
1. Tune (30 hp trials) strategies with oracle drift detector (perfect task boundaries)
2. Then, the drift detectors are hyper-parameter tuned (30 trials) based on those
   baselines.

Once the drift detectors and the strategies hyper-parameters are selected we perform 5
evaluation runs with different seeds.

Experimental limitations:

*   The Hyper-parameter selection phase does not consider the interaction between
    drift detection and continual learning strategy.
*   We only investigate one specific dataset due to the curse of dimensionality on our
    experimental design. Adding another dataset would double the number of experiments.
    We prioritize hyperparameter search quality, multi-seed evaluation, and multiple
    boundary width experiments over multi-dataset benchmarking. 

Trials:
* HP Search:
    *   Strategy Tuning: This phase is performed using an oracle drift detector, which
        provides perfect task boundaries. This allows us to decrease the number of runs
        needed

        $\mathcal{L} \times \mathcal{B} \times \text{trials}$

        $9 \times 3 \times 30 = 810$

    *   Drift Detector Tuning: This phase is performed using recorded error rates from
        the strategy tuning phase. Consequently, this phase is efficient despite the
        large number of trials.

        $\mathcal{L} \times \mathcal{B} \times \mathcal{D} \times \text{trials}$

        $9 \times 3 \times 4 \times 30 = 4050$

*   Evaluation:

    $\mathcal{L} \times \mathcal{B} \times \mathcal{D} \times \text{trials}$

    $9 \times 3 \times 4 \times 5 = 675$
