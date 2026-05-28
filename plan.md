# Experiment 1: Drift Detection

Our goal is to evaluate error-rate-based drift detection algorithms for detecting
gradual task boundaries.

Manipulated Variables:
 * drift detector:
    * ADWIN   (Adaptive Windowing)
    * DDM     (Drift Detection Method)
    * CUSUM   (Page-Hinkley)
    * oracle  (Task boundary oracle)
 * strategy:
    * use boundaries:
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
    * do NOT use boundaries
        * RAR     (repeated augmented rehearsal)
        * ER      (experience replay)
        * FT      (fine-tuning)
 * gradual boundary width:
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

*   The Hyper-parameter selection phase does not consider the interaction between drift
    detection and continual learning strategy. When tuning a strategy using the oracle
    it finds hyper-parameter suited for this ideal case. All experiments operate under these conditions so a fair comparison is possible.
    But it does raise the question: are methods sensitive to delays in identifying task boundaries?
    We investigate this in the "Delay Robustness" experiments by artificially delaying the task boundary oracle. 
    A related question is: if we tune detector and method jointly can we perform better?
    We investigate the 

*   We only investigate one specific dataset due to the curse of dimensionality on our
    experimental design. Adding another dataset would double the number of experiments.
    We prioritize hyperparameter search quality, multi-seed evaluation, and multiple
    boundary width experiments over multi-dataset benchmarking. 

Budget:
```python
# Experiment 1: Drift Detector
# Strategy Tuning:  phase is performed using an oracle drift detector, which provides
# perfect task boundaries. This two phase approach simplifies hyper-parameter tuning.
>>> n_strategies_a = 3 # Do NOT use task boundaries
>>> n_strategies_b = 5 # Does use task boundaries
>>> n_boundaries = 3
>>> n_hp_trials  = 30
>>> n_seeds      = 5
>>> (n_strategies_a + n_strategies_b) * n_boundaries * n_hp_trials # HP Strategy Tuning
720

# Drift Detector Tuning: This phase is performed using recorded error rates from the
# strategy tuning phase. Consequently, this phase is efficient despite the large number
# of trials.
>>> n_drift_detectors = 3 # All detectors except ORACLE
>>> (n_strategies_a + n_strategies_b) * n_boundaries * n_hp_trials * n_drift_detectors
2160

# Evaluation Phase: This phase uses the best configuration from the previous two phases
# and investigates the interaction between detector and strategy.

# We do NOT need to run experiments for different drift detectors if it has no effect
# on the method. We can evaluate the detectors using recorded error rates.
>>> n_seeds = 5
>>> n_strategies_a * n_boundaries * n_seeds
45

# We do run experiments for each drift detector.
>>> n_strategies_b * n_drift_detectors * n_boundaries * n_seeds
225

```

## Experiment 2: Joint Tuning

Does joint hyper-parameter tuning of a strategy and drift detector improve performance?

Manipulated Variables:
* Strategy (all approaches using drift detectors)
    * EWC     (Elastic Weight Consolidation)
    * SI      (Synaptic Intelligence)
    * LWF     (Learning without Forgetting)
    * DER     (dark experience replay)
    * PN      (PackNet)

Dependent Variables:
* drift detection evaluation metrics
* continual learning metrics

Fixed Variables:
* ADWIN drift detector
* gradual boundary width: gradual

Nuisance Variables (Hyper-parameter Selection Phase):
* Tune strategy and detector jointly (30 trials)

Budget:
```python
# Experiment 2: Joint Training
>>> n_strategies = 5
>>> n_hp_trials = 30
>>> n_seeds = 5
>>> n_strategies * n_hp_trials # HP   Trials
150
>>> n_strategies * n_seeds     # Eval Trials
25

```

Limitations:
Our empirical findings are limited to ADWIN's behaviour on this problem. Other algorithms may behave differently. We did not investigate other setups to constrain the experimental scope. We suspect, however, given the algorithmic similarity of drift detectors, these findings may generalise.

## Experiment 3: Delay Robustness

How robust are our investigated strategies to delays in detecting task boundaries?

Manipulated Variables:
* Task boundary oracle delay
* Strategy (all approaches using drift detectors)
    * EWC     (Elastic Weight Consolidation)
    * SI      (Synaptic Intelligence)
    * LWF     (Learning without Forgetting)
    * DER     (dark experience replay)
    * PN      (PackNet)

Dependent Variables:
* drift detection evaluation metrics
* continual learning metrics

Fixed Variables:
* gradual boundary width: gradual

Nuisance Variables (Hyper-parameter Selection Phase):
* Constants set by the hyper-parameter search phase in the "Drift Detection" experiment.

Budget:
```python
# Experiment 3: Delay Robustness
>>> n_delays     = 10
>>> n_strategies = 5
>>> n_seeds      = 5
>>> n_strategies * n_seeds * n_delays # Eval Trials
250

```

# Technical

```
# Config Structure
/base/strategy/EWC,SI,LWF,DER,PN,RAR,ER,FT
/base/boundary/abrupt,gradual,slow
/base/scenario/DomainCIFAR100
/base/drift_detector/ADWIN,DDM,PH,oracle
```

```
# Log Directory Structure
# Strategy Tuning
/logs/exp_01_hp/$SCENARIO_$BOUNDARY/$STRATEGY_oracle/$TRIAL
/logs/exp_01_eval/$SCENARIO_$BOUNDARY/$STRATEGY_$DRIFT_DETECTOR/$TRIAL
/logs/exp_02_hp/$SCENARIO_$BOUNDARY/$STRATEGY_$DRIFT_DETECTOR/$TRIAL
/logs/exp_02_eval/$SCENARIO_$BOUNDARY/$STRATEGY_$DRIFT_DETECTOR/$TRIAL
/logs/exp_03_eval/$SCENARIO_$BOUNDARY/$STRATEGY_$DRIFT_DETECTOR/$TRIAL
```


```bash
# 1. Hyperparameter Search Phase
$ uv run main.py config/00_abrupt/EWC.yml hpsearch hp
# creates: optuna study named 'bocl/hp/RotatedTinyMNIST_0.0/EWC_oracle_MLP'
# creates: log files in 'logs/hp/RotatedTinyMNIST_0.0/EWC_oracle_MLP/*/dd_metrics.pkl'

# 2. Update config files with best hyper-parameters
$ uv run update_hp.py $STUDY_NAME
# updates: config/00_abrupt/EWC.yml with hyper-parameters

# 3. Use the best run to tune a drift detector.
# consumes: log files and study
$ uv run tune_dd.py $STUDY_NAME config/base/drift_detector/ADWIN.yml 
# creates: optuna study named 'bocl/offline_dd/RotatedTinyMNIST_0.0/EWC_ADWIN_MLP
# updates: logs/tune_dd/

# 4. Update config files with best hyper-parameters
$ uv run update_hp.py $DD_STUDY


# 5. Evaluate Model
$ uv run main.pu $CONFIG -a seed=$SEED run

# Best hyper-parameters using oracle drift detector go here.
config/00_abrupt/$LEARNER.yml
```