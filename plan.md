# Experiment 1: Drift Detection

```python
import optuna
from typing import List, Dict

# ---------------------------------------------------------
# Variables & Experimental Constants
# ---------------------------------------------------------
# We hold the dataset constant to avoid a combinatorial explosion 
# and prioritize hyperparameter search, seeds, and boundaries.
CONSTANT_DATASET = "selected_dataset" 

INDEPENDENT_VARS = ["strategy", "detector", "boundary"]
DEPENDENT_VARS = ["cl_accuracy", "cl_metrics", "detector_metrics"]

# Hyper-parameter search settings used in Phases 1 and 2
OPTUNA_TRIALS = 30
SAMPLER = optuna.samplers.TPESampler()

# ---------------------------------------------------------
# Phase 1: Tune the Strategy
# ---------------------------------------------------------
def phase_one_tune_strategy(strategies, boundaries, train_tasks, seeds) -> tuple:
    """
    Tunes the continual learning strategy assuming a perfect drift detector (Oracle).
    """
    best_strategy_configs = {}
    error_rate_streams = {}

    for strategy in strategies:
        for boundary in boundaries:
            # Draw holdout validation set from training data tasks
            val_tasks = create_holdout_split(train_tasks)
            oracle_detector = "Oracle"

            # Auto-tune with Optuna (30 trials, TPE sampler)
            def objective(trial):
                config = sample_strategy_space(trial, strategy)
                return evaluate_learner(strategy, config, oracle_detector, boundary, val_tasks)
            
            study = optuna.create_study(sampler=SAMPLER)
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            best_config = study.best_params
            best_strategy_configs[(strategy, boundary)] = best_config

            # Evaluate with multiple seeds (dictates init, stream order, within-task shuffle)
            streams = []
            for seed in seeds:
                stream = test_then_train_evaluation(
                    strategy, best_config, oracle_detector, boundary, val_tasks, seed
                )
                streams.append(stream)
            
            # Record error-rate stream for Phase 2
            error_rate_streams[(strategy, boundary)] = streams

    return best_strategy_configs, error_rate_streams

# ---------------------------------------------------------
# Phase 2: Tune the Detector
# ---------------------------------------------------------
def phase_two_tune_detector(error_rate_streams, detectors) -> Dict:
    """
    Tunes the detector for each strategy-boundary combination using Phase 1 error streams.
    """
    best_detector_configs = {}

    for (strategy, boundary), streams in error_rate_streams.items():
        for detector in detectors:
            
            def objective(trial):
                config = sample_detector_space(trial, detector)
                return evaluate_detector_on_streams(detector, config, streams)
            
            study = optuna.create_study(sampler=SAMPLER)
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            
            best_detector_configs[(strategy, boundary, detector)] = study.best_params
            
    return best_detector_configs

# ---------------------------------------------------------
# Phase 3: Final Evaluation
# ---------------------------------------------------------
def phase_three_final_evaluation(best_strategy_configs, best_detector_configs, new_seeds) -> Dict:
    """
    Evaluates the best combination of strategy and detector using unobserved random seeds.
    Populates Tables: A_BOCL, G_BOCL, and S_BOCL.
    """
    final_results = {} 

    for (strategy, boundary), strat_config in best_strategy_configs.items():
        # Retrieve the optimal detector and its configuration for this pair
        best_detector, det_config = get_optimal_detector(best_detector_configs, strategy, boundary)
        
        # Evaluate using entirely new random seeds
        results = run_full_pipeline(
            strategy=strategy, 
            strat_config=strat_config, 
            detector=best_detector, 
            det_config=det_config, 
            boundary=boundary, 
            seeds=new_seeds
        )
        final_results[(strategy, boundary)] = results
        
    return final_results

# ---------------------------------------------------------
# Supplementary & Detector Evaluation
# ---------------------------------------------------------
def supplementary_joint_search():
    """
    Small-scale supplementary experiment to observe interaction between 
    detector and learner by performing hyper-parameter search over both jointly.
    """
    # TODO: Implement joint search logic (smaller scale)
    pass 

def evaluate_detectors_loocv(detectors, error_rate_streams, all_seeds) -> Dict:
    """
    Leave-one-out cross-validation to evaluate detectors.
    Prevents hyperparameter search from overfitting to specific error-rate streams.
    Populates Table: dd.
    """
    loocv_results = {}
    for detector in detectors:
        for holdout_seed in all_seeds:
            # Tune using all seeds EXCEPT the holdout seed
            train_seeds = [s for s in all_seeds if s != holdout_seed]
            train_streams = get_streams_by_seeds(error_rate_streams, train_seeds)
            best_config = tune_detector_custom(detector, train_streams)
            
            # Evaluate on the left-out seed
            holdout_stream = get_streams_by_seeds(error_rate_streams, [holdout_seed])
            score = test_detector(detector, best_config, holdout_stream)
            loocv_results[(detector, holdout_seed)] = score
            
    return loocv_results

# ---------------------------------------------------------
# Execution Flow
# ---------------------------------------------------------
if __name__ == "__main__":
    hyperparam_seeds = [42, 43, 44]
    evaluation_seeds = [99, 100, 101] # Must differ from hyperparam_seeds
    
    # 1. Tune Strategy
    strat_configs, err_streams = phase_one_tune_strategy(
        INDEPENDENT_VARS['strategy'], 
        INDEPENDENT_VARS['boundary'], 
        train_tasks, 
        hyperparam_seeds
    )
    
    # 2. Tune Detector
    det_configs = phase_two_tune_detector(err_streams, INDEPENDENT_VARS['detector'])
    
    # 3. Final Evaluation
    final_metrics = phase_three_final_evaluation(strat_configs, det_configs, evaluation_seeds)
```

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