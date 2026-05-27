from itertools import product

STRATEGY = [
    "FT",
    # "EWC",
]
BOUNDARY = [
    "00_abrupt",
    # "01_gradual",
    # "02_slow",
]
DETECTOR = [
    "ADWIN",
]
HP_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS = [5, 6, 7, 8, 9]


def config_file(strategy, boundary, detector=None):
    if detector is None:
        return f"config/{boundary}/{strategy}_oracle_MLP.yml"
    else:
        return f"config/{boundary}/{strategy}_{detector}_{boundary}_MLP.yml"


def study_name(strategy, boundary, detector=None):
    if detector is None:
        return f"bocl/hp/{boundary}/{strategy}_oracle_MLP"
    else:
        return f"bocl/hp/{boundary}/{strategy}_{detector}_{boundary}_MLP"


# for strategy, boundary in product(STRATEGY, BOUNDARY):
#     config = {}
#     config["bases"] = [
#         "scenario.yml",
#         f"boundary/{boundary}.yml",
#         f"learner/{strategy}.yml",
#         "drift_detector/oracle.yml",
#     ]
#     filename = Path(config_file(strategy, boundary))
#     filename.parent.mkdir(parents=True, exist_ok=True)

#     with open(filename, "w") as f:
#         omegaconf.OmegaConf.save(config, f)


"""
PHASE 1: TUNE THE STRATEGY
--------------------------
Tunes the continual learning strategy assuming a perfect drift detector (Oracle).
"""
for strategy, boundary in product(STRATEGY, BOUNDARY):
    print(f"{CMD} {config_file(strategy, boundary).ljust(50)} hpsearch hp")

print()

# Update configs
for strategy, boundary in product(STRATEGY, BOUNDARY):
    print(f"uv run update_hp.py '{study_name(strategy, boundary)}'")

print()

# Oracle Evaluation
for strategy, boundary, eval_seed in product(STRATEGY, BOUNDARY, EVAL_SEEDS):
    args = f"-a label='error-stream' -a seed={eval_seed} -a trial={eval_seed}"
    print(f"{CMD} {args} {config_file(strategy, boundary).ljust(35)} run")

"""
PHASE 2: TUNE THE DRIFT DETECTOR
--------------------------------
Tunes the detector for each strategy-boundary combination using Phase 1 error
streams.
"""
# for strategy, boundary in product(STRATEGY, BOUNDARY):
#     error_streams = f"{strategy}-{boundary}"

#     for detector in DETECTOR:
#         print(f"hp {detector} {error_streams}")

"""
PHASE 3: FINAL EVALUATION
-------------------------
Evaluates the best combination of strategy and detector using unobserved random
seeds.
"""
