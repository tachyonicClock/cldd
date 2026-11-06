"""Pilot candidate second-dataset set-ups: does every strategy beat FT with oracle boundaries?

For each candidate and strategy (abrupt drift, Oracle): tune lr and strategy HP with
`--trials` TPE trials on the validation split, then run the best config on 3 evaluation
seeds. Results go to $LOGDIR/cldd/pilot_<candidate>/ with their own Optuna journal.

Usage (from the repo root): uv run python -P script/pilot_dataset2.py --gpus 0,1,2 --per-gpu 3
"""

import argparse, os, queue, subprocess, sys, threading, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOGS = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
STRATEGIES = ["FT", "EWC", "LWF", "MAS", "RWalk", "SI"]
EVAL_SEEDS = [2201, 9325, 1033]  # first three evaluation seeds of dodo.py
LOG_LOCK = threading.Lock()
CANDIDATES = {
    # Run the repository's dataset config directly as a pilot candidate.
    "adult": None,
    "adult_occupation": None,
    "adult_education": None,
    "credit_limit": None,
    "credit_education": None,
    "credit_education_limit": None,
    "credit_limit_rot": None,
    "covertype_elevation": None,
}


def dataset_cfg(name: str, spec: dict | None) -> Path:
    if spec is None:
        return ROOT / "config" / "dataset" / f"{name}.yml"
    f = LOGS / "pilot_configs" / f"{name}.yml"
    f.parent.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    scenario = spec.pop("scenario")
    f.write_text(yaml.safe_dump({"dataset_label": f"pilot_{name}", "scenario": scenario, **spec}))
    return f


def main_cmd(configs: list[Path], dotlist: list[str], command: list[str]) -> list[str]:
    cmd = [sys.executable, "main.py"]
    for c in configs:
        cmd += ["-c", str(c)]
    for d in dotlist + ["disable_progress_bar=True"]:
        cmd += ["-a", d]
    return cmd + command


def job(name: str, spec: dict | None, strategy: str, trials: int, env: dict, log: Path, gpu: str):
    r = ROOT / "config"
    base = [r / "base.yml", r / f"strategy/{strategy}.yml", r / "detector/oracle.yml",
            r / "boundary/abrupt.yml", dataset_cfg(name, spec)]
    root = LOGS / f"pilot_{name}"
    best = root / "tune_strategy/abrupt" / strategy / "oracle/best_params.yml"
    journal = root / "optuna.journal.log"
    root.mkdir(parents=True, exist_ok=True)
    # Imbalanced tabular sets are tuned on balanced accuracy; accuracy tracks the label prior.
    metric = "balanced_accuracy_seen_avg" if name.startswith(("adult", "credit")) else "accuracy_seen_avg"
    if not best.exists():
        cmd = main_cmd(base + [r / "hp/strategy.yml"],
                       [f"hpsearch.n_trials={trials}", "hpsearch.n_jobs=1", "hpsearch.multivariate=True", f"hpsearch.storage={journal}",
                        f"hpsearch.metric={metric}",
                        "scenario.validation=True", f"dataset_label=pilot_{name}"],
                       ["hpsearch", "tune_strategy"])
        _run(cmd, env, log, f"{name} {strategy} tune", gpu)
    if not best.exists():
        return
    for i, seed in enumerate(EVAL_SEEDS):
        out = root / "evaluate/abrupt" / strategy / f"oracle/{i:03d}/ocl_metrics.pkl"
        if out.exists():
            continue
        cmd = main_cmd(base + [best], [f"seed={seed}", f"trial={i}", "label=evaluate",
                         f"dataset_label=pilot_{name}"], ["run"])
        _run(cmd, env, log, f"{name} {strategy} eval {i}", gpu)


def _run(cmd, env, log, what, gpu):
    t0 = time.time()
    process = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    start = f"{time.strftime('%Y-%m-%d %H:%M:%S')} START gpu={gpu} pid={process.pid} {what}"
    print(start, flush=True)
    with LOG_LOCK, open(log, "a") as fh:
        fh.write(start + "\n")
        fh.flush()
    stdout, stderr = process.communicate()
    status = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {'OK' if process.returncode == 0 else 'FAIL'} gpu={gpu} pid={process.pid} exit={process.returncode} {time.time() - t0:.0f}s {what}"
    print(status, flush=True)
    with LOG_LOCK, open(log, "a") as fh:
        fh.write(status + "\n")
        if process.returncode != 0:
            fh.write(stderr[-3000:] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="candidate names to run")
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    a = ap.parse_args()
    q = queue.Queue()
    for name, spec in CANDIDATES.items():
        if a.only and name not in a.only:
            continue
        for s in a.strategies:
            q.put((name, spec, s))
    log = LOGS / "pilot_dataset2.log"; log.parent.mkdir(parents=True, exist_ok=True)

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        while True:
            try:
                name, spec, s = q.get_nowait()
            except queue.Empty:
                return
            job(name, spec, s, a.trials, env, log, gpu)

    ts = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    for t in ts: t.start()
    for t in ts: t.join()
    print("done")


if __name__ == "__main__":
    main()
