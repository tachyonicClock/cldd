
.PHONY: fmt rfmnist-data


fmt:
	uvx ruff format
	uvx ruff check --fix

rfmnist-data:
	uv run python script/rfmnist.py


clean-logs:
	rm nohup.out || true
	rm -r ${LOGDIR}/cldd || true
	mkdir -p ${LOGDIR}/cldd

rsync-cuda11:
	rsync -a --info=Progress2 --delete cuda11:${LOGDIR}/cldd/ ${LOGDIR}/cldd/
	uv run doit reset-dep


archive_filename:=$(shell date -u +"%Y-%m-%dT%H-%M-%SZ")_$(shell git rev-parse --short HEAD).7z

archive:
	7z a ${ECS_SCRATCH}/archive/cldd/$(archive_filename)  ${LOGDIR}/cldd

.PHONY: run
run:
	nohup notirun.sh ./mpsdodo.py -n 8 > ${LOGDIR}/cldd/nohup.log 2>&1 &

.PHONY: test
test: clean-logs
	DEBUG_BOCL=True uv run doit run

GPU ?= 1
JOBS ?= 6

.PHONY: run-covertype
run-covertype:
	mkdir -p ${LOGDIR}/cldd/covertype
	CLDD_DATASETS=covertype CLDD_GPUS=$(GPU) OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
	nohup uv run doit run -n $(JOBS) tune_strategy tune_detector_strategy error_stream tune_hp_detector select_best evaluate dd_run collect_evaluate dd_collect \
	> ${LOGDIR}/cldd/covertype/nohup.log 2>&1 &

.PHONY: notify-covertype
# Run after `make run-covertype`; the bracket keeps pgrep from matching this shell.
notify-covertype:
	nohup sh -c 'pid=$$(pgrep -of "[d]oit run -n") || exit 1; tail --pid=$$pid -f /dev/null; \
	if [ -f ${LOGDIR}/cldd/covertype/dd_run/data_frame.csv ]; then s=finished; else s="stopped early, see covertype/nohup.log"; fi; \
	discord.sh --text "covertype pipeline $$s"' > /dev/null 2>&1 &

stop:
	pkill -f -SIGINT antonlee-mpsdodo-server

plots:
	uv run analysis/plots.py
	uv run analysis/plot_error_stream.py


fig-to-jpg:
# 	Convert all .pdf figures in fig/ to .jpg
	magick mogrify -format png -density 150 -quality 80 fig/*.pdf

.PHONY: table
table: table/detectors_covertype.tex table/detectors_rfmnist.tex table/triggers.tex

table/detectors_covertype.tex table/detectors_rfmnist.tex &: analysis/results_concise.py analysis/results_full.py analysis/results_check.py analysis/results_strategy.py analysis/common.py analysis/plots.py
	uv run analysis/results_concise.py --detectors-only

table/triggers.tex: analysis/results_triggers.py analysis/results_strategy.py analysis/common.py analysis/plots.py
	uv run analysis/results_triggers.py

.PHONY: fig
fig: fig/contour_acc.pdf fig/drift_speed.pdf fig/error_streams.pdf fig/faults.pdf fig/hm_acc.pdf fig/hm_f1.pdf fig/rfmnist.pdf fig/simulation_gap.pdf fig/strat_acc_f1.pdf

fig/contour_acc.pdf: analysis/plot_contour_acc.py analysis/plots.py analysis/common.py
	uv run analysis/plot_contour_acc.py

fig/drift_speed.pdf: analysis/plot_drift_speed.py analysis/plots.py analysis/common.py
	uv run analysis/plot_drift_speed.py

fig/error_streams.pdf: analysis/plot_error_stream.py analysis/plots.py analysis/common.py
	uv run analysis/plot_error_stream.py

fig/faults.pdf: analysis/plot_faults.py analysis/plots.py analysis/common.py
	uv run analysis/plot_faults.py

fig/hm_acc.pdf: analysis/plot_hm_acc.py analysis/plots.py analysis/common.py
	uv run analysis/plot_hm_acc.py

fig/hm_f1.pdf: analysis/plot_hm_f1.py analysis/plots.py analysis/common.py
	uv run analysis/plot_hm_f1.py

fig/rfmnist.pdf: analysis/plot_rfmnist.py analysis/plots.py analysis/common.py
	uv run analysis/plot_rfmnist.py

fig/simulation_gap.pdf: analysis/plot_simulation_gap.py analysis/plots.py analysis/common.py
	uv run analysis/plot_simulation_gap.py

fig/strat_acc_f1.pdf: analysis/plot_strat_acc_f1.py analysis/plots.py analysis/common.py
	uv run analysis/plot_strat_acc_f1.py
