
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
	mkdir -p ${LOGDIR}/cldd
	CLDD_DATASETS=covertype_elevation CLDD_GPUS=$(GPU) OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
	nohup uv run doit run -n $(JOBS) tune_strategy error_stream tune_hp_detector select_best evaluate dd_run collect_evaluate dd_collect \
	> ${LOGDIR}/cldd/nohup_covertype.log 2>&1 &

.PHONY: notify-covertype
# Run after `make run-covertype`; the bracket keeps pgrep from matching this shell.
notify-covertype:
	nohup sh -c 'pid=$$(pgrep -of "[d]oit run -n") || exit 1; tail --pid=$$pid -f /dev/null; \
	if [ -f ${LOGDIR}/cldd/covertype_elevation/dd_run/data_frame.csv ]; then s=finished; else s="stopped early, see nohup_covertype.log"; fi; \
	discord.sh --text "covertype pipeline $$s"' > /dev/null 2>&1 &

stop:
	pkill -f -SIGINT antonlee-mpsdodo-server

plots:
	uv run analysis/plots.py
	uv run analysis/plot_error_stream.py


fig-to-jpg:
# 	Convert all .pdf figures in fig/ to .jpg
	magick mogrify -format png -density 150 -quality 80 fig/*.pdf
	