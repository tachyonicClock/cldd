
.PHONY: setup-venv clean-venv fmt rfmnist-data

setup-venv:
	mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(UV_ENV_DIR)
	uv venv $(UV_ENV_DIR)/blurry-ocl
	ln -s $(UV_ENV_DIR)/blurry-ocl .venv

clean-venv:
	rm -rf $(UV_ENV_DIR)/blurry-ocl
	rm -rf .venv

fmt:
	uvx ruff format
	uvx ruff check --fix
# 	Clear notebook outputs
	find . -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;

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

stop:
	pkill -f -SIGINT antonlee-mpsdodo-server

plots:
	uv run analysis/plots.py
	uv run analysis/plot_error_stream.py


fig-to-jpg:
# 	Convert all .pdf figures in fig/ to .jpg
	magick mogrify -format png -density 150 -quality 80 fig/*.pdf
	