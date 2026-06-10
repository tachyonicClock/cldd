
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
	rm -r logs/* || true
	rm -r logs || true
	mkdir -p /local/scratch/antonlee/log/blurry-ocl
	ln -s /local/scratch/antonlee/log/blurry-ocl logs

rsync-cuda9: clean-logs
	rsync -aP cuda9:/local/scratch/antonlee/log/blurry-ocl/ logs/


archive_filename:=$(shell date -u +"%Y-%m-%dT%H-%M-%SZ")_$(shell git rev-parse --short HEAD).7z

archive:
	7z a /local/scratch/antonlee/archive/bocl/$(archive_filename)  logs/

DOWNLOAD_ARCHIVE:=2026-06-08T20-46-35Z_4fc8395.7z
download_archive:
	rsync -P lagerfield.ecs.vuw.ac.nz:/local/scratch/antonlee/archive/bocl/$(DOWNLOAD_ARCHIVE) logs/.
# 	Extract
	7z x logs/$(DOWNLOAD_ARCHIVE) logs


.PHONY: run
run:
	nohup notirun.sh ./mpsdodo.py -n 8 > logs/nohup.log 2>&1 &

.PHONY: test
test: clean-logs
	DEBUG_BOCL=True uv run doit run

stop:
	pkill -f -SIGINT antonlee-mpsdodo-server