
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

archive:
	7z a $(shell date -u +"%Y-%m-%dT%H-%M-%SZ")_$(shell git rev-parse --short HEAD).7z logs/