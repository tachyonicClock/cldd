
.PHONY: setup-venv clean-venv fmt

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


clean-logs:
	rm -r logs/*
	rm -r logs
	mkdir -p /local/scratch/antonlee/log/blurry-ocl
	ln -s /local/scratch/antonlee/log/blurry-ocl logs