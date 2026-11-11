.PHONY: install hooks test lint format gpu-check demo collect train eval-expert clean
PY ?= python3
export MUJOCO_GL ?= egl

install:            ## editable install with training + dev extras, plus git hooks
	$(PY) -m pip install -e ".[train,dev]"
	PYTHON=$(PY) bash .devcontainer/install-hooks.sh

gpu-check:          ## CUDA + GPU (EGL) rendering sanity check
	$(PY) -m vla_mujoco.gpu_check

hooks:              ## git pre-commit hooks (large-file / artifact guard, ruff) + Git LFS for meshes
	PYTHON=$(PY) bash .devcontainer/install-hooks.sh

test:               ## fast simulator tests (no GPU, no rendering)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check vla_mujoco tests

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format vla_mujoco tests

demo:               ## render one expert episode to demo.gif
	$(PY) -m vla_mujoco.viz --seed 3 --out demo.gif

collect:            ## 300 mixed-task expert demos (crowded scenes) -> data/mj_demos
	$(PY) -m vla_mujoco.data.collect --out data/mj_demos --episodes 300 --workers 4 --img-size 64 --task random --distractors 2 --table-distractors 1

train:              ## belief (GRU) + pretrained text encoder + flow-matching head
	$(PY) -m vla_mujoco.vla.train --data data/mj_demos --arch gru --head flow --text-encoder pretrained --steps 20000 --amp --out runs/gru_flow

eval-expert:
	$(PY) -m vla_mujoco.vla.evaluate --expert --episodes 50

clean:
	rm -rf runs demo.gif demo_last.png .pytest_cache .ruff_cache
