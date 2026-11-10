.PHONY: install test lint format demo collect train eval-expert clean
PY ?= python3
export MUJOCO_GL ?= egl

install:            ## editable install with training + dev extras
	$(PY) -m pip install -e ".[train,dev]"

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

collect:            ## 200 expert demos -> data/mj_demos
	$(PY) -m vla_mujoco.data.collect --out data/mj_demos --episodes 200 --workers 4

train:
	$(PY) -m vla_mujoco.vla.train --data data/mj_demos --arch gru --steps 20000 --out runs/gru

eval-expert:
	$(PY) -m vla_mujoco.vla.evaluate --expert --episodes 50

clean:
	rm -rf runs demo.gif demo_last.png .pytest_cache .ruff_cache
