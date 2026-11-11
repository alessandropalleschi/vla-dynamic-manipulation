.PHONY: install hooks test lint format gpu-check demo data data-scenes data-demos collect pretrain train eval-expert clean
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

WORKERS ?= 8
IMG ?= 128
SCENES ?= 50000
EPISODES ?= 600
DATA ?= data
# Hardware variant, e.g. the OAK-D Pro wrist camera on the flange adapter plate + the top camera beyond the belt:
#   make data DATA=data_oak CAMS="front wrist overhead" HW="--wrist-mount oak --overhead-view front"
CAMS ?= front wrist
HW ?=

data: data-scenes data-demos   ## rebuild every dataset (git-ignored; each gets a manifest.json with commit + command)

data-scenes:        ## static scenes + instructions for grounding pre-training -> $(DATA)/scenes (memory-mapped)
	$(PY) -m vla_mujoco.data.static_scenes --out $(DATA)/scenes --scenes $(SCENES) --img-size $(IMG) --workers $(WORKERS) \
		--cameras $(CAMS) $(HW)

data-demos:         ## mixed-task expert demos in crowded scenes -> $(DATA)/mj_demos (memory-mapped)
	$(PY) -m vla_mujoco.data.collect --out $(DATA)/mj_demos --episodes $(EPISODES) --workers $(WORKERS) --img-size $(IMG) \
		--task random --distractors 2 --table-distractors 1 --cameras $(CAMS) $(HW)

collect: data-demos

pretrain:           ## grounding pre-training on static scenes (16x16 patch grid at 128 px)
	$(PY) -m vla_mujoco.vla.pretrain_grounding --data data/scenes --out runs/ground_pre --text-encoder pretrained \
		--patch 8 --steps 6000 --batch 64 --amp

train:              ## belief (GRU) + pretrained text encoder + flow-matching head
	$(PY) -m vla_mujoco.vla.train --data data/mj_demos --arch gru --head flow --text-encoder pretrained --patch 8 --slots --init runs/ground_pre/best.pt --steps 20000 --amp --out runs/gru_flow

eval-expert:
	$(PY) -m vla_mujoco.vla.evaluate --expert --episodes 50

clean:
	rm -rf runs demo.gif demo_last.png .pytest_cache .ruff_cache
