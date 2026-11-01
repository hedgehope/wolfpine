# The Naiad experiment harness (see experiments/naiad/run.py).
#
#   make -C experiments/naiad figures        # run the headline sweeps, regenerate every figure
#   make -C experiments/naiad quick          # the same at the smallest sizes (CI)
#   make -C experiments/naiad test           # the harness's own tests
#   make -C experiments/naiad clean          # remove out/

PYTHON ?= python3
JOBS ?= $(shell nproc 2>/dev/null | awk '{print int($$1/2) > 0 ? int($$1/2) : 1}')
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
RUN := cd $(REPO_ROOT) && PYTHONPATH=$(REPO_ROOT)$${PYTHONPATH:+:$$PYTHONPATH} $(PYTHON)

.PHONY: figures quick test clean
figures:
	$(RUN) -m experiments.naiad.all --jobs $(JOBS)

quick:
	$(RUN) -m experiments.naiad.all --quick --jobs $(JOBS)

test:
	$(RUN) -m pytest -q experiments/naiad/tests

clean:
	rm -rf $(REPO_ROOT)/experiments/naiad/out
