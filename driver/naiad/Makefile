# Build and run the Naiad driver examples.
#
#   make                  build every example
#   make run              build and run every example (stops at the first failure)
#   make ex3              build one example
#   make run-ex3          build one example and run it
#   make clean
#
# Each example is a directory of C sources built against its design by
# kernels/naiad/Makefile, which generates the design's naiad.h, linker.ld,
# crt0.S and naiad.mk (python -m framework.naiad.gen) and takes the -march /
# -mabi from naiad.mk, so no compiler flag is spelled out here. ELFs land in
# $(BUILD)/<ex>/<ex>.elf.

NAIAD_CC ?= /opt/llvm/20.1.7/bin/clang
PYTHON ?= python3

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(HERE)/../..)
DESIGNS := $(REPO_ROOT)/framework/naiad/designs
BUILD ?= $(HERE)/build

EXAMPLES := ex1 ex2 ex3 ex4 ex5 ex6

# The design each example is built for and run on.
ex1_DESIGN := $(DESIGNS)/naiad-base.yaml
ex2_DESIGN := $(DESIGNS)/naiad-base-rv64.yaml
ex3_DESIGN := $(DESIGNS)/naiad-dma.yaml
ex4_DESIGN := $(DESIGNS)/naiad-counters.yaml
ex5_DESIGN := $(DESIGNS)/naiad-ame-timed.yaml
ex6_DESIGN := $(HERE)/ex6/naiad-quad.yaml

NAIAD_PY := PYTHONPATH=$(REPO_ROOT)$${PYTHONPATH:+:$$PYTHONPATH} $(PYTHON)

.PHONY: all run clean $(EXAMPLES) $(addprefix run-,$(EXAMPLES))
all: $(EXAMPLES)

$(EXAMPLES):
	$(MAKE) -C $(REPO_ROOT)/kernels/naiad PROG=$@ PROG_DIR=$(HERE)/$@ DESIGN=$($@_DESIGN) \
	        BUILD=$(BUILD)/$@ NAIAD_CC=$(NAIAD_CC) PYTHON=$(PYTHON)

$(addprefix run-,$(EXAMPLES)): run-%: %
	$(NAIAD_PY) -m framework.cli run naiad --design $($*_DESIGN) $(BUILD)/$*/$*.elf

# ex5 also has a host-side driver that feeds the same ELF its inputs.
run-ex5: run-ex5-host
.PHONY: run-ex5-host
run-ex5-host: ex5
	$(NAIAD_PY) $(HERE)/ex5/ex5.py $(BUILD)/ex5/ex5.elf

run: $(addprefix run-,$(EXAMPLES))

clean:
	rm -rf $(BUILD)
