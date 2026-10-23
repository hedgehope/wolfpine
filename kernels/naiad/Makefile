# Build a freestanding Naiad kernel against a design.
#
#   make                       # hello, against framework/naiad/designs/naiad-base.yaml
#   make PROG=hello DESIGN=path/to/design.yaml
#   make run                   # build, then run it with python -m framework.naiad
#
# The design's naiad.h, linker.ld, crt0.S and naiad.mk are generated into
# $(BUILD)/gen by python -m framework.naiad.gen and regenerated when the design
# changes. A kernel is every .c file in $(PROG_DIR) (default: $(PROG)/), plus the
# platform's libc shim.

NAIAD_CC ?= /opt/llvm/20.1.7/bin/clang
PYTHON ?= python3

KERNELS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(KERNELS_DIR)/../..)
PLATFORM := $(REPO_ROOT)/framework/naiad/platform
DESIGN ?= $(REPO_ROOT)/framework/naiad/designs/naiad-base.yaml
PROG ?= hello
PROG_DIR ?= $(KERNELS_DIR)/$(PROG)

DESIGN_PATH := $(abspath $(DESIGN))
BUILD ?= $(KERNELS_DIR)/build/$(basename $(notdir $(DESIGN_PATH)))
GEN := $(BUILD)/gen
ELF := $(BUILD)/$(PROG).elf

GEN_SOURCES := $(wildcard $(REPO_ROOT)/framework/naiad/*.py $(REPO_ROOT)/framework/naiad/gen/*.py $(REPO_ROOT)/framework/naiad/ip/*.py)
NAIAD_PY := PYTHONPATH=$(REPO_ROOT)$${PYTHONPATH:+:$$PYTHONPATH} $(PYTHON)

.PHONY: all run clean
all: $(ELF)

# naiad.mk is the one generated file make tracks; the others are written alongside it.
include $(GEN)/naiad.mk

$(GEN)/naiad.mk: $(DESIGN_PATH) $(GEN_SOURCES)
	$(NAIAD_PY) -m framework.naiad.gen $(DESIGN_PATH) -o $(GEN) > /dev/null

$(GEN)/naiad.h $(GEN)/linker.ld $(GEN)/crt0.S: $(GEN)/naiad.mk ;

ARCH_FLAGS := --target=riscv$(NAIAD_XLEN)-unknown-elf -march=$(NAIAD_MARCH) -mabi=$(NAIAD_MABI) -mcmodel=$(NAIAD_CMODEL)
CFLAGS ?= -O2 -g
ALL_CFLAGS := $(ARCH_FLAGS) $(CFLAGS) -ffreestanding -fno-builtin -nostdlib -ffunction-sections -fdata-sections \
              -Wall -Wextra -I$(GEN) -I$(PLATFORM) -I$(PROG_DIR)
LDFLAGS := $(ARCH_FLAGS) -fuse-ld=lld -nostdlib -static -T $(GEN)/linker.ld -Wl,--gc-sections

SOURCES := $(wildcard $(PROG_DIR)/*.c) $(PLATFORM)/naiad_libc.c
OBJECTS := $(patsubst %.c,$(BUILD)/obj/%.o,$(notdir $(SOURCES))) $(BUILD)/obj/crt0.o

vpath %.c $(PROG_DIR) $(PLATFORM)

$(BUILD)/obj/%.o: %.c $(GEN)/naiad.h $(PLATFORM)/naiad_platform.h
	@mkdir -p $(dir $@)
	$(NAIAD_CC) $(ALL_CFLAGS) -c $< -o $@

$(BUILD)/obj/crt0.o: $(GEN)/crt0.S
	@mkdir -p $(dir $@)
	$(NAIAD_CC) $(ARCH_FLAGS) -c $< -o $@

$(ELF): $(OBJECTS) $(GEN)/linker.ld
	$(NAIAD_CC) $(LDFLAGS) $(OBJECTS) -o $@

run: $(ELF)
	$(NAIAD_PY) -m framework.naiad $(DESIGN_PATH) $(ELF)

clean:
	rm -rf $(BUILD)
