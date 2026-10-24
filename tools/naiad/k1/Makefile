# The K1 kernel pack: the same RVV-intrinsics kernels, built for Linux on the
# SpacemiT K1 (or any RV64GCV Linux, VLEN 256) and for Naiad / Spike.
#
# On the board (native gcc or clang, RVV 1.0 intrinsics):
#   make                       # build/linux/<kernel>
#   ./run_k1.sh > k1.out       # run them all, one combined output
#
# On the host:
#   make naiad                 # ELFs for the Naiad design (kernels/naiad/Makefile)
#   make run-naiad run-spike   # run them on Naiad / on Spike: build/naiad.out, build/spike.out
#   make CROSS=1               # cross-build the Linux binaries (needs a sysroot, below)
#   make CROSS=1 run-qemu      # ... and run them under qemu-riscv64 (RVV 1.0, VLEN 256)
#   python3 compare.py build/naiad.out k1.out
#   make pack                  # k1-pack.tar.gz: everything the board needs
#
# Optional, board only: `make xsmt` builds xsmt/xsmt_gemm.c, an int8 GEMM on
# SpacemiT's IME instructions (smt.vmadot); it needs GCC >= 16.1 or LLVM >= 22.1.
# Optional: `make rvv_check` builds kernels/naiad/rvv_check (C1's self-checking RVV
# test) for Linux; it needs Zvfh, Zvfbfmin and Zvfbfwma (RVV_CHECK_MARCH).
#
# Timing: naiad_cycles() is CLOCK_MONOTONIC_RAW ns by default; TIMER=perf uses a
# perf-event cycle counter, TIMER=rdcycle the CSR (needs kernel.perf_user_access=2).

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(HERE)/../../..)
# In the unpacked pack the platform headers sit in ./platform; in the repo, in framework/.
PLATFORM ?= $(firstword $(wildcard $(HERE)/platform) $(REPO_ROOT)/framework/naiad/platform)
KERNELS := axpy dot gemm reduce gather
EXTRA_KERNEL_DIRS ?=
OUT := $(HERE)/build

MARCH ?= rv64gcv
MABI ?= lp64d
OPT ?= -O2
TIMER ?= clock
TIMER_FLAGS_clock :=
TIMER_FLAGS_perf := -DNAIAD_LINUX_PERF=1
TIMER_FLAGS_rdcycle := -DNAIAD_LINUX_RDCYCLE=1

# Cross-compiling: clang plus a riscv64 Linux sysroot. Ubuntu's cross packages
# (libc6-dev-riscv64-cross, linux-libc-dev-riscv64-cross, libgcc-12-dev-riscv64-cross)
# unpacked with dpkg-deb -x into $(XROOT) give one without root.
CROSS ?= 0
XROOT ?= $(HOME)/.local/share/riscv64-xroot
CLANG ?= /opt/llvm/20.1.7/bin/clang
ifeq ($(CROSS),1)
CC := $(CLANG) --target=riscv64-linux-gnu --sysroot=$(XROOT)/usr/riscv64-linux-gnu --gcc-toolchain=$(XROOT)/usr -fuse-ld=lld -static
else
CC ?= cc
endif
CFLAGS_LINUX := -march=$(MARCH) -mabi=$(MABI) $(OPT) -Wall -Wextra -DNAIAD_PLATFORM_LINUX $(TIMER_FLAGS_$(TIMER)) \
                -I$(PLATFORM) -I$(HERE)/kernels/include

QEMU ?= $(HOME)/.local/share/qemu-bookworm-deb/x/usr/bin/qemu-riscv64-static
QEMU_CPU ?= rv64,v=true,vlen=256,elen=64,vext_spec=v1.0
PYTHON ?= python3
DESIGN ?= $(REPO_ROOT)/framework/naiad/designs/naiad-rvv.yaml

.PHONY: all linux naiad xsmt rvv_check pack clean run-naiad run-spike run-qemu
all: linux

linux: $(KERNELS:%=$(OUT)/linux/%)

.SECONDEXPANSION:
$(OUT)/linux/%: $$(wildcard $(HERE)/kernels/$$*/*.c) $(HERE)/kernels/include/k1common.h $(PLATFORM)/naiad_platform.h $(PLATFORM)/naiad_platform_linux.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_LINUX) $(filter %.c,$^) -o $@

# Naiad ELFs, through the kernels' own Makefile (generated crt0 / linker script / naiad.h).
naiad:
	@for k in $(KERNELS); do \
	  $(MAKE) --no-print-directory -C $(REPO_ROOT)/kernels/naiad PROG=$$k PROG_DIR=$(HERE)/kernels/$$k \
	    DESIGN=$(DESIGN) BUILD=$(OUT)/naiad CFLAGS="$(OPT) -I$(HERE)/kernels/include" PYTHON=$(PYTHON) || exit 1; \
	done

run-naiad: naiad
	$(PYTHON) $(HERE)/run_sim.py --sim naiad --design $(DESIGN) $(KERNELS:%=$(OUT)/naiad/%.elf) > $(OUT)/naiad.out

run-spike: naiad
	$(PYTHON) $(HERE)/run_sim.py --sim spike --design $(DESIGN) $(KERNELS:%=$(OUT)/naiad/%.elf) > $(OUT)/spike.out

run-qemu: linux
	@: > $(OUT)/qemu.out
	@for k in $(KERNELS); do $(QEMU) -cpu $(QEMU_CPU) $(OUT)/linux/$$k >> $(OUT)/qemu.out || echo "# FAILED $$k exit $$?" >> $(OUT)/qemu.out; done

# Optional: SpacemiT IME (XSMTVDot) int8 GEMM. Native on the board only.
# XSMT_EMULATE=1 replaces the instruction by its documented semantics in C, which
# needs no IME toolchain and runs anywhere (it tests the harness, not the hardware).
XSMT_EMULATE ?= 0
ifeq ($(XSMT_EMULATE),1)
XSMT_MARCH ?= $(MARCH)
XSMT_FLAGS := -DXSMT_EMULATE=1
else
XSMT_MARCH ?= rv64gcv_xsmtvdot
XSMT_FLAGS :=
endif
xsmt: $(OUT)/linux/xsmt_gemm
$(OUT)/linux/xsmt_gemm: $(HERE)/xsmt/xsmt_gemm.c $(HERE)/kernels/include/k1common.h FORCE
	@mkdir -p $(dir $@)
	$(CC) $(subst -march=$(MARCH),-march=$(XSMT_MARCH),$(CFLAGS_LINUX)) $(XSMT_FLAGS) $< -o $@
FORCE:

# Optional: the RVV unit's own self-checking test, built for Linux.
RVV_CHECK_DIR ?= $(firstword $(wildcard $(HERE)/kernels_extra/rvv_check) $(REPO_ROOT)/kernels/naiad/rvv_check)
RVV_CHECK_MARCH ?= rv64gcv_zvfh_zvfbfmin_zvfbfwma
rvv_check: $(OUT)/linux/rvv_check
$(OUT)/linux/rvv_check: $(wildcard $(RVV_CHECK_DIR)/*.c) FORCE
	@mkdir -p $(dir $@)
	$(CC) $(subst -march=$(MARCH),-march=$(RVV_CHECK_MARCH),$(CFLAGS_LINUX)) $(filter %.c,$^) -o $@

pack:
	rm -rf $(OUT)/pack && mkdir -p $(OUT)/pack/k1/platform $(OUT)/pack/k1/kernels_extra
	cp -r $(HERE)/Makefile $(HERE)/run_k1.sh $(HERE)/compare.py $(HERE)/README.md $(HERE)/kernels $(HERE)/xsmt $(OUT)/pack/k1/
	cp $(PLATFORM)/naiad_platform.h $(PLATFORM)/naiad_platform_linux.h $(OUT)/pack/k1/platform/
	[ ! -d $(REPO_ROOT)/kernels/naiad/rvv_check ] || cp -r $(REPO_ROOT)/kernels/naiad/rvv_check $(OUT)/pack/k1/kernels_extra/
	tar -C $(OUT)/pack -czf $(HERE)/k1-pack.tar.gz k1
	@echo "wrote $(HERE)/k1-pack.tar.gz"

clean:
	rm -rf $(OUT) $(HERE)/k1-pack.tar.gz
