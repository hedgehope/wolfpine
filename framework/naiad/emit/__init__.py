"""Code emitters for Naiad kernels: C headers specialised to a design's matrix unit.

:mod:`framework.naiad.emit.gemm` writes the GEMM micro-kernel and its blocked
driver (``naiad_gemm.h``). ``kernels/naiad/Makefile`` runs it into each build's
generated directory, next to ``naiad.h``, so a design change only needs a rebuild.
"""
