/*
 * nbody_io.h: host-visible arrays and the pair-force law of the nbody dwarf.
 *
 *   pos[N*3]        positions (x, y, z per particle), in_fmt
 *   params[5]       acc_fmt: eps^2 (softening), sigma^2, 24*eps_LJ, r_cut^2,
 *                   and a power-of-two weight scale for the force-GEMM variant
 *   force[N*3]      output, acc_fmt: F_i = sum_{j != i} w(r_ij^2) (x_j - x_i)
 *   d2[N*N]         output when NB_SAVE_D2: the squared-distance matrix as the
 *                   kernel used it (the AME variants: the GEMM's raw output,
 *                   before the clamp at 0), so the expansion's cancellation
 *                   error can be scored apart from the force error
 *
 * Unit masses. The law: softened gravity w = (r^2 + eps^2)^(-3/2) by default;
 * -DNB_LJ=1 gives Lennard-Jones, w = -24 eps (2 (s/r)^12 - (s/r)^6) / r^2,
 * cut off at r_cut.
 */
#ifndef NBODY_IO_H
#define NBODY_IO_H

#include "naiad_platform.h"
#include "dwfmt.h"

#ifndef NB_N
#error "NB_N (particle count) must be set"
#endif
#ifndef NB_SAVE_D2
#define NB_SAVE_D2 0
#endif
#ifndef NB_LJ
#define NB_LJ 0
#endif

NAIAD_IO in_t pos[NB_N * 3];
NAIAD_IO acc_t params[5];
NAIAD_IO acc_t force[NB_N * 3];
#if NB_SAVE_D2
NAIAD_IO acc_t d2[NB_N * NB_N];
#endif
NAIAD_IO uint64_t region_cycles;

/* w(r^2) for one pair, i != j; the caller has clamped r2 at 0. */
DW_INLINE real pair_weight(real r2) {
#if NB_LJ
    if (r2 >= params[3]) {
        return 0;
    }
    real inv = (real)1 / r2, s2 = params[1] * inv, s6 = s2 * s2 * s2;
    return -params[2] * inv * s6 * (2 * s6 - 1);
#else
    real s = r2 + params[0];
    return (real)1 / (s * dw_sqrt(s));
#endif
}

#endif /* NBODY_IO_H */
