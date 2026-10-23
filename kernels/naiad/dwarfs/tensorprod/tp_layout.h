/*
 * tp_layout.h: the tensorprod I/O arrays and their padded layouts, shared by the
 * ame/ and scalar/ variants and mirrored by spec.py's layout() (see its docstring).
 * Built with -DTP_P, -DTP_E, -DTP_IN / -DTP_ACC (format names).
 */
#ifndef TP_LAYOUT_H
#define TP_LAYOUT_H

#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#define P TP_P
#define E TP_E
#define T 16 /* the tile side the layouts are padded to (the AME variant checks the unit's) */
#define Q (T / P)
#define P2 (P * P)
#define P3 (P * P * P)
#define ROUNDUP(x, m) ((((x) + (m) - 1) / (m)) * (m))
#define CDIV(x, m) (((x) + (m) - 1) / (m))

#if P < 1 || P > T
#error "tensorprod needs 1 <= TP_P <= 16"
#endif

#define LDT ROUNDUP(P2, T)
#define GROUPS_T CDIV(E, Q)
#define ROWS_T (GROUPS_T * Q * P + T)
#define ROWS_R ROUNDUP(CDIV(E * P2, Q), T)
#define GROUPS_S CDIV(E * P, Q)
#define ROWS_S (GROUPS_S * Q * P + T)
#define JOBS_T (GROUPS_T * (LDT / T))
#define JOBS_R (ROWS_R / T)
#define JOBS_S GROUPS_S

/* u, plus a zeroed tail that the AME variant's padded tiles may read past the end. */
#define U_ALLOC (GROUPS_T * Q * P3 + T * P2 + ROWS_R * Q * P + 2 * T * T)

typedef NAIAD_T(TP_IN) in_t;
typedef NAIAD_T(TP_ACC) acc_t;

NAIAD_IO in_t D[P * P];
NAIAD_IO in_t u[U_ALLOC];
NAIAD_IO acc_t out_t[ROWS_T * LDT];
NAIAD_IO acc_t out_r[ROWS_R * T];
NAIAD_IO acc_t out_s[ROWS_S * T];
NAIAD_IO int64_t tp_dims[2];
NAIAD_IO uint64_t region_cycles;

#endif /* TP_LAYOUT_H */
