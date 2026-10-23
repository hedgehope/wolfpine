/*
 * nbody_scalar: the direct all-pairs loop on the F/D unit, in acc_fmt
 * (float for fp32, double for fp64): r^2 from the coordinate differences, so
 * there is no expansion and no cancellation beyond the subtraction itself.
 * Same pair law, same diagonal skip and same unit masses as the AME variants;
 * every ordered pair is visited (no Newton's-third-law halving, matching the
 * GEMM's all-pairs work). With NB_SAVE_D2 it stores its r^2 matrix too.
 */
#include "../nbody_io.h"

#define N NB_N

static real P[N][3];

int main(void) {
    uint64_t start = naiad_cycles();
    for (int i = 0; i < N; i++) {
        P[i][0] = in_to_real(pos[3 * i]);
        P[i][1] = in_to_real(pos[3 * i + 1]);
        P[i][2] = in_to_real(pos[3 * i + 2]);
    }
    for (int i = 0; i < N; i++) {
        const real xi = P[i][0], yi = P[i][1], zi = P[i][2];
        real fx = 0, fy = 0, fz = 0;
        for (int j = 0; j < N; j++) {
            real dx = P[j][0] - xi, dy = P[j][1] - yi, dz = P[j][2] - zi;
            real r2 = dx * dx + dy * dy + dz * dz;
#if NB_SAVE_D2
            d2[(long)i * N + j] = r2;
#endif
            if (j == i) {
                continue;
            }
            real w = pair_weight(r2);
            fx += w * dx;
            fy += w * dy;
            fz += w * dz;
        }
        force[3 * i] = fx;
        force[3 * i + 1] = fy;
        force[3 * i + 2] = fz;
    }
    region_cycles = naiad_cycles() - start;
    return 0;
}
