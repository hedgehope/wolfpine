/*
 * nbody_ame_fgemm: nbody_ame with the force sum on the matrix unit too.
 *
 * F_i = sum_j w_ij (x_j - x_i) = (W X)_i - x_i (W 1)_i, so after the scalar
 * side turns a D tile into weights (rounded to in_fmt, scaled by the
 * power-of-two params[4] so fp16 does not overflow), one more mma with
 * Xop[j] = [x_j, y_j, z_j, 1] accumulates both sums. The scalar work per
 * pair drops to w(r^2) alone; the price is rounding w to in_fmt and the
 * cancellation in (W X)_i - x_i (W 1)_i when the positions are far from the
 * origin.
 */
#define NB_FORCE_GEMM 1
#include "../ame/nbody_ame.c"
