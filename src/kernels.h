#pragma once
#include <vector>

void saxpy_cuda(int n, float alpha, float *x, float *y);
void test_svd_cuda(int n, float *, float *, float *, float *);
void initialize_mpm3d_state(void *&, float *initial_positions);
void advance_mpm3d_state(void *);
std::vector<float> fetch_mpm3d_particles(void *);
