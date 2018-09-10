#pragma once
#include <vector>

void saxpy_cuda(int n, float alpha, float *x, float *y);
void test_svd_cuda(int n, float *, float *, float *, float *);
void initialize_mpm3d_state(int *, int, float *, void *&, float dx, float dt, float *initial_positions);

template <int dim>
void forward_mpm_state(void *, void *);

template <int dim>
void backward_mpm_state(void *, void *);

void set_grad_loss(void *);
