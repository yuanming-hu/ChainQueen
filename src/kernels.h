#pragma once
#include <vector>

void saxpy_cuda(int n, float alpha, float *x, float *y);
void test_svd_cuda(int n, float *, float *, float *, float *);
void initialize_mpm3d_state(int *, int, float *, void *&, float dx, float dt, float *initial_positions);
void forward_mpm3d_state(void *, void *);
void backward_mpm3d_state(void *, void *);
void set_grad_loss(void *);
void set_initial_velocities(void *, float *);
void set_initial_F(void *state_, float *F);
std::vector<float> fetch_mpm3d_particles(void *);
std::vector<float> fetch_mpm3d_grad_v(void *);
std::vector<float> fetch_mpm3d_grad_x(void *);
