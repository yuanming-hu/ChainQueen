#pragma once

#include "state_base.h"

struct State : public StateBase {
  State() {
    num_cells = res[0] * res[1] * res[2];
  }

  TC_FORCE_INLINE __host__ __device__ int grid_size() const {
    return num_cells;
  }

  TC_FORCE_INLINE __device__ int linearized_offset(int x, int y, int z) const {
    return res[2] * (res[1] * x + y) + z;
  }

  TC_FORCE_INLINE __device__ real *grid_node(int offset) const {
    return grid_storage + (dim + 1) * offset;
  }

  TC_FORCE_INLINE __device__ real *grid_node(int x, int y, int z) const {
    return grid_node(linearized_offset(x, y, z));
  }

  TC_FORCE_INLINE __device__ Matrix get_matrix(real *p, int part_id) const {
    return Matrix(
        p[part_id + 0 * num_particles], p[part_id + 1 * num_particles],
        p[part_id + 2 * num_particles], p[part_id + 3 * num_particles],
        p[part_id + 4 * num_particles], p[part_id + 5 * num_particles],
        p[part_id + 6 * num_particles], p[part_id + 7 * num_particles],
        p[part_id + 8 * num_particles]);
  }

  TC_FORCE_INLINE __device__ void set_matrix(real *p,
                                             int part_id,
                                             Matrix m) const {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        p[part_id + (i * 3 + j) * num_particles] = m[i][j];
      }
    }
  }

  TC_FORCE_INLINE __device__ Vector get_vector(real *p, int part_id) {
    return Vector(p[part_id], p[part_id + num_particles],
                  p[part_id + num_particles * 2]);
  }

  TC_FORCE_INLINE __device__ void set_vector(real *p, int part_id, Vector v) {
    for (int i = 0; i < dim; i++) {
      p[part_id + num_particles * i] = v[i];
    }
  }

#define TC_MPM_VECTOR(x)                                                \
  TC_FORCE_INLINE __device__ Vector get_##x(int part_id) {              \
    return get_vector(x##_storage, part_id);                            \
  }                                                                     \
  TC_FORCE_INLINE __device__ void set_##x(int part_id, Vector x) {      \
    return set_vector(x##_storage, part_id, x);                         \
  }                                                                     \
  TC_FORCE_INLINE __device__ Vector get_grad_##x(int part_id) {         \
    return get_vector(grad_##x##_storage, part_id);                     \
  }                                                                     \
  TC_FORCE_INLINE __device__ void set_grad_##x(int part_id, Vector x) { \
    return set_vector(grad_##x##_storage, part_id, x);                  \
  }

  TC_MPM_VECTOR(x);
  TC_MPM_VECTOR(v);

#define TC_MPM_MATRIX(F)                                                \
  TC_FORCE_INLINE __device__ Matrix get_##F(int part_id) {              \
    return get_matrix(F##_storage, part_id);                            \
  }                                                                     \
  TC_FORCE_INLINE __device__ void set_##F(int part_id, Matrix m) {      \
    return set_matrix(F##_storage, part_id, m);                         \
  }                                                                     \
  TC_FORCE_INLINE __device__ Matrix get_grad_##F(int part_id) {         \
    return get_matrix(grad_##F##_storage, part_id);                      \
  }                                                                     \
  TC_FORCE_INLINE __device__ void set_grad_##F(int part_id, Matrix m) { \
    return set_matrix(grad_##F##_storage, part_id, m);                   \
  }

  TC_MPM_MATRIX(F);
  TC_MPM_MATRIX(C);

  /*
  int num_particles;

  real *x_storage;
  real *v_storage;
  real *F_storage;
  real *C_storage;
  real *grid_storage;

  int res[3];
  int num_cells;

  real gravity[3];
  real dx, inv_dx;
  real dt;
  */

  State(int res[dim], int num_particles, real dx, real dt, real gravity[dim]) {
    this->num_cells = 1;
    for (int i = 0; i < dim; i++) {
      this->res[i] = res[i];
      this->num_cells *= res[i];
      this->gravity[i] = gravity[i];
    }
    this->num_particles = num_particles;
    this->dx = dx;
    this->inv_dx = 1.0f / dx;
    this->dt = dt;

    cudaMalloc(&x_storage, sizeof(real) * dim * num_particles);
    cudaMalloc(&v_storage, sizeof(real) * dim * num_particles);
    cudaMalloc(&F_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&C_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&grid_storage, sizeof(real) * (dim + 1) * num_cells);

    std::vector<real> F_initial(num_particles * dim * dim, 0);
    for (int i = 0; i < num_particles; i++) {
      F_initial[i] = 1;
      F_initial[i + num_particles * 4] = 1;
      F_initial[i + num_particles * 8] = 1;
    }
    cudaMemcpy(F_storage, F_initial.data(), sizeof(Matrix) * num_particles,
               cudaMemcpyHostToDevice);
  }

  __host__ std::vector<real> fetch_x() {
    std::vector<real> host_x(dim * num_particles);
    cudaMemcpy(host_x.data(), x_storage, sizeof(Vector) * num_particles,
               cudaMemcpyDeviceToHost);
    return host_x;
  }
};

constexpr int spline_size = 3;

using BSplineWeights = real[dim][spline_size];

template <bool with_grad = false>
struct TransferCommon {
  int base_coord[dim];
  Vector fx;
  real dx, inv_dx;
  BSplineWeights weights[1 + (int)with_grad];

  TC_FORCE_INLINE __device__ TransferCommon(const State &state, Vector x) {
    dx = state.dx;
    inv_dx = state.inv_dx;
    for (int i = 0; i < dim; i++) {
      base_coord[i] = int(x[i] * inv_dx - 0.5);
      real f = x[i] * inv_dx - (real)base_coord[i];
      static_assert(std::is_same<std::decay_t<decltype(fx[i])>, real>::value);
      fx[i] = f;
    }

    // B-Spline weights
    for (int i = 0; i < dim; ++i) {
      weights[0][i][0] = 0.5f * sqr(1.5f - fx[i]);
      weights[0][i][1] = 0.75f - sqr(fx[i] - 1);
      weights[0][i][2] = 0.5f * sqr(fx[i] - 0.5f);
    }

    if (with_grad) {
      // TODO: test
      for (int i = 0; i < dim; ++i) {
        weights[1][i][0] = inv_dx * (-1.5f + fx[i]);
        weights[1][i][1] = inv_dx * (-2 * fx[i] - 2);
        weights[1][i][2] = inv_dx * (fx[i] - 0.5f);
      }
    }
  }

  TC_FORCE_INLINE __device__ real w(int i, int j, int k) {
    return weights[0][0][i] * weights[0][1][j] * weights[0][2][k];
  }

  TC_FORCE_INLINE __device__ Vector dw(int i, int j, int k) {
    return Vector(weights[1][0][i] * weights[0][1][j] * weights[0][2][k],
                  weights[0][0][i] * weights[1][1][j] * weights[0][2][k],
                  weights[0][0][i] * weights[0][1][j] * weights[1][2][k]);
  }

  TC_FORCE_INLINE __device__ Vector dpos(int i, int j, int k) {
    return dx * (Vector(i, j, k) - fx);
  }
};
