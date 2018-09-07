#pragma once

#include "state_base.h"

static constexpr int mpm_enalbe_apic = false;
static constexpr int mpm_enalbe_force = true;
static constexpr int particle_block_dim = 128;
static constexpr int grid_block_dim = 128;

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

  TC_FORCE_INLINE __device__ real *grad_grid_node(int offset) const {
    return grad_grid_storage + (dim + 1) * offset;
  }

  TC_FORCE_INLINE __device__ real *grid_node(int x, int y, int z) const {
    return grid_node(linearized_offset(x, y, z));
  }

  TC_FORCE_INLINE __device__ real *grad_grid_node(int x, int y, int z) const {
    return grad_grid_node(linearized_offset(x, y, z));
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

  TC_FORCE_INLINE __device__ Vector get_grid_velocity(int i, int j, int k) {
    auto g = grid_node(i, j, k);
    return Vector(g);
  }

  TC_FORCE_INLINE __device__ Vector get_grad_grid_velocity(int i,
                                                           int j,
                                                           int k) {
    auto g = grad_grid_node(i, j, k);
    return Vector(g);
  }

  TC_FORCE_INLINE __device__ void set_grid_velocity(int i,
                                                    int j,
                                                    int k,
                                                    Vector v) {
    auto g = grid_node(i, j, k);
    for (int d = 0; d < dim; d++) {
      g[d] = v[d];
    }
  }

  TC_FORCE_INLINE __device__ void set_grad_grid_velocity(int i,
                                                         int j,
                                                         int k,
                                                         Vector v) {
    auto g = grad_grid_node(i, j, k);
    for (int d = 0; d < dim; d++) {
      g[d] = v[d];
    }
  }

  TC_FORCE_INLINE __device__ real get_grid_mass(int i, int j, int k) {
    auto g = grid_node(i, j, k);
    return g[dim];
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
    return get_matrix(grad_##F##_storage, part_id);                     \
  }                                                                     \
  TC_FORCE_INLINE __device__ void set_grad_##F(int part_id, Matrix m) { \
    return set_matrix(grad_##F##_storage, part_id, m);                  \
  }

  TC_MPM_MATRIX(F);
  TC_MPM_MATRIX(P);
  TC_MPM_MATRIX(C);

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
    this->invD = 4 * inv_dx * inv_dx;

    cudaMalloc(&x_storage, sizeof(real) * dim * num_particles);
    cudaMalloc(&v_storage, sizeof(real) * dim * num_particles);
    cudaMalloc(&F_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&P_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&C_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&grid_storage, sizeof(real) * (dim + 1) * num_cells);

    cudaMalloc(&grad_x_storage, sizeof(real) * dim * num_particles);
    cudaMalloc(&grad_v_storage, sizeof(real) * dim * num_particles);
    cudaMalloc(&grad_F_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&grad_P_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&grad_C_storage, sizeof(real) * dim * dim * num_particles);
    cudaMalloc(&grad_grid_storage, sizeof(real) * (dim + 1) * num_cells);

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

  __host__ std::vector<real> fetch_grad_v() {
    std::vector<real> host_grad_v(dim * num_particles);
    cudaMemcpy(host_grad_v.data(), grad_v_storage,
               sizeof(Vector) * num_particles, cudaMemcpyDeviceToHost);
    return host_grad_v;
  }

  __host__ std::vector<real> fetch_grad_x() {
    std::vector<real> host_grad_x(dim * num_particles);
    cudaMemcpy(host_grad_x.data(), grad_x_storage,
               sizeof(Vector) * num_particles, cudaMemcpyDeviceToHost);
    return host_grad_x;
  }

  void clear_gradients() {
    cudaMemset(grad_v_storage, 0, sizeof(real) * dim * num_particles);
    cudaMemset(grad_x_storage, 0, sizeof(real) * dim * num_particles);
    cudaMemset(grad_F_storage, 0, sizeof(real) * dim * dim * num_particles);
    cudaMemset(grad_P_storage, 0, sizeof(real) * dim * dim * num_particles);
    cudaMemset(grad_C_storage, 0, sizeof(real) * dim * dim * num_particles);
    cudaMemset(grad_grid_storage, 0, num_cells * (dim + 1) * sizeof(real));
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
      real f = (real)base_coord[i] - x[i] * inv_dx;
      static_assert(std::is_same<std::decay_t<decltype(fx[i])>, real>::value);
      fx[i] = f;
    }

    // B-Spline weights
    for (int i = 0; i < dim; ++i) {
      weights[0][i][0] = 0.5f * sqr(1.5f + fx[i]);
      weights[0][i][1] = 0.75f - sqr(fx[i] + 1);
      weights[0][i][2] = 0.5f * sqr(fx[i] + 0.5f);
      //       printf("%f\n", weights[0][i][0] + weights[0][i][1] +
      //       weights[0][i][2]);
    }

    if (with_grad) {
      // N(x_i - x_p)
      // TODO: test
      for (int i = 0; i < dim; ++i) {
        weights[1][i][0] = -inv_dx * (1.5f + fx[i]);
        weights[1][i][1] = inv_dx * (2 * fx[i] + 2);
        weights[1][i][2] = -inv_dx * (fx[i] + 0.5f);
        // printf("%f\n", weights[1][i][0] + weights[1][i][1] + weights[1][i][2]);
      }
    }
  }

  TC_FORCE_INLINE __device__ real w(int i, int j, int k) {
    return weights[0][0][i] * weights[0][1][j] * weights[0][2][k];
  }

  template <bool _with_grad = with_grad>
  TC_FORCE_INLINE __device__ std::enable_if_t<_with_grad, Vector> dw(int i,
                                                                     int j,
                                                                     int k) {
    return Vector(weights[1][0][i] * weights[0][1][j] * weights[0][2][k],
                  weights[0][0][i] * weights[1][1][j] * weights[0][2][k],
                  weights[0][0][i] * weights[0][1][j] * weights[1][2][k]);
  }

  TC_FORCE_INLINE __device__ Vector dpos(int i, int j, int k) {
    return dx * (fx + Vector(i, j, k));
  }
};

constexpr real m_p = 100;   // TODO: variable m_p
constexpr real V = 10;     // TODO: variable vol
constexpr real E = 10;     // TODO: variable E
constexpr real nu = 0.3;  // TODO: variable nu
constexpr real mu = E / (2 * (1 + nu)),
               lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
