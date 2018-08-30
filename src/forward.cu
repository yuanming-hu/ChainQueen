#include "kernels.h"
#include "linalg.h"
#include "particle.h"
#include "svd.cuh"
#include <cstdio>
#include <vector>

struct State : public StateBase {
  State() {
    num_cells = res[0] * res[1] * res[2];
  }

  TC_FORCE_INLINE __host__ __device__ int grid_size() {
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

  TC_FORCE_INLINE __device__ Vector get_x(int part_id) {
    return get_vector(x_storage, part_id);
  }

  TC_FORCE_INLINE __device__ void set_x(int part_id, Vector x) {
    return set_vector(x_storage, part_id, x);
  }

  TC_FORCE_INLINE __device__ Vector get_v(int part_id) {
    return get_vector(v_storage, part_id);
  }

  TC_FORCE_INLINE __device__ void set_v(int part_id, Vector x) {
    return set_vector(v_storage, part_id, x);
  }

  TC_FORCE_INLINE __device__ Matrix get_F(int part_id) {
    return get_matrix(F_storage, part_id);
  }

  TC_FORCE_INLINE __device__ void set_F(int part_id, Matrix m) {
    return set_matrix(F_storage, part_id, m);
  }

  TC_FORCE_INLINE __device__ Matrix get_C(int part_id) {
    return get_matrix(C_storage, part_id);
  }

  TC_FORCE_INLINE __device__ void set_C(int part_id, Matrix m) {
    return set_matrix(C_storage, part_id, m);
  }

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

    std::vector<Matrix> F_initial(num_particles);
    for (int i = 0; i < num_particles; i++) {
      F_initial[i] = Matrix(1.0f);
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

__global__ void saxpy(int n, real a, real *x, real *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

void saxpy_cuda(int N, real alpha, real *x, real *y) {
  real *d_x, *d_y;

  cudaMalloc(&d_x, N * sizeof(real));
  cudaMalloc(&d_y, N * sizeof(real));

  cudaMemcpy(d_x, x, N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(real), cudaMemcpyHostToDevice);

  // this is stupidly wrong..... saxpy_g<<<1, 256>>>(n, alpha, x, y);
  saxpy<<<(N + 255) / 256, 256>>>(N, alpha, d_x, d_y);

  cudaMemcpy(y, d_y, N * sizeof(real), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
}

// Gather data from SOA
// Ensure coalesced global memory access

__device__ void svd(Matrix &A, Matrix &U, Matrix &sig, Matrix &V) {
  // clang-format off
  sig[0][1] = sig[0][2] = sig[1][0] = sig[1][2] = sig[2][0] = sig[2][1] = 0;
  svd(
      A[0][0], A[0][1], A[0][2],
      A[1][0], A[1][1], A[1][2],
      A[2][0], A[2][1], A[2][2],
      U[0][0], U[0][1], U[0][2],
      U[1][0], U[1][1], U[1][2],
      U[2][0], U[2][1], U[2][2],
      sig[0][0], sig[1][1], sig[2][2],
      V[0][0], V[0][1], V[0][2],
      V[1][0], V[1][1], V[1][2],
      V[2][0], V[2][1], V[2][2]
  );
  // clang-format on
}

__device__ void polar_decomp(Matrix &A, Matrix &R, Matrix &S) {
  Matrix U, sig, V;
  svd(A, U, sig, V);
  R = U * transposed(V);
  S = V * sig * transposed(V);
}

struct TransferCommon {
  int base_coord[dim];
  Vector fx;
  real weight[dim][spline_size];
  real dx, inv_dx;

  TC_FORCE_INLINE __device__ TransferCommon(const State &state, Vector x) {
    dx = state.dx;
    inv_dx = state.inv_dx;
    for (int p = 0; p < dim; p++) {
      base_coord[p] = int(x[p] * inv_dx - 0.5);
    }
    for (int i = 0; i < dim; i++) {
      fx[i] = x[i] * inv_dx - base_coord[i];
    }

    // B-Spline weights
    for (int i = 0; i < dim; ++i) {
      weight[i][0] = 0.5f * sqr(1.5f - fx[i]);
      weight[i][1] = 0.75f * sqr(fx[i] - 1);
      weight[i][2] = 0.5f * sqr(fx[i] - 0.5);
    }
  }

  TC_FORCE_INLINE __device__ real w(int i, int j, int k) {
    return weight[0][i] * weight[1][j] * weight[2][k];
  }

  TC_FORCE_INLINE __device__ Vector dpos(int i, int j, int k) {
    return dx * (Vector(i, j, k) - fx);
  }
};

using BSplineWeights = real[dim][spline_size];

// Do not consider sorting for now. Use atomics instead.

// One particle per thread
__global__ void P2G(State &state) {
  // constexpr int scratch_size = 8;
  //__shared__ real scratch[dim + 1][scratch_size][scratch_size][scratch_size];

  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  auto inv_dx = state.inv_dx;
  real dt = state.dt;

  Vector x = state.get_x(part_id), v = state.get_v(part_id);
  real mass = 1;    // TODO: variable mass
  real volume = 1;  // TODO: variable vol
  real E = 0000;    // TODO: variable E
  real nu = 0.3;    // TODO: variable nu
  Matrix F = state.get_F(part_id);
  Matrix C = state.get_C(part_id);

  TransferCommon tc(state, x);

  // Fixed corotated
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  real J = determinant(F);
  Matrix r, s;
  polar_decomp(F, r, s);
  Matrix stress =
      -4 * inv_dx * inv_dx * dt * volume *
      (2 * mu * (F - r) * transposed(F) + Matrix(lambda * (J - 1) * J));

  auto affine = stress + mass * C;

  Vector mv = mass * v;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        Vector dpos = tc.dpos(i, j, k);

        real contrib[dim + 1];

        auto tmp = affine * dpos + mass * v;

        auto w = tc.w(i, j, k);
        contrib[0] = mv[0] + tmp[0] * w;
        contrib[1] = mv[1] + tmp[1] * w;
        contrib[2] = mv[2] + tmp[2] * w;
        contrib[3] = mass * w;

        auto node = state.grid_node(tc.base_coord[0] + i, tc.base_coord[1] + j,
                                    tc.base_coord[2] + k);
        for (int p = 0; p <= dim + 1; p++) {
          atomicAdd(&node[p], contrib[p]);
        }
      }
    }
  }
}

void sort(State &state) {
}

__global__ void G2P(State &state) {
  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  auto inv_dx = state.inv_dx;
  real dt = state.dt;

  Vector x = state.get_x(part_id);
  Vector v;
  Matrix F = state.get_F(part_id);
  Matrix C;

  TransferCommon tc(state, x);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        Vector dpos = tc.dpos(i, j, k);

        auto node = state.grid_node(tc.base_coord[0] + i, tc.base_coord[1] + j,
                                    tc.base_coord[2] + k);
        auto node_v = Vector(node[0], node[1], node[2]);

        auto w = tc.w(i, j, k);
        v = v + w * node_v;
        C = C + Matrix::outer_product(w * node_v, 4 * inv_dx * inv_dx * dpos);
      }
    }
  }
  // state.set_x(part_id, x + state.dt * v);
  state.set_v(part_id, v);
  state.set_F(part_id, (Matrix(1) + dt * C) * F);
}

__global__ void test_svd(int n, Matrix *A, Matrix *U, Matrix *sig, Matrix *V) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    svd(A[id], U[id], sig[id], V[id]);
  }
}

void test_svd_cuda(int n, real *A, real *U, real *sig, real *V) {
  Matrix *d_A, *d_U, *d_sig, *d_V;

  cudaMalloc(&d_A, sizeof(Matrix) * (unsigned int)(n));
  cudaMemcpy(d_A, A, sizeof(Matrix) * n, cudaMemcpyHostToDevice);

  cudaMalloc(&d_U, sizeof(Matrix) * (unsigned int)(n));
  cudaMalloc(&d_sig, sizeof(Matrix) * (unsigned int)(n));
  cudaMalloc(&d_V, sizeof(Matrix) * (unsigned int)(n));

  test_svd<<<(n + 127) / 128, 128>>>(n, d_A, d_U, d_sig, d_V);

  std::vector<Matrix> h_U(n), h_sig(n), h_V(n);
  cudaMemcpy(h_U.data(), d_U, sizeof(Matrix) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sig.data(), d_sig, sizeof(Matrix) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_V.data(), d_V, sizeof(Matrix) * n, cudaMemcpyDeviceToHost);

  // Taichi uses column-first storage
  for (int p = 0; p < n; p++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        U[p * 12 + 4 * i + j] = h_U[p][j][i];
        sig[p * 12 + 4 * i + j] = h_sig[p][j][i];
        V[p * 12 + 4 * i + j] = h_V[p][j][i];
      }
    }
  }
}

__global__ void normalize_grid(State &state) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int boundary = 3;
  if (id < state.num_cells) {
    auto node = state.grid_node(id);
    if (node[dim + 1] > 0) {
      real inv_m = 1.0f / node[dim + 1];
      node[0] *= inv_m;
      node[1] *= inv_m;
      node[2] *= inv_m;
      for (int i = 0; i < dim; i++) {
        // TODO: optimize
        node[i] += state.gravity[i] * state.dt;
      }
      int x = id / (state.res[1] * state.res[2]),
          y = id / state.res[2] % state.res[1], z = id % state.res[2];
      if (x < boundary || y < boundary || y < boundary ||
          x + boundary >= state.res[0] || y + boundary >= state.res[1] ||
          z + boundary >= state.res[2]) {
        // All sticky for now
        for (int i = 0; i < dim; i++) {
          node[i] = 0;
        }
      }
    }
  }
}

void advance(State &state) {
  cudaMemset(state.grid_storage,
             state.num_cells * (state.dim + 1) * sizeof(real), 0);
  static constexpr int block_size = 128;
  int num_blocks = (state.num_particles + block_size - 1) / block_size;
  P2G<<<num_blocks, block_size>>>(state);
  return;
  // TODO: This should be done in tf
  int num_blocks_grid = state.grid_size();
  normalize_grid<<<(num_blocks_grid + block_size - 1) / block_size,
                   block_size>>>(state);
  G2P<<<num_blocks, block_size>>>(state);
}

void initialize_mpm3d_state(void *&state_, float *intiial_positions) {
  int res[dim];
  int n = 100;
  res[0] = 100;
  res[1] = 100;
  res[2] = 100;
  int part_n = 30;
  for (int i = 0; i < part_n; i++) {
  }
  int num_particles = part_n * part_n * part_n;

  real gravity[dim];
  for (int i = 0; i < dim; i++) {
    gravity[i] = 0;
  }
  gravity[1] = -9.8f;

  // State(int res[dim], int num_particles, real dx, real dt, real
  auto state = new State(res, num_particles, 1.0f / n, 1e-4f, gravity);
  state_ = state;

  cudaMemcpy(state->x_storage, intiial_positions,
             sizeof(Vector) * num_particles, cudaMemcpyHostToDevice);
}

void advance_mpm3d_state(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  advance(*state);
}

std::vector<float> fetch_mpm3d_particles(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  return state->fetch_x();
}
