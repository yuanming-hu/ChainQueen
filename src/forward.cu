#include "kernels.h"
#include "linalg.h"
#include "state.cuh"
#include <cstdio>
#include <vector>


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

// Do not consider sorting for now. Use atomics instead.

// One particle per thread
__global__ void P2G(State state) {
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
  real E = 10;    // TODO: variable E
  real nu = 0.3;    // TODO: variable nu
  Matrix F = state.get_F(part_id);
  Matrix C = state.get_C(part_id);

  TransferCommon<> tc(state, x);

  // Fixed corotated
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  real J = determinant(F);

  Matrix r, s;
  polar_decomp(F, r, s);
  Matrix stress =
      -4 * inv_dx * inv_dx * dt * volume *
      (2 * mu * (F - r) * transposed(F) + Matrix(lambda * (J - 1) * J));

  auto affine = stress + mass * C;

  // printf("%d %d %d\n", tc.base_coord[0], tc.base_coord[1], tc.base_coord[2]);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        Vector dpos = tc.dpos(i, j, k);

        real contrib[dim + 1];

        auto tmp = affine * dpos + mass * v;

        auto w = tc.w(i, j, k);
        contrib[0] = tmp[0] * w;
        contrib[1] = tmp[1] * w;
        contrib[2] = tmp[2] * w;
        contrib[3] = mass * w;

        auto node = state.grid_node(tc.base_coord[0] + i, tc.base_coord[1] + j,
                                    tc.base_coord[2] + k);
        for (int p = 0; p < dim + 1; p++) {
          atomicAdd(&node[p], contrib[p]);
        }
      }
    }
  }
}

void sort(State state) {
}

__global__ void G2P(State state, State next_state) {
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

  TransferCommon<> tc(state, x);

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
  next_state.set_x(part_id, x + state.dt * v);
  next_state.set_v(part_id, v);
  next_state.set_F(part_id, (Matrix(1) + dt * C) * F);
  next_state.set_C(part_id, C);
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

__global__ void normalize_grid(State state) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int boundary = 3;
  if (id < state.num_cells) {
    auto node = state.grid_node(id);
    if (node[dim] > 0) {
      real inv_m = 1.0f / node[dim];
      node[0] *= inv_m;
      node[1] *= inv_m;
      node[2] *= inv_m;
      for (int i = 0; i < dim; i++) {
        node[i] += state.gravity[i] * state.dt;
      }
      int x = id / (state.res[1] * state.res[2]),
          y = id / state.res[2] % state.res[1], z = id % state.res[2];
      if (x < boundary || y < boundary || y < boundary ||
          x + boundary >= state.res[0] || y + boundary >= state.res[1] ||
          z + boundary >= state.res[2]) {
        // All sticky for now
        /*
        for (int i = 0; i < dim; i++) {
          node[i] = 0;
        }
        */
        node[1] = max(0.0f, node[1]);
      }
    }
  }
}

void advance(const State &state) {
  cudaMemset(state.grid_storage,
             0, state.num_cells * (state.dim + 1) * sizeof(real));
  static constexpr int block_size = 128;
  int num_blocks = (state.num_particles + block_size - 1) / block_size;
  P2G<<<num_blocks, block_size>>>(state);

  auto err = cudaThreadSynchronize();
  if (err) {
    printf("Launch: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  // TODO: This should be done in tf
  int num_blocks_grid = state.grid_size();
  normalize_grid<<<(num_blocks_grid + block_size - 1) / block_size,
                   block_size>>>(state);
  G2P<<<num_blocks, block_size>>>(state, state);
}

void initialize_mpm3d_state(void *&state_, float *initial_positions) {
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
  auto state = new State(res, num_particles, 1.0f / n, 1e-3f, gravity);
  state_ = state;

  cudaMemcpy(state->x_storage, initial_positions,
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
