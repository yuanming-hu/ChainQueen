#include "kernels.h"
#include "linalg.h"
#include "particle.h"
#include "svd.cuh"
#include "../../../../../../../opt/cuda/include/driver_types.h"
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

  TC_FORCE_INLINE __device__ Vector get_vector(real *p, int part_id) {
    return Vector(p[part_id], p[part_id + num_particles],
                  p[part_id + num_particles * 2]);
  }

  TC_FORCE_INLINE __device__ Vector get_v(int part_id) {
    return get_vector(v_storage, part_id);
  }

  TC_FORCE_INLINE __device__ Vector get_x(int part_id) {
    return get_vector(x_storage, part_id);
  }

  TC_FORCE_INLINE __device__ Matrix get_F(int part_id) {
    return get_matrix(F_storage, part_id);
  }

  TC_FORCE_INLINE __device__ Matrix get_C(int part_id) {
    return get_matrix(C_storage, part_id);
  }

  // TC_FORCE_INLINE __device__ get_cell
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

// Do not consider sorting for now. Use atomics instead.

__global__ void P2G(State &state) {
  // One particle per thread

  auto inv_dx = real(1.0) / state.dx;

  constexpr int scratch_size = 8;
  __shared__ real scratch[dim + 1][scratch_size][scratch_size][scratch_size];

  // load from global memory
  int part_id = 0;  // TODO

  real mass = 1;    // TODO: variable mass
  real volume = 1;  // TODO: variable vol
  real E = 10000;   // TODO: variable E
  real nu = 0.3;    // TODO: variable nu

  real dt = state.dt;

  Vector x = state.get_x(part_id), v = state.get_v(part_id);

  real weight[dim][spline_size];
  // Compute B-Spline weights
  for (int v = 0; v < dim; ++v) {
    real d0 = x[v] * inv_dx;
    real z = ((real)1.5 - d0);
    weight[v][0] = (real)0.5 * z * z;
    d0 = d0 - 1.0f;
    weight[v][1] = (real)0.75 - d0 * d0;
    z = (real)1.5 - (1.0f - d0);
    weight[v][2] = (real)0.5 * z * z;
  }

  real val[dim + 1];

  int base_coord[3];
  for (int p = 0; p < 3; p++)
    base_coord[p] = int(x[p] * inv_dx - 0.5);

  Matrix stress;
  Matrix F = state.get_F(part_id);
  // Fixed corotated

  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  real J = determinant(F);
  Matrix r, s;
  polar_decomp(F, r, s);
  stress = -4 * inv_dx * inv_dx * dt * volume *
           (2 * mu * (F - r) * transposed(F) + Matrix(lambda * (J - 1) * J));

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        auto w = weight[0][i] * weight[1][j] * weight[2][k];
        int base_coord[dim];

        val[0] = mass * w;
        Vector dpos;

        /*
        // reduce in warp
        for (int iter = 1; iter <= mark; iter <<= 1) {
          T tmp[4];
          for (int i = 0; i < 4; ++i)
            tmp[i] = __shfl_down(val[i], iter);
          if (interval >= iter)
            for (int i = 0; i < 4; ++i)
              val[i] += tmp[i];
        }

        // cross-warp atomic
        for (int r = 0; r < dim + 1; r++) {
          atomicAdd();
        }
        */

        // scatter mass

        real contrib[dim + 1] = {0};

        auto node = state.grid_node(base_coord[0] + i, base_coord[1] + j,
                                    base_coord[2] + k);
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
  if (id < state.num_cells) {
    auto node = state.grid_node(id);
    int inv_m = max(1e-30f, node[dim + 1]);
    node[0] *= inv_m;
    node[1] *= inv_m;
    node[2] *= inv_m;
  }
}

void advance(State &state) {
  cudaMemset(state.grid_storage,
             state.grid_size() * (state.dim + 1) * sizeof(real), 0);
  // sort(state);
  static constexpr int block_size = 128;
  int num_blocks = (state.num_particles + block_size - 1) / block_size;
  P2G<<<num_blocks, block_size>>>(state);
  // TODO: This should be done in tf
  int num_blocks_grid = state.grid_size();
  normalize_grid<<<(num_blocks_grid + block_size - 1) / block_size,
                   block_size>>>(state);
  G2P<<<num_blocks, block_size>>>(state);
}
