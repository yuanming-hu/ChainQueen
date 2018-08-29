#include "kernels.h"
#include "linalg.h"
#include "particle.h"
#include "svd.cuh"

struct State : public StateBase {
  TC_FORCE_INLINE __device__ int linearized_offset(int x, int y, int z) const {
    return res[2] * (res[1] * x + y) + z;
  }

  TC_FORCE_INLINE __device__ real *grid_node(int x, int y, int z) const {
    return grid_storage + (dim + 1) * (res[2] * (res[1] * x + y) + z);
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
};

constexpr int dim = 3;
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

  //stress = -4 * inv_dx * inv_dx * dt * volume * stress;

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

__global__ void normalize_grid(State &state) {
}

void advance(State &state) {
  sort(state);
  static constexpr int block_size = 128;
  int num_blocks = (state.num_particles + block_size - 1) / block_size;
  P2G<<<num_blocks, block_size>>>(state);
  // normalize_grid<<<>>>(state);
  G2P<<<num_blocks, block_size>>>(state);
}
