#include "kernels.h"
#include "linalg.h"
#include "state.cuh"
#include <cstdio>
#include <vector>

// For deformation gradient update

// Takes B, dL/dAB
// Returns dL/dA
__device__ Matrix dAB2dA(const Matrix &B, const Matrix &dAB) {
  Matrix dA;
  for (int p = 0; p < dim; p++) {
    for (int q = 0; q < dim; q++) {
      for (int j = 0; j < dim; j++) {
        dA[p][q] += dAB[p][j] * B[q][j];
      }
    }
  }
  return dA;
}

// Takes A, B, dL/dAB
// Returns dL/dB
__device__ Matrix dAB2dB(const Matrix &A, const Matrix &dAB) {
  Matrix dB;
  for (int p = 0; p < dim; p++) {
    for (int q = 0; q < dim; q++) {
      for (int i = 0; i < dim; i++) {
        dB[p][q] += dAB[i][q] * A[i][p];
      }
    }
  }
  return dB;
}

__device__ Vector duTv2du(const Vector &v, const real &duTv) {
  return duTv * v;
}

__device__ Vector duTv2dv(const Vector &u, const real &duTv) {
  return duTv * u;
}

__global__ void P2G_backward(State state, State next_state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation

  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  Vector x = state.get_x(part_id), v = state.get_v(part_id);
  Matrix F = state.get_F(part_id);
  Matrix C = state.get_C(part_id);

  auto grad_x_next = next_state.get_grad_x(part_id);
  auto grad_C_next = next_state.get_grad_C(part_id);
  auto grad_v_next = next_state.get_grad_v(part_id);
  auto grad_F_next = next_state.get_grad_F(part_id);
  Matrix G;  // TODO

  // (A) v_p^n+1, accumulate
  grad_v_next = grad_v_next + state.dt * grad_x_next;

  // (B) C_p^n+1, accumulate
  for (int alpha = 0; alpha < dim; alpha++) {
    for (int beta = 0; beta < dim; beta++) {
      for (int gamma = 0; gamma < dim; gamma++) {
        grad_C_next[alpha][beta] +=
            state.dt * grad_F_next[alpha][gamma] * F[beta][gamma];
      }
    }
  }

  next_state.set_grad_v(part_id, grad_v_next);
  next_state.set_grad_C(part_id, grad_C_next);

  TransferCommon<true> tc(state, x);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        real N = tc.w(i, j, k);
        Vector dpos = tc.dpos(i, j, k);

        // (C) v_i^n
        real grad_v_i[dim];
        for (int alpha = 0; alpha < dim; alpha++) {
          grad_v_i[alpha] = grad_v_next[alpha] * N;
          if (mpm_enalbe_apic) {
            for (int beta = 0; beta < dim; beta++) {
              grad_v_i[alpha] += grad_C_next[alpha][beta] * dpos[beta];
            }
          }
        }
        auto grad_n = state.grad_grid_node(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        for (int d = 0; d < dim; d++) {
          atomicAdd(&grad_n[d], grad_v_i[d]);
        }
      }
    }
  }
}

__global__ void grid_backward(State state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < state.num_cells) {
    auto node = state.grid_node(id);
    auto grad_node = state.grad_grid_node(id);
    if (node[dim] > 0) {
      int x = id / (state.res[1] * state.res[2]),
          y = id / state.res[2] % state.res[1], z = id % state.res[2];
      // (D)
      // Convert grad_v to grad_p
      // grad_p = grad_v / m
      auto m = node[dim];
      real inv_m = 1.0f / m;  // TODO: guard?
      auto grad_v_i = state.get_grad_grid_velocity(x, y, z);
      auto grad_p = inv_m * grad_v_i;
      auto v_i = Vector(node);
      auto p_i = m * v_i;
      // (E)
      real grad_m = 0;
      for (int alpha = 0; alpha < dim; alpha++) {
        grad_m -= inv_m * v_i[alpha] * grad_v_i[alpha];
        grad_node[alpha] = grad_p[alpha];
      }
      grad_node[dim] = grad_m;
    }
  }
}

// (F), (G), (H), (I), (J)
__global__ void G2P_backward(State state, State next_state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation
  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  auto x = state.get_x(part_id);
  auto v = state.get_v(part_id);
  auto F = state.get_F(part_id);
  auto C = state.get_C(part_id);
  auto P = state.get_P(part_id);

  Matrix grad_P_next = next_state.get_grad_P(part_id);
  Matrix grad_P, grad_F;
  auto grad_F_next = next_state.get_grad_F(part_id);
  auto grad_C_next = next_state.get_grad_C(part_id);
  auto grad_v_next = next_state.get_grad_v(part_id);
  Matrix grad_C;
  auto C_next = next_state.get_C(part_id);

  TransferCommon<true> tc(state, x);
  Vector grad_v;
  real grad_P_scale = state.dt * state.invD * V;

  // (H) term 2
  Times_Rotated_dP_dF_FixedCorotated(mu, lambda, F.data(), grad_P.data(),
                                     grad_F.data());

  for (int alpha = 0; alpha < dim; alpha++) {
    for (int beta = 0; beta < dim; beta++) {
      // (H) term 1
      for (int gamma = 0; gamma < dim; gamma++) {
        grad_F[alpha][beta] +=
            grad_F_next[gamma][beta] *
            (real(gamma == alpha) + state.dt * C_next[gamma][alpha]);
      }
    }
  }

  // (J) term 1
  Vector grad_x = next_state.get_grad_x(part_id);
  // printf("grad_x %f\n", grad_x[0]);
  auto G = state.invD * state.dt * P * transposed(F) + m_p * C;

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        real N = tc.w(i, j, k);
        Vector dpos = tc.dpos(i, j, k);
        auto grad_p = state.get_grad_grid_velocity(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);

        auto grad_N = tc.dw(i, j, k);
        real grad_v_i[dim];
        real mi = state.get_grid_mass(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        auto vi = state.get_grid_velocity(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        real grad_mi =
            state.grad_grid_node(tc.base_coord[0] + i, tc.base_coord[1] + j,
                                 tc.base_coord[2] + k)[dim];
        for (int alpha = 0; alpha < dim; alpha++) {
          // (F) v_p^n
          grad_v[alpha] += N * m_p * grad_p[alpha];
          grad_v_i[alpha] = grad_v_next[alpha] * N;

          /*
          grad_x[alpha] +=
              grad_N[alpha] * (grad_v_next[alpha] * state.invD +
                               grad_p[alpha] * mi * vi[alpha] + m_p * grad_mi);

          // temporally disable
          for (int beta = 0; beta < dim; beta++) {
            grad_x[alpha] += state.invD * grad_C_next[beta][alpha] *
                                 (grad_N[alpha] * vi[alpha] * dpos[beta] -
                                  tc.w(i, j, k) * vi[alpha]) -
                             grad_p[beta] * G[beta][alpha];
          }
          */

          for (int beta = 0; beta < dim; beta++) {
            // (G) P_p^n
            for (int gamma = 0; gamma < dim; gamma++) {
              grad_P[alpha][beta] +=
                  grad_P_scale * grad_p[alpha] * F[gamma][beta] * dpos[gamma];
              // (H), term 3
              grad_F[alpha][beta] +=
                  grad_P_scale * P[gamma][beta] * dpos[alpha];
            }
            grad_v_i[alpha] += grad_C_next[alpha][beta] * dpos[beta];
            // (I) C_p^n
            grad_C[alpha][beta] += grad_p[alpha] * m_p * dpos[beta];
          }
        }
        state.set_grad_grid_velocity(
            i, j, k, Vector(grad_v_i[0], grad_v_i[1], grad_v_i[2]));
      }
    }
  }
  state.set_grad_v(part_id, grad_v);
  state.set_grad_x(part_id, grad_x);
}

void backward(State &state, State &next) {
  state.clear_gradients();
  int num_blocks =
      (state.num_particles + particle_block_dim - 1) / particle_block_dim;
  int num_blocks_grid = state.grid_size();
  P2G_backward<<<num_blocks, particle_block_dim>>>(state, next);
  auto err = cudaThreadSynchronize();
  if (err) {
    printf("Launch: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  grid_backward<<<state.num_cells / grid_block_dim + 1, grid_block_dim>>>(
      state);
  G2P_backward<<<num_blocks, particle_block_dim>>>(state, next);
}

void backward_mpm3d_state(void *state_, void *next_state_) {
  State *state = reinterpret_cast<State *>(state_);
  State *next_state = reinterpret_cast<State *>(next_state_);
  backward(*state, *next_state);
}

void set_grad_loss(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  state->clear_gradients();
  int num_particles = state->num_particles;
  std::vector<float> grad_x_host(num_particles * dim);
  for (int i = 0; i < num_particles; i++) {
    grad_x_host[i * 3] = 1;
  }
  cudaMemcpy(state->grad_x_storage, grad_x_host.data(),
             sizeof(real) * dim * num_particles, cudaMemcpyHostToDevice);
}

std::vector<float> fetch_mpm3d_grad_v(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  return state->fetch_grad_v();
}

std::vector<float> fetch_mpm3d_grad_x(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  return state->fetch_grad_x();
}
