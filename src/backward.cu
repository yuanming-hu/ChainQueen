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

TC_FORCE_INLINE __device__ real Clamp_Small_Magnitude(const real input) {
  real magnitude = input > 0 ? input : -input;
  real sign = input > 0 ? 1.f : -1.f;
  real output = magnitude > 1e-6 ? magnitude : 1e-6;
  return output * sign;
}

__device__ void Times_Rotated_dP_dF_FixedCorotated(const real mu,
                                                   const real lambda,
                                                   const real *F,
                                                   const real *dF,
                                                   real *dP) {
  real U[9];
  real S[3];
  real V[9];
  svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6],
      U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6],
      V[1], V[4], V[7], V[2], V[5], V[8]);

  //
  real J = S[0] * S[1] * S[2];
  real scaled_mu = 2.f * mu;
  real scaled_lambda = lambda * (J - 1.f);
  real P_hat[3];
  P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
  P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
  P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);

  real dP_hat_dSigma_upper[6];
  scaled_lambda = lambda * (2.f * J - 1.f) * J;
  for (int i = 0; i < 3; ++i)
    dP_hat_dSigma_upper[i] = scaled_mu + lambda * J * J / (S[i] * S[i]);
  dP_hat_dSigma_upper[3] = scaled_lambda / (S[0] * S[1]);
  dP_hat_dSigma_upper[4] = scaled_lambda / (S[0] * S[2]);
  dP_hat_dSigma_upper[5] = scaled_lambda / (S[1] * S[2]);

  scaled_lambda = -lambda * (J - 1.f) * J;
  real M[3];
  M[0] = 0.5f * (2.f * mu + scaled_lambda / (S[0] * S[1]));
  M[1] = 0.5f * (2.f * mu + scaled_lambda / (S[0] * S[2]));
  M[2] = 0.5f * (2.f * mu + scaled_lambda / (S[2] * S[2]));
  //

  real P[3];
  P[0] = 0.5 * (P_hat[0] + P_hat[1]) / Clamp_Small_Magnitude(S[0] + S[1]);
  P[1] = 0.5 * (P_hat[0] + P_hat[2]) / Clamp_Small_Magnitude(S[0] + S[2]);
  P[2] = 0.5 * (P_hat[1] + P_hat[2]) / Clamp_Small_Magnitude(S[1] + S[2]);

  real dF_hat[9];
  dF_hat[0] = (dF[0] * U[0] + dF[1] * U[1] + dF[2] * U[2]) * V[0] +
              (dF[3] * U[0] + dF[4] * U[1] + dF[5] * U[2]) * V[1] +
              (dF[6] * U[0] + dF[7] * U[1] + dF[8] * U[2]) * V[2];
  dF_hat[1] = (dF[0] * U[3] + dF[1] * U[4] + dF[2] * U[5]) * V[0] +
              (dF[3] * U[3] + dF[4] * U[4] + dF[5] * U[5]) * V[1] +
              (dF[6] * U[3] + dF[7] * U[4] + dF[8] * U[5]) * V[2];
  dF_hat[2] = (dF[0] * U[6] + dF[1] * U[7] + dF[2] * U[8]) * V[0] +
              (dF[3] * U[6] + dF[4] * U[7] + dF[5] * U[8]) * V[1] +
              (dF[6] * U[6] + dF[7] * U[7] + dF[8] * U[8]) * V[2];
  dF_hat[3] = (dF[0] * U[0] + dF[1] * U[1] + dF[2] * U[2]) * V[3] +
              (dF[3] * U[0] + dF[4] * U[1] + dF[5] * U[2]) * V[4] +
              (dF[6] * U[0] + dF[7] * U[1] + dF[8] * U[2]) * V[5];
  dF_hat[4] = (dF[0] * U[3] + dF[1] * U[4] + dF[2] * U[5]) * V[3] +
              (dF[3] * U[3] + dF[4] * U[4] + dF[5] * U[5]) * V[4] +
              (dF[6] * U[3] + dF[7] * U[4] + dF[8] * U[5]) * V[5];
  dF_hat[5] = (dF[0] * U[6] + dF[1] * U[7] + dF[2] * U[8]) * V[3] +
              (dF[3] * U[6] + dF[4] * U[7] + dF[5] * U[8]) * V[4] +
              (dF[6] * U[6] + dF[7] * U[7] + dF[8] * U[8]) * V[5];
  dF_hat[6] = (dF[0] * U[0] + dF[1] * U[1] + dF[2] * U[2]) * V[6] +
              (dF[3] * U[0] + dF[4] * U[1] + dF[5] * U[2]) * V[7] +
              (dF[6] * U[0] + dF[7] * U[1] + dF[8] * U[2]) * V[8];
  dF_hat[7] = (dF[0] * U[3] + dF[1] * U[4] + dF[2] * U[5]) * V[6] +
              (dF[3] * U[3] + dF[4] * U[4] + dF[5] * U[5]) * V[7] +
              (dF[6] * U[3] + dF[7] * U[4] + dF[8] * U[5]) * V[8];
  dF_hat[8] = (dF[0] * U[6] + dF[1] * U[7] + dF[2] * U[8]) * V[6] +
              (dF[3] * U[6] + dF[4] * U[7] + dF[5] * U[8]) * V[7] +
              (dF[6] * U[6] + dF[7] * U[7] + dF[8] * U[8]) * V[8];

  real dP_hat[9];
  dP_hat[0] = dP_hat_dSigma_upper[0] * dF_hat[0] +
              dP_hat_dSigma_upper[3] * dF_hat[4] +
              dP_hat_dSigma_upper[4] * dF_hat[8];
  dP_hat[4] = dP_hat_dSigma_upper[3] * dF_hat[0] +
              dP_hat_dSigma_upper[1] * dF_hat[4] +
              dP_hat_dSigma_upper[5] * dF_hat[8];
  dP_hat[8] = dP_hat_dSigma_upper[4] * dF_hat[0] +
              dP_hat_dSigma_upper[5] * dF_hat[4] +
              dP_hat_dSigma_upper[2] * dF_hat[8];
  dP_hat[3] = ((M[0] + P[0]) * dF_hat[3] + (M[0] - P[0]) * dF_hat[1]);
  dP_hat[1] = ((M[0] - P[0]) * dF_hat[3] + (M[0] + P[0]) * dF_hat[1]);
  dP_hat[6] = ((M[1] + P[1]) * dF_hat[6] + (M[1] - P[1]) * dF_hat[2]);
  dP_hat[2] = ((M[1] - P[1]) * dF_hat[6] + (M[1] + P[1]) * dF_hat[2]);
  dP_hat[7] = ((M[2] + P[2]) * dF_hat[7] + (M[2] - P[2]) * dF_hat[5]);
  dP_hat[5] = ((M[2] - P[2]) * dF_hat[7] + (M[2] + P[2]) * dF_hat[5]);

  dP[0] = (dP_hat[0] * U[0] + dP_hat[1] * U[3] + dP_hat[2] * U[6]) * V[0] +
          (dP_hat[3] * U[0] + dP_hat[4] * U[3] + dP_hat[5] * U[6]) * V[3] +
          (dP_hat[6] * U[0] + dP_hat[7] * U[3] + dP_hat[8] * U[6]) * V[6];
  dP[1] = (dP_hat[0] * U[1] + dP_hat[1] * U[4] + dP_hat[2] * U[7]) * V[0] +
          (dP_hat[3] * U[1] + dP_hat[4] * U[4] + dP_hat[5] * U[7]) * V[3] +
          (dP_hat[6] * U[1] + dP_hat[7] * U[4] + dP_hat[8] * U[7]) * V[6];
  dP[2] = (dP_hat[0] * U[2] + dP_hat[1] * U[5] + dP_hat[2] * U[8]) * V[0] +
          (dP_hat[3] * U[2] + dP_hat[4] * U[5] + dP_hat[5] * U[8]) * V[3] +
          (dP_hat[6] * U[2] + dP_hat[7] * U[5] + dP_hat[8] * U[8]) * V[6];
  dP[3] = (dP_hat[0] * U[0] + dP_hat[1] * U[3] + dP_hat[2] * U[6]) * V[1] +
          (dP_hat[3] * U[0] + dP_hat[4] * U[3] + dP_hat[5] * U[6]) * V[4] +
          (dP_hat[6] * U[0] + dP_hat[7] * U[3] + dP_hat[8] * U[6]) * V[7];
  dP[4] = (dP_hat[0] * U[1] + dP_hat[1] * U[4] + dP_hat[2] * U[7]) * V[1] +
          (dP_hat[3] * U[1] + dP_hat[4] * U[4] + dP_hat[5] * U[7]) * V[4] +
          (dP_hat[6] * U[1] + dP_hat[7] * U[4] + dP_hat[8] * U[7]) * V[7];
  dP[5] = (dP_hat[0] * U[2] + dP_hat[1] * U[5] + dP_hat[2] * U[8]) * V[1] +
          (dP_hat[3] * U[2] + dP_hat[4] * U[5] + dP_hat[5] * U[8]) * V[4] +
          (dP_hat[6] * U[2] + dP_hat[7] * U[5] + dP_hat[8] * U[8]) * V[7];
  dP[6] = (dP_hat[0] * U[0] + dP_hat[1] * U[3] + dP_hat[2] * U[6]) * V[2] +
          (dP_hat[3] * U[0] + dP_hat[4] * U[3] + dP_hat[5] * U[6]) * V[5] +
          (dP_hat[6] * U[0] + dP_hat[7] * U[3] + dP_hat[8] * U[6]) * V[8];
  dP[7] = (dP_hat[0] * U[1] + dP_hat[1] * U[4] + dP_hat[2] * U[7]) * V[2] +
          (dP_hat[3] * U[1] + dP_hat[4] * U[4] + dP_hat[5] * U[7]) * V[5] +
          (dP_hat[6] * U[1] + dP_hat[7] * U[4] + dP_hat[8] * U[7]) * V[8];
  dP[8] = (dP_hat[0] * U[2] + dP_hat[1] * U[5] + dP_hat[2] * U[8]) * V[2] +
          (dP_hat[3] * U[2] + dP_hat[4] * U[5] + dP_hat[5] * U[8]) * V[5] +
          (dP_hat[6] * U[2] + dP_hat[7] * U[5] + dP_hat[8] * U[8]) * V[8];
};

// Constitutive models
/*
__device__ Matrix dP2dF_fixed_corotated(const Matrix &R,
                                        const Matrix &F,
                                        const Matrix &dF) {
  //Matrix dR, dS dS = dF * transposed(R) +
  //                   F * transposed(dR) return std::make_pair(dR, dS);
}
*/

//
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
          for (int beta = 0; beta < dim; beta++) {
            grad_v_i[alpha] += grad_C_next[alpha][beta] * dpos[beta];
          }
        }
        state.set_grad_grid_velocity(
            i, j, k, Vector(grad_v_i[0], grad_v_i[1], grad_v_i[2]));
      }

      /*
      auto grad_N = tc.dw(i, j, k);

      for (int alpha = 0; alpha < dim; alpha++) {
        grad_x[alpha] +=
            grad_N[alpha] * (grad_v_next[alpha] * state.invD +
                             grad_p[alpha] * mi * vi[alpha] + m_p * grad_mi);
        for (int beta = 0; beta < dim; beta++) {
          grad_x[alpha] += state.invD * grad_C_next[beta][alpha] *
                               (grad_N[alpha] * vi[alpha] * dpos[beta] -
                                tc.w(i, j, k) * vi[alpha]) -
                           grad_p[beta] * G[beta][alpha];
          grad_C[alpha][beta] += grad_p[alpha] * m_p * (dpos[beta]);
        }
      }
      */
    }
  }
}

__global__ void grid_backward(State state, State next_state) {
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
      state.set_grad_grid_velocity(x, y, z, grad_p);
      // (E)
      real grad_m = 0;
      for (int alpha = 0; alpha < dim; alpha++) {
        grad_m -= inv_m * v_i[alpha] * grad_v_i[alpha];
        atomicAdd(&grad_node[alpha], grad_v_i[alpha]);
      }
      atomicAdd(&grad_node[dim], grad_m);
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

  Vector x = state.get_x(part_id), v = state.get_v(part_id);
  Matrix F = state.get_F(part_id);
  Matrix C = state.get_C(part_id);
  Matrix P = state.get_P(part_id);

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
  auto G = state.invD * state.dt * P * transposed(F) + m_p * C;

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        real N = tc.w(i, j, k);
        Vector dpos = tc.dpos(i, j, k);
        auto grad_p = state.get_grad_grid_velocity(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);

        auto grad_N = tc.dw(i, j, k);
        // () v_i^n
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

          grad_x[alpha] +=
              grad_N[alpha] * (grad_v_next[alpha] * state.invD +
                               grad_p[alpha] * mi * vi[alpha] + m_p * grad_mi);
          for (int beta = 0; beta < dim; beta++) {
            grad_x[alpha] += state.invD * grad_C_next[beta][alpha] *
                                 (grad_N[alpha] * vi[alpha] * dpos[beta] -
                                  tc.w(i, j, k) * vi[alpha]) -
                             grad_p[beta] * G[beta][alpha];
          }

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
  grid_backward<<<state.num_cells / grid_block_dim + 1, grid_block_dim>>>(state,
                                                                          next);
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
    grad_x_host[i * 3] = num_particles;
  }
  cudaMemcpy(state->grad_x_storage, grad_x_host.data(),
             sizeof(real) * dim * num_particles, cudaMemcpyHostToDevice);
}

std::vector<float> fetch_mpm3d_grad_v(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  return state->fetch_grad_v();
}
