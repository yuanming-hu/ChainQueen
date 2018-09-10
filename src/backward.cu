#include "kernels.h"
#include "linalg.h"
#include "state.cuh"
#include <cstdio>
#include <vector>

// For deformation gradient update


template <int dim>
__global__ void P2G_backward(State state, State next_state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation
  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  Vector x = state.get_x(part_id);
  Vector v = state.get_v(part_id);
  Matrix F = state.get_F(part_id);
  Matrix C = state.get_C(part_id);

  auto grad_x_next = next_state.get_grad_x(part_id);
  auto grad_v_next = next_state.get_grad_v(part_id);
  auto grad_F_next = next_state.get_grad_F(part_id);
  auto grad_C_next = next_state.get_grad_C(part_id);

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

  // Accumulate to grad_v and grad_C
  next_state.set_grad_v(part_id, grad_v_next);
  next_state.set_grad_C(part_id, grad_C_next);

  TransferCommon<dim, true> tc(state, x);

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
            grad_v_i[alpha] +=
                state.invD * N * grad_C_next[alpha][beta] * dpos[beta];
          }
        }
        auto grad_n = state.grad_grid_node(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        for (int d = 0; d < dim; d++) {
          // printf("grad_v_i %d %f\n", d, grad_v_i[d]);
          atomicAdd(&grad_n[d], grad_v_i[d]);
        }
      }
    }
  }
}

template <int dim>
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
      for (int d = 0; d < dim; d++) {
        // printf("g v %f\n", v_i[d]);
      }
      // printf("g m %f\n", m);
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
template <int dim>
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

  auto grad_F_next = next_state.get_grad_F(part_id);
  auto grad_C_next = next_state.get_grad_C(part_id);
  auto grad_P_next = next_state.get_grad_P(part_id);
  auto grad_v_next = next_state.get_grad_v(part_id);

  auto C_next = next_state.get_C(part_id);

  Matrix grad_P;
  Matrix grad_F;
  Matrix grad_C;

  TransferCommon<dim, true> tc(state, x);
  {
    /*
    real dx = 1e-4f;
    TransferCommon<true> tc2(state, x + Vector(0, 0, dx));
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          auto d = tc.dw(i, j, k);
          printf("%f %f\n", d[2], (tc2.w(i, j, k) - tc.w(i, j, k))/ dx);
        }
      }
    }
    */
  }

  Vector grad_v;
  real grad_P_scale = state.dt * state.invD * state.V_p;

  // (G) Compute grad_P
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        real N = tc.w(i, j, k);
        Vector dpos = tc.dpos(i, j, k);
        auto grad_p = state.get_grad_grid_velocity(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        auto grad_N = tc.dw(i, j, k);
        for (int alpha = 0; alpha < dim; alpha++) {
          for (int beta = 0; beta < dim; beta++) {
            // (G) P_p^n
            for (int gamma = 0; gamma < dim; gamma++) {
              grad_P[alpha][beta] += -N * grad_P_scale * grad_p[alpha] *
                                     F[gamma][beta] * dpos[gamma];
            }
            // (I) C_p^n
            if (mpm_enalbe_apic)
              grad_C[alpha][beta] += N * grad_p[alpha] * state.m_p * dpos[beta];
          }
        }
      }
    }
  }

  // (H) term 2
  Times_Rotated_dP_dF_FixedCorotated(state.mu, state.lambda, F.data(), grad_P.data(),
                                     grad_F.data());
  /*
  Matrix grad_F2;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Matrix inc = F, dec = F;
      real delta = 1e-2f;
      inc[i][j] += delta;
      dec[i][j] -= delta;
      auto diff = (1 / (2 * delta)) * (PK1(inc) - PK1(dec));
      grad_F2 = grad_F2 + grad_P[i][j] * diff;
    }
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%d %d:  %f %f\n", i, j, grad_F2[i][j] * 1e8, grad_F[i][j] * 1e8);
    }
  }

  grad_F = grad_F2;
  */

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
  auto G = -state.invD * state.dt * P * transposed(F);
  if (mpm_enalbe_apic) {
    G = G + state.m_p * C;
  }

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        real N = tc.w(i, j, k);
        Vector dpos = tc.dpos(i, j, k);
        auto grad_p = state.get_grad_grid_velocity(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);

        for (int d = 0; d < dim; d++) {
          // printf("grad p[%d] %.10f\n", d, grad_p[d]);
        }

        auto grad_N = tc.dw(i, j, k);
        auto n = state.grid_node(tc.base_coord[0] + i, tc.base_coord[1] + j,
                                 tc.base_coord[2] + k);
        auto mi = state.get_grid_mass(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        // printf(" m m %f %f\n", mi, n[dim]);
        auto vi = state.get_grid_velocity(
            tc.base_coord[0] + i, tc.base_coord[1] + j, tc.base_coord[2] + k);
        auto grad_mi =
            state.grad_grid_node(tc.base_coord[0] + i, tc.base_coord[1] + j,
                                 tc.base_coord[2] + k)[dim];

        // printf("%.10f\n", grad_p[0]);
        // printf("%.10f\n", grad_p[1]);
        // printf("%.10f\n", grad_p[2]);
        // printf("\n");
        for (int alpha = 0; alpha < dim; alpha++) {
          // (F) v_p^n
          grad_v[alpha] += N * state.m_p * grad_p[alpha];

          // (J) term 5
          grad_x[alpha] += grad_N[alpha] * grad_mi * state.m_p;

          for (int beta = 0; beta < dim; beta++) {
            for (int gamma = 0; gamma < dim; gamma++) {
              // (H), term 3
              grad_F[alpha][beta] += -N * grad_p[gamma] * grad_P_scale *
                                     P[gamma][beta] * dpos[alpha];
            }

            // (J), term 2
            grad_x[alpha] += grad_v_next[beta] * grad_N[alpha] * vi[beta];
            // (J), term 3
            auto tmp = grad_N[alpha] * vi[alpha] * dpos[beta] - N * vi[alpha];
            grad_x[alpha] += state.invD * grad_C[beta][alpha] * tmp;
            // (J), term 4
            grad_x[alpha] +=
                grad_p[beta] *
                (grad_N[alpha] * (state.m_p * v[beta] + (G * dpos)[beta]) -
                 N * G[beta][alpha]);
          }
        }
      }
    }
  }
  state.set_grad_x(part_id, grad_x);
  /*
  for (int i = 0; i < dim; i++) {
    printf("v %d %f %f\n", i, grad_v[i], grad_x[i]);
  }
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      printf("m %d %d %f %f %f %f\n", i, j, grad_F[i][j], grad_C[i][j], F[i][j], grad_P[i][j]);
    }
  }
  */
  state.set_grad_v(part_id, grad_v);
  state.set_grad_F(part_id, grad_F);
  state.set_grad_C(part_id, grad_C);
}

template <int dim>
void backward(State &state, State &next) {
  state.clear_gradients();
  int num_blocks =
      (state.num_particles + particle_block_dim - 1) / particle_block_dim;
  int num_blocks_grid = state.grid_size();
  P2G_backward<dim><<<num_blocks, particle_block_dim>>>(state, next);
  auto err = cudaThreadSynchronize();
  if (err) {
    printf("Launch: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  grid_backward<dim>
      <<<state.num_cells / grid_block_dim + 1, grid_block_dim>>>(state);
  G2P_backward<dim><<<num_blocks, particle_block_dim>>>(state, next);
}

static constexpr int dim = 3;
void MPMGradKernelLauncher(int res[dim],
                           int num_particles,
                           real dx,
                           real dt,
                           real gravity[dim],
                           const real *inx,
                           const real *inv,
                           const real *inF,
                           const real *inC,
                           const real *outx,
                           const real *outv,
                           const real *outF,
                           const real *outC,
                           const real *outP,
                           const real *outgrid,
                           real *grad_inx,
                           real *grad_inv,
                           real *grad_inF,
                           real *grad_inC,
                           const real *grad_outx,
                           const real *grad_outv,
                           const real *grad_outF,
                           const real *grad_outC,
                           const real *grad_outP,
                           const real *grad_outgrid) {
  //printf("MPM_grad Kernel launch~~\n");
  auto current = new State(res, num_particles, dx, dt, gravity, (real *)inx,
                           (real *)inv, (real *)inF, (real *)inC, (real *)outP,
                           (real *)outgrid, grad_inx, grad_inv, grad_inF,
                           grad_inC, (real *)grad_outP, (real *)grad_outgrid);
  auto next = new State(res, num_particles, dx, dt, gravity, (real *)outx,
                            (real *)outv, (real *)outF, (real *)outC, NULL,
                            NULL, (real *)grad_outx, (real *)grad_outv,
                            (real *)grad_outF, (real *)grad_outC, NULL, NULL);
  backward<dim>(*current, *next);
  //printf("MPM_grad Kernel Finish~~\n");
}

void backward_mpm3d_state(void *state_, void *next_state_) {
  State *state = reinterpret_cast<State *>(state_);
  State *next_state = reinterpret_cast<State *>(next_state_);
  backward<dim>(*state, *next_state);
}

void set_grad_loss(void *state_) {
  State *state = reinterpret_cast<State *>(state_);
  state->clear_gradients();
  int num_particles = state->num_particles;
  std::vector<float> grad_x_host(num_particles * dim);
  for (int i = 0; i < num_particles; i++) {
    grad_x_host[i] = 1.0f / num_particles;
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
