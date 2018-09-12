#include "kernels.h"
#include "linalg.h"
#include "state.cuh"
#include <cstdio>
#include <vector>

// Gather data from SOA
// Ensure coalesced global memory access

// Do not consider sorting for now. Use atomics instead.

// One particle per thread
template <int dim>
__global__ void P2G(TState<dim> state) {
  // constexpr int scratch_size = 8;
  //__shared__ real scratch[dim + 1][scratch_size][scratch_size][scratch_size];

  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  real dt = state.dt;

  auto x = state.get_x(part_id);
  auto v = state.get_v(part_id);
  auto F = state.get_F(part_id);
  auto C = state.get_C(part_id);

  TransferCommon<dim> tc(state, x);

  auto A = state.get_A(part_id);

  // Fixed corotated
  auto P = PK1(state.mu, state.lambda, F) + F * A;
  state.set_P(part_id, P);
  auto stress = -state.invD * dt * state.V_p * P * transposed(F);

  auto affine =
      real(mpm_enalbe_force) * stress + real(mpm_enalbe_apic) * state.m_p * C;

#pragma unroll
  for (int i = 0; i < kernel_volume<dim>(); i++) {
    auto dpos = tc.dpos(i);

    real contrib[dim + 1];

    auto tmp = affine * dpos + state.m_p * v;

    auto w = tc.w(i);
    for (int d = 0; d < dim; d++) {
      contrib[d] = tmp[d] * w;
    }
    contrib[dim] = state.m_p * w;

    auto node = state.grid_node(tc.base_coord + offset_from_scalar<dim>(i));
    for (int p = 0; p < dim + 1; p++) {
      atomicAdd(&node[p], contrib[p]);
    }
  }
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      //printf("forward A[%d][%d] %f\n", i, j, A[i][j]);
    }
  }
}

template <int dim>
__global__ void normalize_grid(TState<dim> state) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int boundary = 3;
  if (id < state.num_cells) {
    auto node = state.grid_node(id);
    if (node[dim] > 0) {
      real inv_m = 1.0f / node[dim];
      for (int i = 0; i < dim; i++) {
        node[i] *= inv_m;
      }
      for (int i = 0; i < dim; i++) {
        node[i] += state.gravity[i] * state.dt;
      }
      /*
      int x = id / (state.res[1] * state.res[2]),
          y = id / state.res[2] % state.res[1], z = id % state.res[2];
      if (x < boundary || y < boundary || y < boundary ||
          x + boundary >= state.res[0] || y + boundary >= state.res[1] ||
          z + boundary >= state.res[2]) {
        node[1] = max(0.0f, node[1]);
      }
      */
      int y;
      if (dim == 3) {
        y = id / state.res[2] % state.res[1];
      } else {
        y = id % state.res[1];
      }
      if (y < boundary) {
        node[1] = max(0.0f, node[1]);
      }
    }
  }
}

template <int dim>
__global__ void G2P(TState<dim> state, TState<dim> next_state) {
  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  real dt = state.dt;
  auto x = state.get_x(part_id);
  typename TState<dim>::Vector v;
  auto F = state.get_F(part_id);
  typename TState<dim>::Matrix C;

  TransferCommon<dim> tc(state, x);

  for (int i = 0; i < kernel_volume<dim>(); i++) {
    auto dpos = tc.dpos(i);
    auto node = state.grid_node(tc.base_coord + offset_from_scalar<dim>(i));
    auto node_v = TState<dim>::Vector(node);
    auto w = tc.w(i);
    v = v + w * node_v;
    C = C + TState<dim>::Matrix::outer_product(w * node_v, state.invD * dpos);
  }
  next_state.set_x(part_id, x + state.dt * v);
  next_state.set_v(part_id, v);
  next_state.set_F(part_id, (typename TState<dim>::Matrix(1) + dt * C) * F);
  next_state.set_C(part_id, C);
}

template <int dim>
void advance(TState<dim> &state, TState<dim> &new_state) {
  cudaMemset(state.grid_storage, 0,
             state.num_cells * (state.dim + 1) * sizeof(real));
  int num_blocks =
      (state.num_particles + particle_block_dim - 1) / particle_block_dim;
  P2G<dim><<<num_blocks, particle_block_dim>>>(state);

  auto err = cudaThreadSynchronize();
  if (err) {
    printf("Launch: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  normalize_grid<dim>
      <<<(state.grid_size() + grid_block_dim - 1) / grid_block_dim,
         grid_block_dim>>>(state);
  G2P<dim><<<num_blocks, particle_block_dim>>>(state, new_state);
}

// compability
void MPMKernelLauncher(int dim_,
                       int *res,
                       int num_particles,
                       real dx,
                       real dt,
                       real E,
                       real nu,
                       real m_p,
                       real V_p,
                       real *gravity,
                       const real *inx,
                       const real *inv,
                       const real *inF,
                       const real *inC,
                       const real *inA,
                       real *outx,
                       real *outv,
                       real *outF,
                       real *outC,
                       real *outP,
                       real *outgrid) {
  if (dim_ == 3) {
    constexpr int dim = 3;
    auto instate =
        new TState<dim>(res, num_particles, dx, dt, gravity, (real *)inx,
                        (real *)inv, (real *)inF, (real *)inC, (real *)inA,outP,  outgrid);
    instate->set(V_p, m_p, E, nu);
    auto outstate = new TState<dim>(res, num_particles, dx, dt, gravity, outx,
                                    outv, outF, outC, nullptr, nullptr, nullptr);
    outstate->set(V_p, m_p, E, nu);
    advance<dim>(*instate, *outstate);
  } else {
    constexpr int dim = 2;
    auto instate =
        new TState<dim>(res, num_particles, dx, dt, gravity, (real *)inx,
                        (real *)inv, (real *)inF, (real *)inC,(real *)inA, outP,  outgrid);
    instate->set(V_p, m_p, E, nu);
    auto outstate = new TState<dim>(res, num_particles, dx, dt, gravity, outx,
                                    outv, outF, outC, nullptr, nullptr, nullptr);
    outstate->set(V_p, m_p, E, nu);
    advance<dim>(*instate, *outstate);
  }
}
void P2GKernelLauncher(int dim_,
                       int *res,
                       int num_particles,
                       real dx,
                       real dt,
                       real E,
                       real nu,
                       real m_p,
                       real V_p,
                       real *gravity,
                       const real *inx,
                       const real *inv,
                       const real *inF,
                       const real *inC,
                       const real *inA,
                       real *outP,
                       real *outgrid) {
  if (dim_ == 3) {
    constexpr int dim = 3;
    auto state =
        new TState<dim>(res, num_particles, dx, dt, gravity, (real *)inx,
                        (real *)inv, (real *)inF, (real *)inC, (real *)inA, outP, outgrid);
    int num_blocks =
      (num_particles + particle_block_dim - 1) / particle_block_dim;
    cudaMemset(outgrid, 0, state->num_cells * (dim + 1) * sizeof(real));
    state->set(V_p, m_p, E, nu);
    P2G<dim><<<num_blocks, particle_block_dim>>>(*state);
  } else {
    constexpr int dim = 2;
    auto state =
        new TState<dim>(res, num_particles, dx, dt, gravity, (real *)inx,
                        (real *)inv, (real *)inF, (real *)inC, (real *)inA, outP, outgrid);
    int num_blocks =
      (num_particles + particle_block_dim - 1) / particle_block_dim;
    cudaMemset(outgrid, 0, state->num_cells * (dim + 1) * sizeof(real));
    state->set(V_p, m_p, E, nu);
    P2G<dim><<<num_blocks, particle_block_dim>>>(*state);
  }
}


template <int dim>
void initialize_mpm_state(int *res,
                          int num_particles,
                          float *gravity,
                          void *&state_,
                          float dx,
                          float dt,
                          float *initial_positions) {
  // State(int res[dim], int num_particles, real dx, real dt, real
  auto state = new TState<dim>(res, num_particles, dx, dt, gravity);
  state_ = state;
  cudaMemcpy(state->x_storage, initial_positions,
             sizeof(TVector<real, dim>) * num_particles,
             cudaMemcpyHostToDevice);
}

template <int dim>
void forward_mpm_state(void *state_, void *new_state_) {
  auto *state = reinterpret_cast<TState<dim> *>(state_);
  auto *new_state = reinterpret_cast<TState<dim> *>(new_state_);
  advance<dim>(*state, *new_state);
}

template void initialize_mpm_state<2>(int *res,
                                      int num_particles,
                                      float *gravity,
                                      void *&state_,
                                      float dx,
                                      float dt,
                                      float *initial_positions);
template void initialize_mpm_state<3>(int *res,
                                      int num_particles,
                                      float *gravity,
                                      void *&state_,
                                      float dx,
                                      float dt,
                                      float *initial_positions);
template void forward_mpm_state<2>(void *, void *);
template void forward_mpm_state<3>(void *, void *);
/*
constexpr int dim = 3;
void P2GKernelLauncher(int res[dim],
                       int num_particles,
                       real dx,
                       real dt,
                       real gravity[dim],
                       const real *inx,
                       const real *inv,
                       const real *inF,
                       const real *inC,
                       real *outP,
                       real *outgrid) {
  auto state =
      new TState<dim>(res, num_particles, dx, dt, gravity, (real *)inx,
                      (real *)inv, (real *)inF, (real *)inC, outP, outgrid);
  cudaMemset(outgrid, 0, state->num_cells * (dim + 1) * sizeof(real));
  int num_blocks =
      (num_particles + particle_block_dim - 1) / particle_block_dim;
  P2G<dim><<<num_blocks, particle_block_dim>>>(*state);
}

void G2PKernelLauncher(int res[dim],
                       int num_particles,
                       real dx,
                       real dt,
                       real gravity[dim],
                       const real *inx,
                       const real *inv,
                       const real *inF,
                       const real *inC,
                       const real *inP,
                       const real *ingrid,
                       real *outx,
                       real *outv,
                       real *outF,
                       real *outC) {
  auto instate = new TState<dim>(res, num_particles, dx, dt, gravity,
                                 (real *)inx, (real *)inv, (real *)inF,
                                 (real *)inC, (real *)inP, (real *)ingrid);
  auto outstate = new TState<dim>(res, num_particles, dx, dt, gravity, outx,
                                  outv, outF, outC, nullptr, nullptr);
  int num_blocks =
      (num_particles + particle_block_dim - 1) / particle_block_dim;
  G2P<dim><<<num_blocks, particle_block_dim>>>(*instate, *outstate);
}
*/
