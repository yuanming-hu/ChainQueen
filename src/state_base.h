#pragma once

struct StateBase {
  using real = float;

  static constexpr int dim = 3;
  int num_particles;
  int res[3];

  real V = 10;     // TODO: variable vol
  real E = 5;      // TODO: variable E
  real nu = 0.3;   // TODO: variable nu
  real m_p = 100;  // TODO: variable m_p
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));

  real *x_storage;
  real *v_storage;
  real *F_storage;
  real *P_storage;
  real *C_storage;
  real *grid_storage;

  real *grad_x_storage;
  real *grad_v_storage;
  real *grad_F_storage;
  real *grad_P_storage;
  real *grad_C_storage;
  real *grad_grid_storage;

  int num_cells;

  real gravity[3];
  real dx, inv_dx, invD;
  real dt;
};

void advance(StateBase &state);
