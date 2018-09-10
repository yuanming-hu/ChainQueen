#pragma once

struct StateBase {
  using real = float;

  static constexpr int dim = 3;
  int num_particles;
  int res[3];

  real V_p = 10;   // TODO: variable vol
  real m_p = 100;  // TODO: variable m_p
  real E = 500;      // TODO: variable E
  real nu = 0.3;   // TODO: variable nu
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));

  StateBase() {
    set(10, 100, 5, 0.3);
  }

  void set(real V_p, real m_p, real E, real nu) {
    this->V_p = V_p;
    this->m_p = m_p;
    this->E = E;
    this->nu = nu;
    this->mu = E / (2 * (1 + nu));
    this->lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  }

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
