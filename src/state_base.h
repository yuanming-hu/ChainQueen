#pragma once


struct StateBase {
  using real = float;

  static constexpr int dim = 3;
  int num_particles;
  int res[3];
  real *x_storage;
  real *v_storage;
  real *F_storage;
  real *C_storage;
  real *grid_storage;

  real *grad_x_storage;
  real *grad_v_storage;
  real *grad_F_storage;
  real *grad_C_storage;
  real *grad_grid_storage;

  int num_cells;

  real gravity[3];
  real dx, inv_dx;
  real dt;
};

void advance(StateBase &state);
