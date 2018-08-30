#pragma once


struct StateBase {
  using real = float;

  static constexpr int dim = 3;
  int num_particles;
  real *x_storage;
  real *v_storage;
  real *F_storage;
  real *C_storage;

  int res[3];
  real *grid_storage;
  int num_cells;

  real gravity[3];
  real dx, inv_dx;
  real dt;
};

void advance(StateBase &state);
