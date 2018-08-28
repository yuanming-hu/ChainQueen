#pragma once

#include "linalg.h"

struct State {
  Vector *x;
  Vector *v;
  Matrix *F;
  Matrix *C;

  Vector gravity;
  real dx;
  real dt;
};

void advance(State &state);
