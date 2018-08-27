using real = float;

class Vector {
public:
  static constexpr int dim = 3;

  real d[dim];

  Vector(real x = 0) {
    for (int i = 0; i < dim; i++) {
      d[i] = x;
    }
  }
};

class Matrix {
public:
  static constexpr int dim = 3;

  real d[dim][dim];

  Matrix() {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d[i][j] = 0;
      }
    }
  }
};
