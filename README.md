# The (Legacy) ChainQueen Differentiable MPM Solver

## Note: this is an old (Oct 2018) version of ChainQueen.

## Installing the CUDA solver

- Install `taichi` by executing:
  ```
  wget https://raw.githubusercontent.com/yuanming-hu/taichi/master/install.py
  python3 install.py
  ```
- Make sure you are using `gcc-6`. If not, please install `export CXX=g++-6 CC=gcc-6`.
- Put this repo in `taichi/projects/`
- execute ```ti build```

## Discretization Cheatsheet
(Assuming quadratic B-spline)
<img src="/data/images/comparison.jpg" with="1000">
