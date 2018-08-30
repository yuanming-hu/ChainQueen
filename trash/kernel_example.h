// kernel_example.h
#pragma once

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device &d, int size, const T *in, T *out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice &d, int size, const T *in, T *out);
};
#endif
