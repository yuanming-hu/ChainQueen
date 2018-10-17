#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "config.h"

using namespace tensorflow;

/*
    Register Inc operation
*/

REGISTER_OP("Inc")
    .Input("x: float")     //(batch_size, dim, particles)
    .Output("x_out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      return Status::OK();
    });

void IncKernelLauncher(const float *inx, float *outx);

class IncOpGPU : public OpKernel {
 public:
  explicit IncOpGPU(OpKernelConstruction *context) : OpKernel(context) {
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &inx = context->input(0);
    const TensorShape &x_shape = inx.shape();

    Tensor *outx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &outx));

    auto f_inx = inx.flat<float>();
    auto f_outx = outx->template flat<float>();
    IncKernelLauncher(f_inx.data(), f_outx.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Inc").Device(DEVICE_GPU), IncOpGPU);
