#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "config.h"

using namespace tensorflow;

constexpr int dim = TC_DMPM_DIM;

/*
    Register MPM operation
*/

REGISTER_OP("Mpm")
    .Input("position: float")     //(batch_size, dim, particles)
    .Input("velocity: float")     //(batch_size, dim, particles)
    .Input("affine: float")       //(batch_size, dim, dim, particles)
    .Input("deformation: float")  //(batch_size, dim, dim, particles
    .Attr("dt: float = 0.01")
    .Attr("dx: float = 0.01")
    .Output("position_out: float")
    .Output("velocity_out: float")
    .Output("affine_out: float")
    .Output("deformation_out: float")
    .Output("poly_out: float")  //(batch_size, dim, dim, particles)
    .Output("grid_out: float")  //(batch_size, dim + 1, num_cells)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {

      shape_inference::ShapeHandle x_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x_shape));
      shape_inference::ShapeHandle v_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &v_shape));
      shape_inference::ShapeHandle F_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &F_shape));
      shape_inference::ShapeHandle C_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &C_shape));

      shape_inference::DimensionHandle temp;

      shape_inference::DimensionHandle batch_size = c->Dim(x_shape, 0);
      shape_inference::DimensionHandle batch_sizev = c->Dim(v_shape, 0);
      shape_inference::DimensionHandle batch_sizeF = c->Dim(F_shape, 0);
      shape_inference::DimensionHandle batch_sizeC = c->Dim(C_shape, 0);
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizev, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeF, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeC, &temp));

      shape_inference::DimensionHandle dim = c->Dim(x_shape, 1);
      shape_inference::DimensionHandle dimv = c->Dim(v_shape, 1);
      shape_inference::DimensionHandle dimF1 = c->Dim(F_shape, 1);
      shape_inference::DimensionHandle dimF2 = c->Dim(F_shape, 2);
      shape_inference::DimensionHandle dimC1 = c->Dim(C_shape, 1);
      shape_inference::DimensionHandle dimC2 = c->Dim(C_shape, 2);
      TF_RETURN_IF_ERROR(c->Merge(dim, dimv, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimF1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimF2, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimC1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimC2, &temp));

      shape_inference::DimensionHandle particle = c->Dim(x_shape, 2);
      shape_inference::DimensionHandle particlev = c->Dim(v_shape, 2);
      shape_inference::DimensionHandle particleF = c->Dim(F_shape, 3);
      shape_inference::DimensionHandle particleC = c->Dim(C_shape, 3);
      TF_RETURN_IF_ERROR(c->Merge(particle, particlev, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleF, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleC, &temp));

      c->set_output(0, x_shape);
      c->set_output(1, v_shape);
      c->set_output(2, F_shape);
      c->set_output(3, C_shape);
      c->set_output(4, C_shape);
      auto dim_ = *((int *)dim.Handle());
      // printf("dim %d\n", dim_);
      int res[3];
      int num_cells = 1;
      for(int i = 0; i < dim_; i++) {
        res[i] = 100;
        num_cells *= res[i];
      }
      std::vector<shape_inference::DimensionHandle> new_shape;
      new_shape.clear();
      new_shape.push_back(batch_size);
      new_shape.push_back(
          c->MakeDim(shape_inference::DimensionOrConstant(num_cells)));
      new_shape.push_back(
          c->MakeDim(shape_inference::DimensionOrConstant(dim_ + 1)));
      c->set_output(5, c->MakeShape(new_shape));

      return Status::OK();
    });

/*
    MPM Operation GPU
*/

void MPMKernelLauncher(int dim,
                       int *res,
                       int num_particles,
                       float dx,
                       float dt,
                       float *gravity,
                       const float *inx,
                       const float *inv,
                       const float *inF,
                       const float *inC,
                       float *outx,
                       float *outv,
                       float *outF,
                       float *outC,
                       float *outP,
                       float *outgrid);

class MPMOpGPU : public OpKernel {
 private:
  float dt_;
  float dx_;
 public:
  explicit MPMOpGPU(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("dt", &dt_));
    OP_REQUIRES(context, dt_ > 0,
                errors::InvalidArgument("Need dt > 0, got ", dt_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dx", &dx_));
    OP_REQUIRES(context, dx_ > 0,
                errors::InvalidArgument("Need dx > 0, got ", dx_));
  }

  void Compute(OpKernelContext *context) override {
    // get the x
    const Tensor &inx = context->input(0);

    // get the v tensor
    const Tensor &inv = context->input(1);

    // get the F tensor
    const Tensor &inF = context->input(2);

    // get the C tensor
    const Tensor &inC = context->input(3);

    // check shapes of input and weights
    const TensorShape &x_shape = inx.shape();
    const TensorShape &v_shape = inv.shape();
    const TensorShape &F_shape = inF.shape();
    const TensorShape &C_shape = inC.shape();
    TensorShape P_shape = inC.shape();
    TensorShape grid_shape = inx.shape();

    // Check that inputs' dimensional
    DCHECK_EQ(x_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(F_shape.dims(), 4);
    DCHECK_EQ(C_shape.dims(), 4);

    const int batch_size = x_shape.dim_size(0);
    // printf("batch_size %d\n", batch_size);

    const int dim = x_shape.dim_size(1);
    // printf("dim %d\n", dim);
    int res[dim];
    float gravity[dim];
    int num_cells = 1;
    for (int i = 0; i < dim; i++) {
      res[i] = 100;
      num_cells *= res[i];
      gravity[i] = 0;
    }
    int grid_shape2 = dim + 1;
    // printf("MPMOpGPU\n");
    // float dx = 1.0f / res[0];

    const int particles = x_shape.dim_size(2);
    // printf("particles %d\n", particles);

    // Check input batch_size
    DCHECK_EQ(batch_size, v_shape.dim_size(0));
    DCHECK_EQ(batch_size, F_shape.dim_size(0));
    DCHECK_EQ(batch_size, C_shape.dim_size(0));

    // Check input dim
    DCHECK_EQ(dim, v_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(2));
    DCHECK_EQ(dim, C_shape.dim_size(1));
    DCHECK_EQ(dim, C_shape.dim_size(2));

    // Check input particles
    DCHECK_EQ(particles, v_shape.dim_size(2));
    DCHECK_EQ(particles, F_shape.dim_size(3));
    DCHECK_EQ(particles, C_shape.dim_size(3));

    // create output tensor
    Tensor *outx = NULL;
    Tensor *outv = NULL;
    Tensor *outF = NULL;
    Tensor *outC = NULL;
    Tensor *outP = NULL;
    Tensor *outgrid = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &outx));
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &outv));
    OP_REQUIRES_OK(context, context->allocate_output(2, F_shape, &outF));
    OP_REQUIRES_OK(context, context->allocate_output(3, C_shape, &outC));
    OP_REQUIRES_OK(context, context->allocate_output(4, P_shape, &outP));
    grid_shape.set_dim(1, num_cells);
    grid_shape.set_dim(2, grid_shape2);
    OP_REQUIRES_OK(context, context->allocate_output(5, grid_shape, &outgrid));

    auto f_inx = inx.flat<float>();
    auto f_inv = inv.flat<float>();
    auto f_inF = inF.flat<float>();
    auto f_inC = inC.flat<float>();
    auto f_outx = outx->template flat<float>();
    auto f_outv = outv->template flat<float>();
    auto f_outF = outF->template flat<float>();
    auto f_outC = outC->template flat<float>();
    auto f_outP = outP->template flat<float>();
    auto f_outgrid = outgrid->template flat<float>();

    MPMKernelLauncher(dim, res, particles, dx_, dt_, gravity, f_inx.data(),
                      f_inv.data(), f_inF.data(), f_inC.data(), f_outx.data(),
                      f_outv.data(), f_outF.data(), f_outC.data(),
                      f_outP.data(), f_outgrid.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Mpm").Device(DEVICE_GPU), MPMOpGPU);
