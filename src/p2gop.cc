#if (0)
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("P2g")
  .Input("position: float")         //(batch_size, dim, particles)
  .Input("velocity: float")         //(batch_size, dim, particles)
  .Input("affine: float")           //(batch_size, dim, dim, particles)
  .Input("deformation: float")      //(batch_size, dim, dim, particles)
  .Output("poly_out: float")        //(batch_size, dim, dim, particles)
  .Output("grid_out: float")        //(batch_size, dim + 1, num_cells)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    int res[3] = {100, 100, 100};
    int num_cells = res[0] * res[1] * res[2];
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

    c->set_output(0, C_shape);
    std::vector<shape_inference::DimensionHandle> new_shape;
    new_shape.clear();
    new_shape.push_back(batch_size);
    new_shape.push_back(
                     c->MakeDim(shape_inference::DimensionOrConstant(num_cells)));
    new_shape.push_back(c->MakeDim(shape_inference::DimensionOrConstant(4)));
    c->set_output(1, c->MakeShape(new_shape));

    
    return Status::OK();
  });


void P2GKernelLauncher(
    int res[3], int num_particles, float dx, float dt, float gravity[3],
    const float *inx, const float *inv, const float *inF, const float *inC,
    float *outP, float *outgrid);

class P2GOpGPU : public OpKernel {
public:
  explicit P2GOpGPU(OpKernelConstruction* context) : OpKernel(context) {
  }
  
  void Compute(OpKernelContext* context) override {
    int res[3] = {100, 100, 100};
    float gravity[3] = {0, -0, 0};
    float dx = 1.0f / res[0];
    float dt = 1e-2f;
    int num_cells = res[0] * res[1] * res[2];
    //printf("MPMOpGPU\n");

    // get the x
    const Tensor& inx = context->input(0);
    
    // get the v tensor
    const Tensor& inv = context->input(1);
      
    // get the F tensor
    const Tensor& inF = context->input(2);
      
    // get the C tensor
    const Tensor& inC = context->input(3);
    
    // check shapes of input and weights
    const TensorShape& x_shape = inx.shape();
    const TensorShape& v_shape = inv.shape();
    const TensorShape& F_shape = inF.shape();
    const TensorShape& C_shape = inC.shape();
    TensorShape P_shape = inC.shape();
    TensorShape grid_shape = inx.shape();
    
    //Check that inputs' dimensional
    DCHECK_EQ(x_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(F_shape.dims(), 4);
    DCHECK_EQ(C_shape.dims(), 4);

    const int batch_size = x_shape.dim_size(0);
    //printf("batch_size %d\n", batch_size);

    const int dim = x_shape.dim_size(1);
    //printf("dim %d\n", dim);

    const int particles = x_shape.dim_size(2);
    //printf("particles %d\n", particles);

    //Check input batch_size
    DCHECK_EQ(batch_size, v_shape.dim_size(0));
    DCHECK_EQ(batch_size, F_shape.dim_size(0));
    DCHECK_EQ(batch_size, C_shape.dim_size(0));
    
    //Check input dim
    DCHECK_EQ(dim, v_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(2));
    DCHECK_EQ(dim, C_shape.dim_size(1));
    DCHECK_EQ(dim, C_shape.dim_size(2));
    
    //Check input particles
    DCHECK_EQ(particles, v_shape.dim_size(2));
    DCHECK_EQ(particles, F_shape.dim_size(3));
    DCHECK_EQ(particles, C_shape.dim_size(3));
            
    // create output tensor
    Tensor* outP = NULL;
    Tensor* outgrid = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, P_shape, &outP));
    grid_shape.set_dim(1, num_cells);
    grid_shape.set_dim(2, dim + 1);
    OP_REQUIRES_OK(context, context->allocate_output(1, grid_shape, &outgrid));
    
    auto f_inx = inx.flat<float>();
    auto f_inv = inv.flat<float>();
    auto f_inF = inF.flat<float>();
    auto f_inC = inC.flat<float>();
    auto f_outP = outP->template flat<float>();
    auto f_outgrid = outgrid->template flat<float>();
    

    P2GKernelLauncher(
        res, particles, dx, dt, gravity,
        f_inx.data(),
        f_inv.data(),
        f_inF.data(),
        f_inC.data(),
        f_outP.data(),
        f_outgrid.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("P2g").Device(DEVICE_GPU), P2GOpGPU);
#endif
