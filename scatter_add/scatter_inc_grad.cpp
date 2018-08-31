#include "scatter_inc.h"

REGISTER_OP("ScatterIncGrad")
        .Input("grad_updated: float32")
        .Input("index: int32")
        .Output("grad_delta: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            shape_inference::ShapeHandle index_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &index_shape));
            c->set_output(0, index_shape);
            return Status::OK();
        });
;

class ScatterIncGradOp : public OpKernel {
public:
    explicit ScatterIncGradOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    void Compute(OpKernelContext *context) override {
        DCHECK_EQ(3, context->num_inputs());

        const Tensor &grad_updated = context->input(0);
        const Tensor &index = context->input(1);

        const TensorShape &index_shape = index.shape();

        Tensor *grad_delta = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, index_shape, &grad_delta));

        auto grad_updated_tensor = grad_updated.tensor<float, 2>();
        auto index_tensor = index.tensor<int, 2>();

        auto grad_delta_tensor = grad_delta->tensor<float, 2>();

        for (int t = 0; t < grad_delta->shape().dim_size(0); t++) {
            int m = index.shape().dim_size(1);
            for (int i = 0; i < m; i++) {
                grad_delta_tensor(t, i) = grad_updated_tensor(t, index_tensor(t, i));
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ScatterIncGrad").Device(DEVICE_CPU), ScatterIncGradOp);

