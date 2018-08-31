#include "scatter_inc.h"

REGISTER_OP("ScatterIncGradGrad")
        .Input("grad_output: float32")
        .Input("index: int32")
        .Input("output_shape_dummy: float32")
        .Output("grad_input: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            shape_inference::ShapeHandle output_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &output_shape));
            c->set_output(0, output_shape);
            return Status::OK();
        });
;

class ScatterIncGradGradOp : public OpKernel {
public:
    explicit ScatterIncGradGradOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    void Compute(OpKernelContext *context) override {
        DCHECK_EQ(3, context->num_inputs());

        const Tensor &grad_y = context->input(0);
        const Tensor &index = context->input(1);
        const Tensor &output_dummy = context->input(2);

        const TensorShape &index_shape = index.shape();
        const TensorShape &output_shape = output_dummy.shape();

        Tensor *grad_x = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &grad_x));

        auto grad_y_tensor = grad_y.tensor<float, 2>();
        auto grad_x_tensor = grad_x->tensor<float, 2>();
        auto index_tensor = index.tensor<int, 2>();

        for (int t = 0; t < grad_y.shape().dim_size(0); t++) {
            int n = output_dummy.shape().dim_size(1);
            int m = index.shape().dim_size(1);
            for (int i = 0; i < n; i++) {
                grad_x_tensor(t, i) = 0;
            }
            for (int i = 0; i < m; i++) {
                grad_x_tensor(t, index_tensor(t, i)) += grad_y_tensor(t, i);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ScatterIncGradGrad").Device(DEVICE_CPU), ScatterIncGradGradOp);

