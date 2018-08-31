#include "scatter_inc.h"

REGISTER_OP("ScatterInc")
        .Input("base: float")
        .Input("index: int32")
        .Input("delta: float")
        .Output("updated: float")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            shape_inference::ShapeHandle base_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &base_shape));
            c->set_output(0, base_shape);
            return Status::OK();
        });

class ScatterIncOp : public OpKernel {
public:
    explicit ScatterIncOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    void Compute(OpKernelContext *context) override {
        DCHECK_EQ(3, context->num_inputs());

        const Tensor &base = context->input(0);
        const Tensor &index = context->input(1);
        const Tensor &delta = context->input(2);

        const TensorShape &base_shape = base.shape();
        const TensorShape &index_shape = index.shape();

        DCHECK_EQ(base_shape.dims(), 2);

        Tensor *updated = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, base_shape, &updated));

        auto base_tensor = base.tensor<float, 2>();
        auto index_tensor = index.tensor<int, 2>();
        auto delta_tensor = delta.tensor<float, 2>();

        auto updated_tensor = updated->tensor<float, 2>();

        for (int t = 0; t < updated->shape().dim_size(0); t++) {
            int n = base.shape().dim_size(1);
            int m = index.shape().dim_size(1);
            for (int i = 0; i < n; i++) {
                updated_tensor(t, i) = base_tensor(t, i);
            }
            for (int i = 0; i < m; i++) {
                updated_tensor(t, index_tensor(t, i)) += delta_tensor(t, i);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ScatterInc").Device(DEVICE_CPU), ScatterIncOp);
