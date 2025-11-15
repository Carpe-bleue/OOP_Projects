// ...existing code...
#include <assert.h>
#include "evaluation.h"
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <limits> // <- added


evaluation::evaluation(const std::vector<expression> &exprs)
    : exprs_(exprs), result_(0), result_tensor_()
{
    for (const auto &expr : exprs_)
    {
        printf("Expression id: %d, operator: %s, operands: ", expr.expr_id, expr.op_type.c_str());
        for (size_t i = 0; i < expr.inputs.size(); ++i)
        {
            printf("%d", expr.inputs[i]);
            if (i + 1 < expr.inputs.size())
                printf(", ");
        }
        printf("\n");
    }
    for (size_t i = 0; i < exprs_.size(); ++i)
    {
        id_to_index[exprs_[i].expr_id] = i;
    }
}

void evaluation::add_kwargs_double(const char *key, double value)
{
    printf("Key: %s, Value: %f\n", key, value);
    input_values[key] = value;
}

void evaluation::add_kwargs_ndarray(const char *key, int dim, size_t shape[], double data[])
{
    size_t total = 1;
    printf("Key: %s, Dim: %d (", key, dim);
    for (int i = 0; i != dim; ++i)
    {
        printf("%zu,", shape[i]);
        total *= shape[i];
    }
    printf(") Data pointer: %p\n", data);
    input_shapes[key] = std::vector<size_t>(shape, shape + dim);
    input_vectors[key] = std::vector<double>(data, data + total);
}

tensor evaluation::eval_exprs(int idx) const
{
    const expression &expr = exprs_[idx];
    printf("EVAL: expr_id=%d, op_type=%s\n", expr.expr_id, expr.op_type.c_str());

    for (size_t i = 0; i < expr.inputs.size(); ++i)
    {
        int input_idx = id_to_index.at(expr.inputs[i]);
        const expression &input_expr = exprs_[input_idx];
        printf("  Input %zu: expr_id=%d, op_type=%s\n", i, input_expr.expr_id, input_expr.op_type.c_str());
    }

    if (expr.op_type == "Input" || expr.op_type == "Input2d")
    {
        auto it_scalar = input_values.find(expr.op_name);
        if (it_scalar != input_values.end())
        {
            return tensor(it_scalar->second);
        }
        auto it_vec = input_vectors.find(expr.op_name);
        auto it_shape = input_shapes.find(expr.op_name);
        if (it_vec != input_vectors.end() && it_shape != input_shapes.end())
        {
            // handle normal Input (leave shape/data as provided)
            if (expr.op_type == "Input")
            {
                size_t total_size = 1;
                for (size_t dim : it_shape->second) total_size *= dim;
                if (it_vec->second.size() != total_size)
                    throw std::runtime_error("Input vector size does not match shape for: " + expr.op_name);
                return tensor(it_shape->second, it_vec->second);
            }

            // Input2d: grader/python provides data in (N,H,W,C). Golden expects (N,C,H,W).
            // Convert data order from N,H,W,C -> N,C,H,W
            const std::vector<size_t> &s = it_shape->second;
            if (s.size() != 4)
                throw std::runtime_error("Input2d expects 4D data");
            size_t N = s[0], H = s[1], W = s[2], C = s[3];
            size_t total = N * H * W * C;
            if (it_vec->second.size() != total)
                throw std::runtime_error("Input2d data size mismatch for: " + expr.op_name);

            std::vector<size_t> shape_vec = {N, C, H, W};
            std::vector<double> data_vec;
            data_vec.reserve(total);
            // iterate in N,C,H,W order, reading from original N,H,W,C layout
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx_in = ((n * H + h) * W + w) * C + c; // N,H,W,C layout
                            data_vec.push_back(it_vec->second[idx_in]);
                        }
                    }
                }
            }
            return tensor(shape_vec, data_vec);
        }
        throw std::runtime_error("Missing input for: " + expr.op_name);
    }

    if (expr.op_type == "Const")
    {
        auto it = expr.params_double.find("value");
        if (it != expr.params_double.end())
        {
            printf("Const scalar: %f\n", it->second);
            return tensor(it->second);
        }
        auto it_nd = expr.params_ndarray.find("value");
        if (it_nd != expr.params_ndarray.end())
        {
            printf("Const tensor: shape = ");
            for (size_t s : it_nd->second.shape)
                printf("%zu ", s);
            printf("\n");
            return tensor(it_nd->second.shape, it_nd->second.data);
        }
        printf("Const value missing!\n");
        throw std::runtime_error("Missing const value");
    }

    if (expr.op_type == "Add" || expr.op_type == "Sub" || expr.op_type == "Mul")
    {
        // Get input tensors
        int idx_a = id_to_index.at(expr.inputs[0]);
        int idx_b = id_to_index.at(expr.inputs[1]);
        tensor a = eval_exprs(idx_a);
        tensor b = eval_exprs(idx_b);

        // Scalar case
        if (a.data().size() == 1 && b.data().size() == 1)
        {
            double result = 0.0;
            if (expr.op_type == "Add")
                result = a.data()[0] + b.data()[0];
            if (expr.op_type == "Sub")
                result = a.data()[0] - b.data()[0];
            if (expr.op_type == "Mul")
                result = a.data()[0] * b.data()[0];
            return tensor(result);
        }

        // Elementwise tensor case (same shape)
        if (a.shape() == b.shape())
        {
            std::vector<double> out(a.data().size());
            for (size_t i = 0; i < out.size(); ++i)
            {
                if (expr.op_type == "Add")
                    out[i] = a.data()[i] + b.data()[i];
                if (expr.op_type == "Sub")
                    out[i] = a.data()[i] - b.data()[i];
                if (expr.op_type == "Mul")
                    out[i] = a.data()[i] * b.data()[i];
            }
            return tensor(a.shape(), out);
        }

        // Scalar-tensor broadcasting
        if (a.data().size() == 1)
        {
            std::vector<double> out(b.data().size());
            for (size_t i = 0; i < out.size(); ++i)
            {
                if (expr.op_type == "Add")
                    out[i] = a.data()[0] + b.data()[i];
                if (expr.op_type == "Sub")
                    out[i] = a.data()[0] - b.data()[i];
                if (expr.op_type == "Mul")
                    out[i] = a.data()[0] * b.data()[i];
            }
            return tensor(b.shape(), out);
        }
        if (b.data().size() == 1)
        {
            std::vector<double> out(a.data().size());
            for (size_t i = 0; i < out.size(); ++i)
            {
                if (expr.op_type == "Add")
                    out[i] = a.data()[i] + b.data()[0];
                if (expr.op_type == "Sub")
                    out[i] = a.data()[i] - b.data()[0];
                if (expr.op_type == "Mul")
                    out[i] = a.data()[i] * b.data()[0];
            }
            return tensor(a.shape(), out);
        }

        // matrix multiply case
        if (a.shape().size() == 2 && b.shape().size() == 2 &&
            a.shape()[1] == b.shape()[0])
        {
            size_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
            std::vector<double> out(m * n, 0.0);
            for (size_t i = 0; i < m; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    for (size_t l = 0; l < k; ++l)
                    {
                        out[i * n + j] += a.data()[i * k + l] * b.data()[l * n + j];
                    }
                }
            }
            std::vector<size_t> shape_vec = {m, n};
            return tensor(shape_vec, out);
        }

        // inside your Add/Sub/Mul block, after the scalar & same-shape cases,
        // add 1-D bias broadcasting (broadcast b of shape [D] across trailing dim of a)
        if (a.shape().size() >= 1 && b.shape().size() == 1 && b.shape()[0] == a.shape().back())
        {
            std::vector<double> out(a.data().size());
            size_t inner = a.shape().back();
            size_t outer = a.data().size() / inner;
            for (size_t i = 0; i < outer; ++i)
            {
                for (size_t j = 0; j < inner; ++j)
                {
                    size_t idx = i * inner + j;
                    if (expr.op_type == "Add")
                        out[idx] = a.data()[idx] + b.data()[j];
                    if (expr.op_type == "Sub")
                        out[idx] = a.data()[idx] - b.data()[j];
                    if (expr.op_type == "Mul")
                        out[idx] = a.data()[idx] * b.data()[j];
                }
            }
            return tensor(a.shape(), out);
        }
        // symmetric case: a is 1-D bias and b has trailing dim == a[0]
        if (b.shape().size() >= 1 && a.shape().size() == 1 && a.shape()[0] == b.shape().back())
        {
            std::vector<double> out(b.data().size());
            size_t inner = b.shape().back();
            size_t outer = b.data().size() / inner;
            for (size_t i = 0; i < outer; ++i)
            {
                for (size_t j = 0; j < inner; ++j)
                {
                    size_t idx = i * inner + j;
                    if (expr.op_type == "Add")
                        out[idx] = a.data()[j] + b.data()[idx];
                    if (expr.op_type == "Sub")
                        out[idx] = a.data()[j] - b.data()[idx];
                    if (expr.op_type == "Mul")
                        out[idx] = a.data()[j] * b.data()[idx];
                }
            }
            return tensor(b.shape(), out);
        }

        throw std::runtime_error("Shape mismatch or unsupported broadcast in op: " + expr.op_type);
    }

    if (expr.op_type == "Linear")
    {
        // input tensor
        tensor in = eval_exprs(id_to_index.at(expr.inputs[0]));
        auto it_w = expr.params_ndarray.find("weight");
        if (it_w == expr.params_ndarray.end()) throw std::runtime_error("Linear missing weight");
        auto &wnd = it_w->second; // shape [out_features, in_features]
        size_t out_features = wnd.shape[0];
        size_t in_features = wnd.shape[1];

        // bias optional
        std::vector<double> bias_vec(out_features, 0.0);
        auto it_b = expr.params_ndarray.find("bias");
        if (it_b != expr.params_ndarray.end()) bias_vec = it_b->second.data;

        // compute outer (batch)
        if (in.data().size() % in_features != 0) throw std::runtime_error("Linear input size mismatch");
        size_t outer = in.data().size() / in_features;
        std::vector<size_t> out_shape = {outer, out_features};
        std::vector<double> out(outer * out_features, 0.0);

        for (size_t i = 0; i < outer; ++i)
        {
            for (size_t of = 0; of < out_features; ++of)
            {
                double acc = 0.0;
                for (size_t k = 0; k < in_features; ++k)
                {
                    // weight layout: [out_features, in_features]
                    acc += in.data()[i * in_features + k] * wnd.data[of * in_features + k];
                }
                acc += bias_vec[of];
                out[i * out_features + of] = acc;
            }
        }
        return tensor(out_shape, out);
    }

    if (expr.op_type == "ReLU")
    {
        tensor x = eval_exprs(id_to_index.at(expr.inputs[0]));
        std::vector<double> out(x.data().size());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = std::max(0.0, x.data()[i]);

        printf("ReLU: input shape=[");
        for (size_t s : x.shape())
            printf("%zu,", s);
        printf("], output data size=%zu\n", out.size());
        fflush(stdout);

        // verify sizes
        size_t expected = 1;
        for (size_t s : x.shape())
            expected *= s;
        printf("ReLU: expected=%zu, out.size=%zu, out.data=%p\n", expected, out.size(), out.data());
        fflush(stdout);
        if (expected != out.size())
        {
            printf("ReLU size mismatch: expected %zu but computed %zu\n", expected, out.size());
            fflush(stdout);
            throw std::runtime_error("ReLU output size mismatch");
        }

        // ensure lifetime: copy shape to local vector and use vector-copy ctor
        std::vector<size_t> shape_vec = x.shape();
        printf("About to construct tensor (vec): dim=%zu, shape_ptr=%p, data_ptr=%p\n",
               shape_vec.size(), shape_vec.data(), out.data());
        fflush(stdout);
        tensor result(shape_vec, out);
        printf("Constructed tensor (vec), about to return\n");
        fflush(stdout);
        return result;
    }

    if (expr.op_type == "Flatten")
    {
        tensor x = eval_exprs(id_to_index.at(expr.inputs[0]));
        if (x.shape().size() < 2)
            throw std::runtime_error("Flatten expects at least 2D input");
        size_t batch = x.shape()[0];
        size_t rest = 1;
        for (size_t i = 1; i < x.shape().size(); ++i)
            rest *= x.shape()[i];
        std::vector<size_t> shape_vec = {batch, rest};
        return tensor(shape_vec, x.data());
    }
    if (expr.op_type == "MatMul")
    {
        tensor A = eval_exprs(id_to_index.at(expr.inputs[0]));
        tensor B = eval_exprs(id_to_index.at(expr.inputs[1]));
        if (A.shape().size() != 2 || B.shape().size() != 2 || A.shape()[1] != B.shape()[0])
            throw std::runtime_error("MatMul expects two 2D tensors with matching inner dim");
        size_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
        std::vector<double> out(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t l = 0; l < k; ++l)
                    out[i * n + j] += A.data()[i * k + l] * B.data()[l * n + j];
        return tensor(std::vector<size_t>{m, n}, out);
    }
    // ...existing code...

    // Conv2D (channels-last) - no padding, stride = 1; supports weight layout [out_ch, in_ch, kh, kw]
    if (expr.op_type == "Conv2d" || expr.op_type == "Conv")
    {
        tensor input = eval_exprs(id_to_index.at(expr.inputs[0]));  // shape: [N,C,H,W]
        auto it_w = expr.params_ndarray.find("weight");
        if (it_w == expr.params_ndarray.end()) throw std::runtime_error("Conv2d missing weight");
        auto &wnd = it_w->second; // shape [out_ch, in_ch, kh, kw]
        std::vector<double> bias_vec;
        auto it_b = expr.params_ndarray.find("bias");
        bool has_bias = (it_b != expr.params_ndarray.end());
        if (has_bias) bias_vec = it_b->second.data;

        auto in_shape = input.shape();
        auto wt_shape = wnd.shape;
        if (in_shape.size() != 4 || wt_shape.size() != 4)
            throw std::runtime_error("Conv2d expects input [N,C,H,W] and weight [out,in,kh,kw]");

        size_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
        size_t out_ch = wt_shape[0], in_ch = wt_shape[1], kh = wt_shape[2], kw = wt_shape[3];
        if (in_ch != C) throw std::runtime_error("Conv2d input channels mismatch");

        size_t outH = H - kh + 1;
        size_t outW = W - kw + 1;
        std::vector<size_t> out_shape = {N, out_ch, outH, outW};
        std::vector<double> out(N * out_ch * outH * outW, 0.0);

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t oc = 0; oc < out_ch; ++oc)
            {
                for (size_t oh = 0; oh < outH; ++oh)
                {
                    for (size_t ow = 0; ow < outW; ++ow)
                    {
                        double acc = 0.0;
                        for (size_t ic = 0; ic < in_ch; ++ic)
                        {
                            for (size_t kh_i = 0; kh_i < kh; ++kh_i)
                            {
                                for (size_t kw_i = 0; kw_i < kw; ++kw_i)
                                {
                                    size_t ih = oh + kh_i;
                                    size_t iw = ow + kw_i;
                                    size_t in_idx = ((n * C + ic) * H + ih) * W + iw; // N,C,H,W
                                    size_t wt_idx = ((oc * in_ch + ic) * kh + kh_i) * kw + kw_i;
                                    acc += input.data()[in_idx] * wnd.data[wt_idx];
                                }
                            }
                        }
                        if (has_bias) acc += bias_vec[oc];
                        size_t out_idx = ((n * out_ch + oc) * outH + oh) * outW + ow; // N,outC,outH,outW
                        out[out_idx] = acc;
                    }
                }
            }
        }
        return tensor(out_shape, out);
    }

    if (expr.op_type == "MaxPool2d" || expr.op_type == "MaxPool")
    {
        int k = 2, s = 2;
        auto it_k = expr.params_double.find("kernel_size");
        if (it_k != expr.params_double.end()) k = (int)it_k->second;
        auto it_s = expr.params_double.find("stride");
        if (it_s != expr.params_double.end()) s = (int)it_s->second;

        tensor x = eval_exprs(id_to_index.at(expr.inputs[0])); // [N,C,H,W]
        auto shp = x.shape();
        if (shp.size() != 4) throw std::runtime_error("MaxPool2d expects 4D input");
        size_t N = shp[0], C = shp[1], H = shp[2], W = shp[3];
        size_t outH = (H - k) / s + 1;
        size_t outW = (W - k) / s + 1;
        std::vector<size_t> out_shape = {N, C, outH, outW};
        std::vector<double> out(N * C * outH * outW, -std::numeric_limits<double>::infinity());

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t c = 0; c < C; ++c)
            {
                for (size_t oh = 0; oh < outH; ++oh)
                {
                    for (size_t ow = 0; ow < outW; ++ow)
                    {
                        double best = -std::numeric_limits<double>::infinity();
                        for (int kh_i = 0; kh_i < k; ++kh_i)
                        {
                            for (int kw_i = 0; kw_i < k; ++kw_i)
                            {
                                size_t ih = oh * s + kh_i;
                                size_t iw = ow * s + kw_i;
                                size_t idx = ((n * C + c) * H + ih) * W + iw; // N,C,H,W
                                best = std::max(best, x.data()[idx]);
                            }
                        }
                        size_t out_idx = ((n * C + c) * outH + oh) * outW + ow; // N,C,outH,outW
                        out[out_idx] = best;
                    }
                }
            }
        }
        return tensor(out_shape, out);
    }

    // single final throw for unsupported ops
    throw std::runtime_error("Unsupported op_type for eval_exprs: " + expr.op_type);
}

int evaluation::execute()
{
    try
    {
        result_tensor_ = eval_exprs(exprs_.size() - 1);
        if (!result_tensor_.data().empty())
        {
            result_ = result_tensor_.data()[0];
        }
        else
        {
            result_ = 0.0;
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        printf("evaluation exception: %s\n", e.what());
        fflush(stdout);
        return -1;
    }
    catch (...)
    {
        printf("evaluation unknown exception (non-std)\n");
        fflush(stdout);
        return -1;
    }
}

double &evaluation::get_result()
{
    printf("Returning result: %f\n", result_);
    return result_;
}

const tensor &evaluation::get_result_tensor() const
{
    return result_tensor_;
}
// ...existing code...