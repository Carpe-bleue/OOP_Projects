#include <assert.h>
#include "evaluation.h"
#include <stdexcept>
#include <cstdio>

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
    if (expr.op_type == "Input")
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
            size_t total_size = 1;
            for (size_t dim : it_shape->second)
            {
                total_size *= dim;
            }
            if (it_vec->second.size() != total_size)
            {
                throw std::runtime_error("Input vector size does not match shape for: " + expr.op_name);
            }
            return tensor(it_shape->second.size(), it_shape->second.data(), it_vec->second.data());
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
            return tensor(it_nd->second.shape.size(), it_nd->second.shape.data(), it_nd->second.data.data());
        }
        printf("Const value missing!\n");
        throw std::runtime_error("Missing const value");
    }

    if (expr.op_type == "Add" || expr.op_type == "Sub" || expr.op_type == "Mul") {
    // Get input tensors
    int idx_a = id_to_index.at(expr.inputs[0]);
    int idx_b = id_to_index.at(expr.inputs[1]);
    tensor a = eval_exprs(idx_a);
    tensor b = eval_exprs(idx_b);

    // Scalar case
    if (a.data().size() == 1 && b.data().size() == 1) {
        double result = 0.0;
        if (expr.op_type == "Add") result = a.data()[0] + b.data()[0];
        if (expr.op_type == "Sub") result = a.data()[0] - b.data()[0];
        if (expr.op_type == "Mul") result = a.data()[0] * b.data()[0];
        return tensor(result);
    }

    if (a.shape() == b.shape()) {
        std::vector<double> out(a.data().size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (expr.op_type == "Add") out[i] = a.data()[i] + b.data()[i];
            if (expr.op_type == "Sub") out[i] = a.data()[i] - b.data()[i];
            if (expr.op_type == "Mul") out[i] = a.data()[i] * b.data()[i];
        }
        return tensor(a.shape().size(), a.shape().data(), out.data());
    }

    if (a.data().size() == 1) {
        std::vector<double> out(b.data().size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (expr.op_type == "Add") out[i] = a.data()[0] + b.data()[i];
            if (expr.op_type == "Sub") out[i] = a.data()[0] - b.data()[i];
            if (expr.op_type == "Mul") out[i] = a.data()[0] * b.data()[i];
        }
        return tensor(b.shape().size(), b.shape().data(), out.data());
    }
    if (b.data().size() == 1) {
        std::vector<double> out(a.data().size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (expr.op_type == "Add") out[i] = a.data()[i] + b.data()[0];
            if (expr.op_type == "Sub") out[i] = a.data()[i] - b.data()[0];
            if (expr.op_type == "Mul") out[i] = a.data()[i] * b.data()[0];
        }
        return tensor(a.shape().size(), a.shape().data(), out.data());
    }

    if (a.shape().size() == 2 && b.shape().size() == 2 &&
        a.shape()[1] == b.shape()[0]) {
        size_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
        std::vector<double> out(m * n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t l = 0; l < k; ++l) {
                    out[i * n + j] += a.data()[i * k + l] * b.data()[l * n + j];
                }
            }
        }
        size_t shape[2] = {m, n};
        return tensor(2, shape, out.data());
    }

    throw std::runtime_error("Shape mismatch or unsupported broadcast in op: " + expr.op_type);
}
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
    catch (...)
    {
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
