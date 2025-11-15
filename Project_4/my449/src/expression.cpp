#include "expression.h"

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs)
{
    this->expr_id = expr_id;
    this->op_name = op_name;
    this->op_type = op_type;
    this->inputs.assign(inputs, inputs + num_inputs); // Copy inputs array to vector
}

void expression::add_op_param_double(
    const char *key,
    double value)
{
    params_double[key] = value;
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    ndarray nd;
    nd.dim = dim;
    nd.shape.assign(shape, shape + dim);
    size_t total = 1;
    for (int i = 0; i < dim; ++i) total *= shape[i];
    nd.data.assign(data, data + total);
    params_ndarray[std::string(key)] = std::move(nd);
}
