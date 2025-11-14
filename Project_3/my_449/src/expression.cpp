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
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
}
