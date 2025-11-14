#include "program.h"
#include "evaluation.h"

program::program()
{
}

void program::append_expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    expression expr(expr_id, op_name, op_type, inputs, num_inputs);
    exprs.push_back(expr);
}

int program::add_op_param_double(
    const char *key,
    double value)
{
    if (exprs.empty()) return -1; // No expressions to add param to

    // Add the parameter to the last expression
    exprs.back().params_double[key] = value;
    return 0;
}

int program::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    if (exprs.empty()) return -1; // No expressions to add param to

    expression::ndarray arr;
    arr.shape.assign(shape, shape + dim);
    arr.dim = dim;
    size_t total = 1;
    for (int i = 0; i < dim; ++i) total *= shape[i];
    arr.data.assign(data, data + total);

    exprs.back().params_ndarray[key] = arr;
    return 0;
}

evaluation *program::build()
{
    
    return new evaluation(exprs); // assuming exprs is a vector<expression>

    //return nullptr;
}
