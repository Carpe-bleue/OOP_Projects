#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>

class evaluation;

class expression
{
    friend class evaluation;
public:
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);


    int expr_id;
    std::string op_name;
    std::string op_type;
    std::vector<int> inputs;

    std::map<std::string, double> params_double;

    struct ndarray {
        int dim;
        std::vector<size_t> shape;
        std::vector<double> data;
    };
    std::map<std::string, ndarray> params_ndarray;

}; // class expression

#endif // EXPRESSION_H
