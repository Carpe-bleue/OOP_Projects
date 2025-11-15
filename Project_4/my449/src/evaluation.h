#ifndef EVALUATION_H
#define EVALUATION_H
#include "expression.h"
#include <vector>
#include <map>
#include <string>
#include <cstdio>


class tensor {
public:
    tensor() {}
    explicit tensor(double value) : shape_({}), data_({value}) {}
    tensor(int dim, const size_t* shape, const double* data) {
        printf("tensor ctor called\n"); fflush(stdout);
        shape_.assign(shape, shape + dim);
        size_t total = 1;
        for (int i = 0; i < dim; ++i) total *= shape[i];
        data_.assign(data, data + total);
        printf("tensor ctor: dim=%d, shape=[", dim); fflush(stdout);
        for (int i = 0; i < dim; ++i) printf("%zu,", shape[i]);
        printf("], data size=%zu\n", data_.size()); fflush(stdout);
    }

    tensor(const std::vector<size_t>& shape_vec, const std::vector<double>& data_vec) {
        printf("tensor ctor called\n"); fflush(stdout);
        shape_ = shape_vec;
        data_ = data_vec;
        printf("tensor ctor (vec): dim=%zu, shape=[", shape_.size()); fflush(stdout);
        for (size_t i = 0; i < shape_.size(); ++i) printf("%zu,", shape_[i]);
        printf("], data size=%zu\n", data_.size()); fflush(stdout);
    }

    const size_t *get_shape_array() const { return shape_.empty() ? nullptr : shape_.data(); }
    const double *get_data_array() const { return data_.empty() ? nullptr : data_.data(); }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<double>& data() const { return data_; }
private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
};

class evaluation {
public:
    evaluation(const std::vector<expression> &exprs);
    void add_kwargs_double(const char *key, double value);
    void add_kwargs_ndarray(const char *key, int dim, size_t shape[], double data[]);
    int execute();
    double &get_result();
    const tensor& get_result_tensor() const;
private:
    std::vector<expression> exprs_;
    std::map<std::string, double> input_values;
    std::map<int, size_t> id_to_index;
    double result_;
    tensor result_tensor_;
    std::map<std::string, std::vector<double>> input_vectors;
    std::map<std::string, std::vector<size_t>> input_shapes;
    
    tensor eval_exprs(int idx) const;
};
#endif // EVALUATION_H