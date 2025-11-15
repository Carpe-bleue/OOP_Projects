#include <stdio.h>
#include "src/libeasynn.h"

int main()
{
    program *prog = create_program();

    // Input node
    int inputs0[] = {};
    append_expression(prog, 0, "x", "Input", inputs0, 0);

    // ReLU node
    int inputs1[] = {0};
    append_expression(prog, 1, "", "ReLU", inputs1, 1);

    evaluation *eval = build(prog);

    // Prepare 4D tensor data
    size_t shape[4] = {2, 2, 2, 2};
    double data[16];
    for (int i = 0; i < 16; ++i) data[i] = i - 8; // Some negative, some positive

    add_kwargs_ndarray(eval, "x", 4, shape, data);

    int dim = 0;
    const size_t *out_shape = nullptr;
    const double *out_data = nullptr;
    if (execute(eval, &dim, &out_shape, &out_data) != 0)
    {
        printf("evaluation fails\n");
        return -1;
    }

    printf("Output tensor: dim=%d, shape=[", dim);
    for (int i = 0; i < dim; ++i) printf("%zu,", out_shape[i]);
    printf("]\n");
    printf("Output data:\n");
    for (int i = 0; i < 16; ++i) printf("%f ", out_data[i]);
    printf("\n");

    return 0;
}