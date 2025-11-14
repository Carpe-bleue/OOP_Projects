/**
 * A simple test program helps you to debug your easynn implementation.
 */

#include <stdio.h>
#include "src/libeasynn.h"

// Two class declarations and six function declarations are in src/libeasynn.h

int main()
{
    program *prog = create_program();

    int inputs0[] = {};
    append_expression(prog, 0, "a", "Input", inputs0, 0);

    //Added
    //int inputs1[] = {};
    //append_expression(prog, 1, "x", "Const", inputs1, 0);
    //add_op_param_double(prog, "value", 3);
    //

    int inputs1[] = {0, 0};
    append_expression(prog, 1, "", "Add", inputs1, 2);

    int inputs2[] = {0, 1};
    append_expression(prog, 2, "", "Add", inputs2, 2);

    printf("*** flag 1 ***\n");

    evaluation *eval = build(prog);
    add_kwargs_double(eval, "a", 2);

    printf("*** flag 2 ***\n");


    int dim = 0;
    const size_t *shape = nullptr;
    const double *data = nullptr;
    if (execute(eval, &dim, &shape, &data) != 0)
    {
        printf("***flag 3 ***\n");
        printf("evaluation fails\n");
        return -1;
    }

    if (dim == 0)
        printf("res = %f\n", data[0]);
    else
        printf("result as tensor is not supported yet\n");

    return 0;
}
