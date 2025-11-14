import numpy as np
import easynn as nn

# Create a numpy array of 10 rows and 5 columns.
# Set the element at row i and column j to be i+j.
def Q1():
    a = np.zeros((10,5))
    for i in range(10):
        for j in range(5):
            a[i,j]=i+j
    return a

# Add two numpy arrays together.
def Q2(a, b):
    return a+b

# Multiply two 2D numpy arrays using matrix multiplication.
def Q3(a, b):
    return a@b

# For each row of a 2D numpy array, find the column index
# with the maximum element. Return all these column indices.
def Q4(a):
    col_ind = []
    for r in range(len(a)):

        #a[r].index(cmax) seemed to work 
        ind_c_max = np.argmax(a[r])
        col_ind.append(ind_c_max)

    return col_ind

#Q4([[1,3,2,4],[3,4,7],[9,7,8,4]])

# Solve Ax = b.
def Q5(A, b):
    x = nn.Input("x")
    A_expr = nn.Const(A)
    b_expr = nn.Const(b)
    # x = x.resolve()(A_expr, b_expr)
    # x = b_expr * nn.Const(1/A)   # x = b / A
    # return x
    return np.linalg.solve(A,b)

# Return an EasyNN expression for a+b.
def Q6():
    a = nn.Input("a")
    b = nn.Input("b")
    expression = a+b
    return expression

# Return an EasyNN expression for a+b*c.
def Q7():
    a = nn.Input("a")
    b = nn.Input("b")
    c = nn.Input("c")
    expression = a+b*c
    return expression


# Given A and b, return an EasyNN expression for Ax+b.
def Q8(A,b):
    A = nn.Const(A)
    x = nn.Input("x")   # runtime input
    b = nn.Const(b)  # runtime input
    expression = A*x+b
    return expression

# Given n, return an EasyNN expression for x**n.
def Q9(n):
    x = nn.Input("x")
    expression = x
    for i in range(n-1):
        expression = expression*x
    return expression

# Return an EasyNN expression to compute
# the element-wise absolute value |x|.
def Q10():
    x = nn.Input("x")
    neg_x = x.__neg__()
    expression = nn.ReLU()(x)+nn.ReLU()(neg_x)

    return expression
