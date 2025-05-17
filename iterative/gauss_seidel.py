import numpy as np
from utils.matrix_utils import pretty_print_matrix, get_random_regular_matrix

def gauss_seidel(A, b, x0, iterations):
    n = A.shape[0]
    x = x0.copy()
    steps = []
    for k in range(iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        steps.append(x_new.copy())
        x = x_new
    return steps

