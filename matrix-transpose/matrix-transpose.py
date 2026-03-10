import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    n, m = np.array(A).shape
    B = np.zeros((m, n))

    for i in range(n):
        for j in range(m):
            B[j][i] = A[i][j]

    return B
