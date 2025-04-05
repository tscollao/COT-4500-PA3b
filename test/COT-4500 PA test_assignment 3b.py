import numpy as np

# Task 1: Gaussian Elimination and Backward Substitution
def gaussian_elimination(matrix, b):
    n = len(matrix)
    # Augmenting matrix [A|b]
    augmented_matrix = np.hstack((matrix, b.reshape(-1, 1)))

    # Forward Elimination
    for i in range(n):
        # Pivoting: Ensure the diagonal element is the largest
        if augmented_matrix[i, i] == 0:
            for j in range(i+1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break

        # Make the elements below the pivot zero
        for j in range(i+1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Backward Substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:n])) / augmented_matrix[i, i]
    
    return x

# Task 2: LU Factorization and Determinant Calculation
def lu_factorization(matrix):
    n = len(matrix)
    L = np.zeros_like(matrix)
    U = np.zeros_like(matrix)

    for i in range(n):
        # Upper triangular matrix U
        U[i, i:] = matrix[i, i:]
        for j in range(i):
            U[i, j] = 0

        # Lower triangular matrix L
        L[i, i] = 1
        for j in range(i+1, n):
            L[j, i] = matrix[j, i] / U[i, i]
            U[i, i:] -= L[j, i] * U[i, i:]

    # Determinant is the product of the diagonal elements of U
    determinant = np.prod(np.diagonal(U))
    
    return determinant, L, U

# Task 3: Check if matrix is Diagonally Dominant
def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = np.sum(np.abs(matrix[i])) - np.abs(matrix[i, i])
        if np.abs(matrix[i, i]) <= row_sum:
            return False
    return True

# Task 4: Check if the matrix is Positive Definite
def is_positive_definite(matrix):
    try:
        # Matrix must be symmetric and all eigenvalues must be positive
        eigenvalues = np.linalg.eigvals(matrix)
        if np.all(eigenvalues > 0):
            return True
        else:
            return False
    except np.linalg.LinAlgError:
        return False

# 1. Gaussian Elimination and Backward Substitution
A1 = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]], dtype=float)
b1 = np.array([6, 0, -3], dtype=float)
x1 = gaussian_elimination(A1, b1)
print("Solution to the system of equations:", x1)

# 2. LU Factorization
A2 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
determinant, L, U = lu_factorization(A2)
print("\nLU Factorization of the matrix A:")
print("Determinant:", determinant)
print("L matrix:")
print(L)
print("U matrix:")
print(U)

# 3. Check if matrix is Diagonally Dominant
A3 = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]], dtype=float)
print("\nIs the matrix diagonally dominant?", is_diagonally_dominant(A3))

# 4. Check if matrix is Positive Definite
A4 = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]], dtype=float)
print("\nIs the matrix positive definite?", is_positive_definite(A4))
