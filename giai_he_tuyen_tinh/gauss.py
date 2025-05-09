import numpy as np
from sympy import Matrix, symbols, solve_linear_system

def read_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Tìm vị trí dòng phân cách "---"
    separator_index = lines.index('---\n')

    # Đọc ma trận A
    A = np.array([list(map(float, line.split())) for line in lines[:separator_index]])

    # Đọc vector B
    B = np.array([list(map(float, line.split())) for line in lines[separator_index + 1:]])

    return A, B

def gauss_elimination_check(A, B, tol=1e-10):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape
    Ab = np.hstack([A, B])
    pivot_columns = []

    print("Ma trận mở rộng ban đầu:")
    print(Ab)
    print("-" * 50)

    for i in range(min(m, n)):
        # Kiểm tra pivot, không hoán đổi hàng
        if abs(Ab[i, i]) < tol:
            print(f"Pivot tại [{i}, {i}] quá nhỏ ({Ab[i, i]}), bỏ qua cột {i}.")
            continue

        pivot_columns.append(i)

        # Khử các phần tử bên dưới pivot
        for j in range(i + 1, m):
            if abs(Ab[j, i]) > tol:  # Chỉ khử nếu phần tử không quá nhỏ
                factor = Ab[j, i] / Ab[i, i]
                Ab[j, i:] -= factor * Ab[i, i:]
                print(f"Khử hàng {j} bằng hàng {i} với hệ số {factor:.6f}:")
                print(Ab)
                print("-" * 50)

    # Kiểm tra vô nghiệm
    for i in range(m):
        if np.all(np.abs(Ab[i, :-B.shape[1]]) < tol) and np.any(np.abs(Ab[i, -B.shape[1]:]) > tol):
            print("Hệ phương trình vô nghiệm.")
            return {
                "type": "inconsistent",
                "Ab": Ab
            }

    # Kiểm tra vô số nghiệm
    rank = len(pivot_columns)
    if rank < n:
        print("Hệ có vô số nghiệm.")
        return {
            "type": "infinite",
            "Ab": Ab
        }
    
    def back_substitution(Ab, m, n):
        num_b_columns = Ab.shape[1] - n
        X = np.zeros((n, num_b_columns))
        for k in range(num_b_columns):
            for i in range(n - 1, -1, -1):
                if i in pivot_columns:
                    pivot_idx = pivot_columns.index(i)
                    X[i, k] = (Ab[pivot_idx, n + k] - np.dot(Ab[pivot_idx, i + 1:n], X[i + 1:, k])) / Ab[pivot_idx, i]
        return X

    X = back_substitution(Ab, m, n)
    print("Nghiệm duy nhất cho từng cột của B:")
    print(X)

    return {
        "type": "unique",
        "solution": X,
        "Ab": Ab
    }

# Đọc ma trận từ file
filename = 'matrix.txt'
A, B = read_matrix_from_file(filename)

# Giải hệ phương trình
gauss_elimination_check(A, B)