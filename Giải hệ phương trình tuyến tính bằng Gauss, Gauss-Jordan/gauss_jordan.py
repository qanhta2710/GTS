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

    row = 0
    for col in range(n):
        if row >= m:
            break

        # Tìm pivot: ưu tiên |pivot| = 1, nếu không có thì chọn giá trị tuyệt đối lớn nhất
        pivot_row = row
        pivot_value = abs(Ab[row, col])
        has_one = abs(Ab[row, col]) == 1.0

        for j in range(row + 1, m):
            abs_val = abs(Ab[j, col])
            if abs_val == 1.0:
                pivot_row = j
                pivot_value = abs_val
                has_one = True
                break
            elif not has_one and abs_val > pivot_value:
                pivot_row = j
                pivot_value = abs_val

        if pivot_value < tol:
            print(f"Pivot tại cột {col} quá nhỏ ({Ab[row, col]}), bỏ qua cột {col}.")
            continue

        # Hoán đổi hàng nếu cần
        if pivot_row != row:
            Ab[[row, pivot_row]] = Ab[[pivot_row, row]]
            print(f"Hoán đổi hàng {row} và hàng {pivot_row}:")
            print(Ab)
            print("-" * 50)

        pivot_columns.append(col)

        # Chuẩn hóa pivot về 1
        pivot = Ab[row, col]
        Ab[row, :] /= pivot
        print(f"Chuẩn hóa hàng {row} để pivot = 1:")
        print(Ab)
        print("-" * 50)

        # Khử tất cả các phần tử khác trong cột col
        for j in range(m):
            if j != row and abs(Ab[j, col]) > tol:
                factor = Ab[j, col]
                Ab[j, :] -= factor * Ab[row, :]
                print(f"Khử hàng {j} bằng hàng {row} với hệ số {factor:.6f}:")
                print(Ab)
                print("-" * 50)

        row += 1

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
            for i in range(n):
                if i in pivot_columns:
                    pivot_idx = pivot_columns.index(i)
                    X[i, k] = Ab[pivot_idx, n + k]
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