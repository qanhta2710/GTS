import numpy as np
from sympy import Matrix, symbols, solve_linear_system

def read_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Tìm vị trí dòng phân cách "---"
    separator_index = lines.index('---\n')

    # Đọc ma trận A
    A = np.array([list(map(float, line.split())) for line in lines[:separator_index]])

    # Đọc vector b
    b = np.array([float(line.strip()) for line in lines[separator_index + 1:]])

    return A, b

def gauss_elimination_check(A, b, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    m, n = A.shape
    Ab = np.hstack([A, b])  # Ma trận mở rộng
    pivot_columns = []

    print("Ma trận mở rộng ban đầu:")
    print(Ab)
    print("-" * 50)

    for i in range(min(m, n)):
        # Kiểm tra và hoán đổi hàng nếu cần
        if abs(Ab[i, i]) < tol:
            for t in range(i + 1, m):
                if abs(Ab[t, i]) > tol:
                    Ab[[i, t]] = Ab[[t, i]]
                    print(f"Hoán đổi hàng {i} và hàng {t}:")
                    print(Ab)
                    print("-" * 50)
                    break
        if abs(Ab[i, i]) < tol:
            continue

        pivot_columns.append(i)

        # Khử các phần tử bên dưới phần tử chốt
        for j in range(i + 1, m):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
            print(f"Khử hàng {j} bằng hàng {i} với hệ số {factor:.6f}:")
            print(Ab)
            print("-" * 50)

    # Kiểm tra vô nghiệm
    for i in range(m):
        if np.all(np.abs(Ab[i, :-1]) < tol) and abs(Ab[i, -1]) > tol:
            print("Hệ phương trình vô nghiệm.")
            return {
                "type": "inconsistent",
                "pivot_columns": pivot_columns,
                "Ab": Ab
            }

    # Kiểm tra vô số nghiệm
    if len(pivot_columns) < n:
        print("Hệ có vô số nghiệm.")
        aug = Matrix(np.hstack([A, b]).tolist())
        vars = symbols(f'x1:{n+1}')
        sol = solve_linear_system(aug, *vars)

        print("Nghiệm tổng quát:")
        if sol:
            for var, expr in sol.items():
                print(f"{var} = {expr}")
        else:
            print("Các biến tự do (free variables):")
            free_vars = [var for var in vars if var not in sol]
            for var in free_vars:
                print(f"{var} = tự do")

        return {
            "type": "infinite",
            "pivot_columns": pivot_columns,
            "general_solution": sol,
            "Ab": Ab
        }

    # Hệ có nghiệm duy nhất → giải bằng thế ngược
    x = np.zeros(n)
    print("Back substitution steps:")
    for i in reversed(range(len(pivot_columns))):
        row = pivot_columns[i]
        s = np.dot(Ab[row, row+1:n], x[row+1:n])
        x[row] = (Ab[row, -1] - s) / Ab[row, row]

        print(f"Iteration for row {row}: x = {x}")
    
    print("Nghiệm duy nhất:")
    print(x)

    return {
        "type": "unique",
        "pivot_columns": pivot_columns,
        "solution": x,
        "Ab": Ab
    }

# Đọc ma trận từ file
filename = 'matrix.txt'
A, b = read_matrix_from_file(filename)

# Giải hệ phương trình
gauss_elimination_check(A, b)