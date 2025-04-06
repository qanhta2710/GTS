import numpy as np

import numpy as np
from sympy import Matrix, symbols, solve_linear_system

def gauss_elimination_check(A, b, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    m, n = A.shape
    Ab = np.hstack([A, b])  # Ma trận mở rộng
    pivot_columns = []

    for i in range(min(m, n)):
        if abs(Ab[i, i]) < tol:
            for t in range(i + 1, m):
                if abs(Ab[t, i]) > tol:
                    Ab[[i, t]] = Ab[[t, i]]
                    print(Ab)
                    break
        if abs(Ab[i, i]) < tol:
            continue
        pivot_columns.append(i)
        for j in range(i + 1, m):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
        print(Ab)

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
        # Print the intermediate solution vector x after each iteration
        print(f"Iteration for row {row}: x = {x}")
    
    print("Nghiệm duy nhất:")
    print(x)

    return {
        "type": "unique",
        "pivot_columns": pivot_columns,
        "solution": x,
        "Ab": Ab
    }

A = [[2, 3, -5, 8],
     [3, -2, 1, 5],
     [1, -18, 23, -17],
     [8, -1, -3, 18]]
b = [4, 2, -10, 8]

gauss_elimination_check(A, b)
