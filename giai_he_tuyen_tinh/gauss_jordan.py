import numpy as np

def gauss_jordan(A, b, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    m, n = A.shape
    Ab = np.hstack([A, b])  # Ma trận mở rộng
    pivot_columns = []

    row = 0
    for col in range(n):
        # Tìm dòng có phần tử khác 0 lớn nhất tại cột hiện tại
        pivot = np.argmax(np.abs(Ab[row:, col])) + row
        if abs(Ab[pivot, col]) < tol:
            continue
        # Hoán đổi dòng
        Ab[[row, pivot]] = Ab[[pivot, row]]
        pivot_columns.append(col)

        # Chia dòng hiện tại để phần tử pivot = 1
        Ab[row] = Ab[row] / Ab[row, col]

        # Khử các phần tử khác trên/dưới pivot
        for r in range(m):
            if r != row:
                Ab[r] -= Ab[r, col] * Ab[row]
        

        print(Ab)
        print("-" * 40)

        row += 1
        if row == m:
            break

    # Kiểm tra vô nghiệm
    for i in range(m):
        if np.all(np.abs(Ab[i, :-1]) < tol) and abs(Ab[i, -1]) > tol:
            print("Hệ phương trình vô nghiệm.")
            return {
                "type": "inconsistent",
                "Ab": Ab,
                "pivot_columns": pivot_columns
            }

    # Kiểm tra vô số nghiệm
    if len(pivot_columns) < n:
        print("Hệ có vô số nghiệm.")
        from sympy import Matrix, symbols, solve_linear_system
        aug = Matrix(Ab.tolist())
        vars = symbols(f'x1:{n+1}')
        sol = solve_linear_system(aug, *vars)

        print("Nghiệm tổng quát:")
        print(sol)

        return {
            "type": "infinite",
            "general_solution": sol,
            "Ab": Ab,
            "pivot_columns": pivot_columns
        }

    # Hệ có nghiệm duy nhất
    solution = Ab[:, -1]
    print("Nghiệm duy nhất:")
    print(solution)

    return {
        "type": "unique",
        "solution": solution,
        "Ab": Ab,
        "pivot_columns": pivot_columns
    }

# Ví dụ
A = [[2, 3, -5, 8],
     [3, -2, 1, 5],
     [1, -18, 23, -17],
     [8, -1, -3, 18]]
b = [4, 2, -10, 8]

result = gauss_jordan(A, b)
