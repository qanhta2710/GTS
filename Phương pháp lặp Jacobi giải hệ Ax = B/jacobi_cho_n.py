import numpy as np

def read_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n, m = map(int, lines[0].split())  # Đọc n, m
    # Đọc ma trận A (n x n)
    A = np.array([list(map(float, lines[1 + i].split())) for i in range(n)])
    # Đọc ma trận B (n x m)
    B = np.array([list(map(float, lines[1 + n + i].split())) for i in range(n)])
    # Kiểm tra và đọc X0 nếu có, nếu không thì khởi tạo X0 là ma trận 0 (n x m)
    x0_start = 1 + 2 * n
    if len(lines) >= x0_start + n:
        X0 = np.array([list(map(float, lines[x0_start + i].split())) for i in range(n)])
        if X0.shape != (n, m):
            raise ValueError(f"Ma trận X0 phải có kích thước {n}x{m}")
    else:
        X0 = np.zeros((n, m))  # Khởi tạo X0 với kích thước đúng (n x m)
    return A, B, X0

def is_row_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off_diag_sum:
            return False
    return True

def is_column_diagonally_dominant(A):
    n = A.shape[0]
    for j in range(n):
        diag = abs(A[j, j])
        off_col_sum = sum(abs(A[i, j]) for i in range(n) if i != j)
        if diag <= off_col_sum:
            return False
    return True

def compute_T(A):
    n = A.shape[0]
    T = np.zeros((n, n))
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("Phần tử đường chéo a_ii bằng 0!")
        T[i, i] = 1 / A[i, i]
    return T

def compute_F(A):
    n = A.shape[0]
    T = compute_T(A)
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = -A[i, j] / A[i, i]
    return F

def compute_D(A, B):
    n, m = B.shape
    T = compute_T(A)
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = T[i, i] * B[i, j]
    return D

def compute_norm_F_infinity(A):
    n = A.shape[0]
    norms = []
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        norms.append(off_diag_sum / diag)
    return max(norms)

def compute_q(A):
    n = A.shape[0]
    norms = []
    for j in range(n):
        diag = abs(A[j, j])
        off_col_sum = sum(abs(A[i, j]) for i in range(n) if i != j)
        norms.append(off_col_sum / diag)
    return max(norms)

def compute_lambda(A):
    n = A.shape[0]
    diag_elements = [abs(A[i, i]) for i in range(n)]
    return max(diag_elements) / min(diag_elements)

def jacobi_iteration(A, B, norm_type, k, X0=None):
    n, m = B.shape
    if X0 is not None:
        X0 = np.array(X0)
        if X0.shape != (n, m):
            X0 = X0[:, :m]  # Cắt X0 để khớp với kích thước (n, m)
        X = X0.copy()
    else:
        X = np.zeros((n, m))
    F = compute_F(A)
    D = compute_D(A, B)

    if norm_type == 'infinity':
        norm_F = compute_norm_F_infinity(A)
    else:
        q = compute_q(A)
        lambda_val = compute_lambda(A)

    print(f"\nChuẩn sử dụng: {'vô cực' if norm_type == 'infinity' else '1'}")
    if norm_type == 'infinity':
        print(f"||F||_infinity = {norm_F:.4f}")
    else:
        print(f"q = {q:.4f}")
        print(f"lambda = {lambda_val:.4f}")
    print("\nBắt đầu lặp Jacobi:")
    print(f"{'Lần lặp':<10} {'X':<40} {'Sai số tuyệt đối':<15}")
    print("-" * 65)

    for i in range(k):
        X_prev = X.copy()
        X = F @ X_prev + D
        if norm_type == 'infinity':
            error = np.max(np.abs(X - X_prev))
        else:
            error = np.sum(np.abs(X - X_prev))
        x_str = "[" + "  ".join(f"{val:.6f}" for val in X.flatten()) + "]"
        print(f"{i+1:<10} {x_str:<40} {error:<15.6e}")

    print(f"\nSai số tuyệt đối ở lần lặp cuối (lần {k}): {error:.6e}")
    return X, k, error

def main():
    filename = 'input.txt'
    try:
        A, B, X0 = read_input(filename)
    except FileNotFoundError:
        print(f"Không tìm thấy file {filename}")
        return
    k = 4
    print(f"Số lần lặp: {k}")
    print("Ma trận A:")
    print(A)
    print("Ma trận B:")
    print(B)
    print("Ma trận X0:")
    print(X0)
    print(f"Số lần lặp: {k}")
    
    try:
        F = compute_F(A)
        D = compute_D(A, B)
        print("\nMa trận F = I - TA:")
        print(np.round(F, 4))
        print("Ma trận D = TB:")
        print(np.round(D, 4))
    except ValueError as e:
        print(f"Lỗi: {e}")
        return
    
    if is_row_diagonally_dominant(A):
        print("\nMa trận chiếm ưu thế đường chéo theo hàng.")
        norm_type = 'infinity'
    elif is_column_diagonally_dominant(A):
        print("\nMa trận chiếm ưu thế đường chéo theo cột.")
        norm_type = '1'
    else:
        print("\nA không là ma trận chéo trội hàng và cột nên không thực hiện lặp theo Jacobi được.")
        return
    
    X, iterations, final_error = jacobi_iteration(A, B, norm_type, k, X0)
    
    print(f"\nKết quả cuối cùng:")
    x_str = "[" + "  ".join(f"{val:.6f}" for val in X.flatten()) + "]"
    print(f"Nghiệm X:\n{x_str}")
    print(f"Số lần lặp: {iterations}")
    print(f"Sai số tuyệt đối cuối cùng: {final_error:.6e}")

if __name__ == "__main__":
    main()