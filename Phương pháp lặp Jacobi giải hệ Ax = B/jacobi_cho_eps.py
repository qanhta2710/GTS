import numpy as np

def read_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n, m = map(int, lines[0].split())
    # Đọc ma trận A
    A = np.array([list(map(float, lines[1 + i].split())) for i in range(n)])
    # Đọc ma trận B
    B = np.array([list(map(float, lines[1 + n + i].split())) for i in range(n)])
    # Đọc X0 nếu có
    x0_start = 1 + 2 * n
    if len(lines) >= x0_start + n:
        X0 = np.array([float(lines[x0_start + i]) for i in range(n)]).reshape((n, m))
    else:
        X0 = np.zeros((n, m))
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
    # Tính F = I - TA
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
    # Tính D = TB
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

def jacobi_iteration(A, B, norm_type, epsilon=1e-4, max_iterations=100, X0=None):
    n, m = B.shape
    if X0 is not None:
        X = np.array(X0).reshape((n, m))
    else:
        X = np.zeros((n, m))  # Khởi tạo X0 = 0 cho mọi hệ
    F = compute_F(A)
    D = compute_D(A, B)
    iterations = 0

    if norm_type == 'infinity':
        norm_F = compute_norm_F_infinity(A)
        error_bound = (1 - norm_F) * epsilon / norm_F
    else:
        q = compute_q(A)
        lambda_val = compute_lambda(A)
        error_bound = (1 - q) * epsilon / (lambda_val * q)

    print(f"\nChuẩn sử dụng: {'vô cực' if norm_type == 'infinity' else '1'}")
    if norm_type == 'infinity':
        print(f"||F||_infinity = {norm_F:.4f}")
    else:
        print(f"q = {q:.4f}")
        print(f"lambda = {lambda_val:.4f}")
    print(f"Giới hạn sai số lý thuyết: {error_bound:.6e}")
    print("\nBắt đầu lặp Jacobi:")
    print(f"{'Lần lặp':<10} {'X':<40} {'Sai số':<15}")
    print("-" * 65)

    for k in range(max_iterations):
        X_prev = X.copy()
        X = F @ X_prev + D

        # Tính sai số: lấy max trên toàn bộ ma trận X (tất cả các hệ)
        if norm_type == 'infinity':
            error = np.max(np.abs(X - X_prev))
        else:
            error = np.sum(np.abs(X - X_prev))
        # Định dạng nghiệm với 6 chữ số sau dấu phẩy
        x_str = "[" + "  ".join(f"{val:.6f}" for val in X.flatten()) + "]"
        print(f"{k+1:<10} {x_str:<40} {error:<15.6e}")

        iterations += 1
        if error < error_bound:
            break

    return X, iterations, error_bound

def main():
    filename = 'input.txt'
    try:
        A, B, X0 = read_input(filename)
    except FileNotFoundError:
        print(f"Không tìm thấy file {filename}")
        return

    print("Ma trận A:")
    print(A)
    print("Ma trận B:")
    print(B)
    print("Ma trận X0:")
    print(X0)
    
    # Tính và hiển thị F và D
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
    
    # Kiểm tra chéo trội
    if is_row_diagonally_dominant(A):
        print("\nMa trận chiếm ưu thế đường chéo theo hàng.")
        norm_type = 'infinity'
    elif is_column_diagonally_dominant(A):
        print("\nMa trận chiếm ưu thế đường chéo theo cột.")
        norm_type = '1'
    else:
        print("\nA không là ma trận chéo trội hàng và cột nên không thực hiện lặp theo Jacobi được.")
        return
    
    # Thực hiện lặp Jacobi
    epsilon = 1e-6
    norm_type = 'infinity' if is_row_diagonally_dominant(A) else '1'
    X, iterations, error_bound = jacobi_iteration(A, B, norm_type, epsilon)
    
    print(f"\nKết quả cuối cùng:")
    x_str = "[" + "  ".join(f"{val:.6f}" for val in X.flatten()) + "]"
    print(f"Nghiệm X:\n{x_str}")
    print(f"Số lần lặp: {iterations}")

if __name__ == "__main__":
    main()