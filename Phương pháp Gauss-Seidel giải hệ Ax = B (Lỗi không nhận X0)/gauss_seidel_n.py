import numpy as np

def is_row_dominant(A):
    """Kiểm tra ma trận A có chéo trội hàng không."""
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off_diag_sum:
            return False
    return True

def is_column_dominant(A):
    """Kiểm tra ma trận A có chéo trội cột không."""
    n = A.shape[0]
    for j in range(n):
        diag = abs(A[j, j])
        off_diag_sum = sum(abs(A[i, j]) for i in range(n) if i != j)
        if diag <= off_diag_sum:
            return False
    return True

def read_input(file_path):
    """Đọc ma trận A, B và tùy chọn X0 từ file."""
    with open(file_path, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    
    # Đọc n và m
    n, m = map(int, lines[0])
    
    # Đọc ma trận A (n dòng)
    A = np.array([list(map(float, lines[i])) for i in range(1, n + 1)])
    
    # Đọc ma trận B (n dòng)
    B = np.array([list(map(float, lines[i])) for i in range(n + 1, 2 * n + 1)])
    
    # Kiểm tra kích thước
    if A.shape != (n, n) or B.shape != (n, m):
        raise ValueError("Kích thước ma trận không khớp với n, m")
    
    # Kiểm tra và đọc X0 nếu có
    X0 = None
    if len(lines) >= 2 * n + 1 + n:
        try:
            X0 = np.array([list(map(float, lines[i])) for i in range(2 * n + 1, 3 * n + 1)])
            if X0.shape != (n, m):
                print("Kích thước X0 không khớp, gán X0 = 0")
                X0 = None
        except:
            print("Không đọc được X0 từ file, gán X0 = 0")
            X0 = None
    
    return A, B, X0

def gauss_seidel_fixed_iterations(A, B, num_steps, X0=None):
    """
    Phiên bản Gauss-Seidel lặp đúng `num_steps` bước, in nghiệm và sai số tại bước cuối cùng.
    """
    n = A.shape[0]
    m = B.shape[1] if B.ndim == 2 else 1

    # Kiểm tra chéo trội
    row_dominant = is_row_dominant(A)
    if not row_dominant:
        if not is_column_dominant(A):
            raise ValueError("A không là ma trận chéo trội hàng hoặc cột")

    # Khởi tạo T, L, U
    A_diag = np.diag(A)
    if np.any(A_diag == 0):
        raise ValueError("Phần tử đường chéo Aii phải khác 0")
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j < i:
                L[i, j] = -A[i, j] / A[i, i]
            elif i < j:
                U[i, j] = -A[i, j] / A[i, i]

    # Chuẩn hóa B -> D
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    D = np.zeros((n, m))
    for i in range(n):
        D[i, :] = B[i, :] / A[i, i]

    # Khởi tạo X0
    if X0 is None or X0.shape != (n, m):
        print("Không có X0 hợp lệ, gán X0 = 0")
        X = np.zeros((n, m))
    else:
        X = X0.copy()

    # In X0
    print("\nInitial X0:")
    np.set_printoptions(formatter={'float': '{:.5f}'.format})
    print(X)

    # Chuẩn bị lặp
    I_minus_L = np.eye(n) - L

    print(f"\nThực hiện {num_steps} bước lặp Gauss-Seidel:")
    for iter in range(num_steps):
        X_old = X.copy()
        rhs = np.dot(U, X_old) + D
        X = np.linalg.solve(I_minus_L, rhs)

        print(f"\nIteration {iter + 1}:")
        print(X)

    # Tính sai số tuyệt đối tại bước cuối
    if row_dominant:
        abs_error = np.max(np.abs(X - X_old))
        print(f"\nSai số tuyệt đối cuối cùng (chuẩn ∞): {abs_error:.8e}")
    else:
        abs_error = np.sum(np.abs(X - X_old))
        print(f"\nSai số tuyệt đối cuối cùng (chuẩn 1): {abs_error:.8e}")

    return X


# Ví dụ sử dụng
if __name__ == "__main__":
    file_path = "input.txt"

    A, B, X0 = read_input(file_path)

    num_steps = 5
    
    # Gọi hàm Gauss-Seidel
    solution = gauss_seidel_fixed_iterations(A, B, num_steps, X0=None)
    
    # In nghiệm cuối cùng
    print("\nFinal Solution:")
    np.set_printoptions(formatter={'float': '{:.5f}'.format})
    print(solution)
    np.set_printoptions()