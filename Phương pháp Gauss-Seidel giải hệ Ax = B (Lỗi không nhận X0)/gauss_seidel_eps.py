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

def gauss_seidel(A, B, epsilon, X0=None, max_iterations=1000):
    """
    Thuật toán Gauss-Seidel giải hệ phương trình AX = B, hiển thị Xn, sai số, q và s mỗi bước lặp.
    
    Parameters:
    A: Ma trận hệ số (n x n, numpy array).
    B: Ma trận hằng số (n x m, numpy array).
    epsilon: Sai số tuyệt đối.
    X0: Ma trận nghiệm ban đầu (n x m, numpy array), mặc định None.
    max_iterations: Số lần lặp tối đa.
    
    Returns:
    X: Nghiệm của hệ phương trình (n x m, numpy array).
    """
    n = A.shape[0]
    m = B.shape[1] if B.ndim == 2 else 1
    
    # Bước 1: Kiểm tra chéo trội hàng
    row_dominant = is_row_dominant(A)
    
    # Bước 2: Kiểm tra chéo trội cột nếu không chéo trội hàng
    if not row_dominant:
        if not is_column_dominant(A):
            raise ValueError("A không là ma trận chéo trội hàng và cột nên không thực hiện lặp Gauss-Seidel được")
    
    # Bước 3: Khởi tạo ma trận T = diag(1/a11, 1/a22, ..., 1/ann)
    A_diag = np.diag(A)
    if np.any(A_diag == 0):
        raise ValueError("Phần tử đường chéo Aii phải khác 0")
    T = np.diag(1.0 / A_diag)
    
    # Bước 4: Khởi tạo ma trận L và U
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j < i:
                L[i, j] = -A[i, j] / A[i, i]
            elif i < j:
                U[i, j] = -A[i, j] / A[i, i]
    
    # Bước 5: Khởi tạo ma trận D = TB
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    D = np.zeros((n, m))
    for i in range(n):
        D[i, :] = B[i, :] / A[i, i]
    
    # Bước 6: Khởi tạo X0
    if X0 is None:
        X = np.zeros((n, m))
        print("Không có X0 trong file, gán X0 = 0")
    else:
        X = X0.copy()
        if X.shape != (n, m):
            print("Kích thước X0 không khớp, gán X0 = 0")
            X = np.zeros((n, m))
    
    # In X0
    print("\nInitial X0:")
    np.set_printoptions(formatter={'float': '{:.5f}'.format})
    print(X)
    
    # Bước 7-8 hoặc 9-10: Tùy thuộc vào chéo trội hàng hay cột
    if row_dominant:
        # Bước 7: Tính s và q
        s = 0
        q_terms = []
        for i in range(n):
            lower_sum = sum(abs(A[i, j]) / abs(A[i, i]) for j in range(i))
            upper_sum = sum(abs(A[i, j]) / abs(A[i, i]) for j in range(i + 1, n))
            q_terms.append(lower_sum - upper_sum)
        q = max(q_terms)
        threshold = (1 - q) * epsilon / q
        
        print(f"\nParameters: q = {q:.5f}, s = {s:.5f}, Threshold = {threshold:.5f}")
        
        # Bước 8: Lặp với công thức Xn = L*Xn + U*Xn-1 + D
        I_minus_L = np.eye(n) - L
        for n_iter in range(max_iterations):
            X_old = X.copy()
            # Xn = (I - L)^(-1) * (U*Xn-1 + D)
            rhs = np.dot(U, X_old) + D
            X = np.linalg.solve(I_minus_L, rhs)
            
            # Tính sai số
            norm_inf = np.max(np.abs(X - X_old))
            
            # Hiển thị thông tin lặp
            print(f"\nIteration {n_iter + 1}:")
            print("Xn:")
            print(X)
            print(f"q: {q:.5f}, s: {s:.5f}")
            print(f"Error (||Xn - Xn-1||_inf): {norm_inf:.5f}")
            print(f"Threshold: {threshold:.5f}")
            
            # Kiểm tra điều kiện dừng
            if norm_inf <= threshold:
                print(f"Converged after {n_iter + 1} iterations")
                np.set_printoptions()
                return X
    else:
        # Bước 9: Tính s và q
        s_terms = []
        q_terms = []
        for j in range(n):
            lower_sum = sum(abs(A[i, j]) / abs(A[j, j]) for i in range(j))
            upper_sum = sum(abs(A[i, j]) / abs(A[j, j]) for i in range(j + 1, n))
            s_terms.append(upper_sum)
            q_terms.append(lower_sum - upper_sum)
        s = max(s_terms)
        q = max(q_terms)
        threshold = (1 - q) * (1 - s) * epsilon / q
        
        print(f"\nParameters: q = {q:.5f}, s = {s:.5f}, Threshold = {threshold:.5f}")
        
        # Bước 10: Lặp với công thức Xn = L*Xn + U*Xn-1 + D
        I_minus_L = np.eye(n) - L
        for n_iter in range(max_iterations):
            X_old = X.copy()
            # Xn = (I - L)^(-1) * (U*Xn-1 + D)
            rhs = np.dot(U, X_old) + D
            X = np.linalg.solve(I_minus_L, rhs)
            
            # Tính sai số
            norm_1 = np.sum(np.abs(X - X_old))
            
            # Hiển thị thông tin lặp
            print(f"\nIteration {n_iter + 1}:")
            print("Xn:")
            print(X)
            print(f"q: {q:.5f}, s: {s:.5f}")
            print(f"Error (||Xn - Xn-1||_1): {norm_1:.5f}")
            print(f"Threshold: {threshold:.5f}")
            
            # Kiểm tra điều kiện dừng
            if norm_1 <= threshold:
                print(f"Converged after {n_iter + 1} iterations")
                np.set_printoptions()
                return X
    
    print("Did not converge within max iterations")
    np.set_printoptions()
    return X

# Ví dụ sử dụng
if __name__ == "__main__":
    file_path = "input.txt"

    A, B, X0 = read_input(file_path)

    epsilon = 1e-6
    
    # Gọi hàm Gauss-Seidel
    solution = gauss_seidel(A, B, epsilon, X0)
    
    # In nghiệm cuối cùng
    print("\nFinal Solution:")
    np.set_printoptions(formatter={'float': '{:.5f}'.format})
    print(solution)
    np.set_printoptions()