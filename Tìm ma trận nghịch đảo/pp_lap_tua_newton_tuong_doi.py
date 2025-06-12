import numpy as np

def read_input(file_path):
    """Đọc ma trận A từ file input.txt."""
    with open(file_path, 'r') as f:
        n = int(f.readline().strip())
        A = np.array([list(map(float, f.readline().strip().split())) for _ in range(n)])
    if A.shape != (n, n):
        raise ValueError("Ma trận A phải là ma trận vuông")
    return A

def matrix_norm_inf(A):
    """Tính chuẩn vô cực của ma trận."""
    return np.max(np.sum(np.abs(A), axis=1))

def matrix_norm_1(A):
    """Tính chuẩn 1 của ma trận."""
    return np.max(np.sum(np.abs(A), axis=0))

def choose_initial_guess(A):
    """Chọn ma trận ban đầu X0 = alpha * A^T."""
    norm_inf = matrix_norm_inf(A)
    norm_1 = matrix_norm_1(A)
    alpha = 1.0 / (norm_inf * norm_1)  # Đảm bảo ||I - A X0|| < 1
    X0 = alpha * A.T
    return X0

def newton_iteration(A, epsilon=1e-4, relative_error=False, max_iter=1000):
    """Tìm ma trận nghịch đảo bằng phương pháp lặp tựa Newton."""
    n = A.shape[0]
    I = np.eye(n)
    X = choose_initial_guess(A)
    delta = 1e-10  # Hằng số nhỏ để tránh chia cho 0
    
    # Kiểm tra điều kiện hội tụ
    convergence_norm = matrix_norm_inf(I - A @ X)
    if convergence_norm >= 1:
        print(f"Cảnh báo: Chuẩn ||I - A X0||_inf = {convergence_norm:.6f} >= 1, có thể không hội tụ")
    
    print("\nInitial X0:")
    np.set_printoptions(formatter={'float': '{:.5f}'.format})
    print(X)
    
    print(f"\n{'Lần lặp':<10}{'Ma trận X_k':<40}{'Sai số':<20}{'Ngưỡng':<20}")
    print("-" * 90)
    
    for k in range(max_iter):
        X_old = X.copy()
        # Công thức lặp Newton: X_{k+1} = X_k (2I - A X_k)
        X = X_old @ (2 * I - A @ X_old)
        
        # Tính sai số
        error = matrix_norm_inf(X - X_old)
        if relative_error:
            norm_X = matrix_norm_inf(X)
            error = error / (norm_X + delta) if norm_X != 0 else error
        
        # Định dạng ma trận X_k
        X_str = np.array2string(X, formatter={'float': '{:.5f}'.format}, separator='  ').replace('\n', '\n        ')
        print(f"{k+1:<10}{X_str:<40}{error:<20.6e}{epsilon:<20.6e}")
        
        if error < epsilon:
            print(f"Converged after {k+1} iterations")
            np.set_printoptions()
            return X
    
    print("Did not converge within max iterations")
    np.set_printoptions()
    return X

def main():
    try:
        # Đọc ma trận A từ file
        file_path = "input.txt"
        A = read_input(file_path)
        print("Ma trận A:")
        print(A)
        
        # Nhập thông tin từ người dùng
        error_type = input("Sử dụng sai số tương đối? (y/n): ").strip().lower() == 'y'
        epsilon = float(input("Nhập sai số epsilon (mặc định 1e-4): ") or 1e-4)
        
        # Tính ma trận nghịch đảo
        A_inv = newton_iteration(A, epsilon, error_type)
        
        # In kết quả
        print("\nMa trận nghịch đảo A^-1:")
        np.set_printoptions(formatter={'float': '{:.5f}'.format})
        print(A_inv)
        
        # Kiểm tra A * A^-1
        print("\nKiểm tra A * A^-1 (gần với ma trận đơn vị):")
        print(np.round(A @ A_inv, decimals=6))
        
    except FileNotFoundError:
        print(f"Không tìm thấy file {file_path}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    main()