import numpy as np

def read_matrix(file_path):
    """
    Đọc ma trận A từ file input.txt.
    """
    try:
        with open(file_path, 'r') as f:
            n = int(f.readline().strip())
            A = np.zeros((n, n))
            for i in range(n):
                A[i] = list(map(float, f.readline().strip().split()))
        return A
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file: {e}")

def matrix_norm_inf(M):
    """
    Tính chuẩn vô cực của ma trận.
    """
    return np.max(np.sum(np.abs(M), axis=1))

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

def newton_iteration_absolute_error(A, n=10):
    """
    Phương pháp lặp tựa Newton với số lần lặp cố định n.
    In ma trận X_k và sai số ước lượng sau mỗi bước lặp.
    """
    # Kiểm tra ma trận vuông
    if A.shape[0] != A.shape[1]:
        raise ValueError("Ma trận A phải là ma trận vuông")

    X = choose_initial_guess(A)
    print("Ma trận ban đầu X_0:")
    print(np.round(X, 6))

    # Tính G_0 = I - A X_0
    I = np.eye(A.shape[0])
    G_0 = I - A @ X
    q = matrix_norm_inf(G_0)
    print(f"Chuẩn q = ||I - A X_0||_inf = {q:.6f}")

    norm_X0 = matrix_norm_inf(X)
    print(f"||X_0||_inf = {norm_X0:.6f}\n")
    print(f"{'Bước':<8}{'Ma trận X_k':<40}{'Sai số ước lượng':<20}")
    print("-" * 68)

    for k in range(n):
    # Tính sai số ước lượng: ||X_0|| * q^(2^k) / (1 - q)
        error = norm_X0 * (q ** (2 ** k)) / (1 - q)
    # Định dạng ma trận X_k
        X_str = np.array2string(np.round(X, 6), separator='  ', suppress_small=True).replace('\n', '\n        ')
        print(f"{k:<8}{X_str:<40}{error:<20.6e}")

        # Cập nhật X_{k+1} = X_k (2I - A X_k)
        X = X @ (2 * I - A @ X)

    print(f"\nHoàn thành {n} bước lặp")
    print("Ma trận nghịch đảo cuối cùng X:")
    print(np.round(X, 6))
    return X, n

def main():
    try:
        # Đọc ma trận A từ file
        A = read_matrix('input.txt')
        print("Ma trận A:")
        print(A)

        # Thực hiện lặp Newton với số lần lặp cố định
        n = 10  # Số lần lặp cố định
        X, iterations = newton_iteration_absolute_error(A, n)
        
        # Kiểm tra kết quả
        I = np.eye(A.shape[0])
        verification = A @ X
        print("\nKiểm tra A @ X (gần ma trận đơn vị):")
        print(np.round(verification, 6))
        print(f"Số bước lặp: {iterations}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    main()