import numpy as np

def power_method(A, Y, E, max_iter=200):
    np.set_printoptions(precision=6, suppress=True)
    B = [Y.copy()]
    for m in range(1, max_iter + 1):
        Z = np.dot(A, B[-1])
        print(f"\nBước lặp {m}:")
        print("  A * B[m-1] =", np.round(Z, 6))
        maxi = np.max(np.abs(Z))
        if maxi == 0:
            return None, None, "Vector Z bằng 0"
        B.append(Z / maxi)
        print("  B[m] (chuẩn hóa) =", np.round(B[-1], 6))
        print("  Sai số max |F| =", round(np.max(np.abs(B[-1] - B[-2])), 6))
        if np.max(np.abs(B[-1] - B[-2])) <= E or np.max(np.abs(B[-1] + B[-2])) <= E:
            v = B[-1]
            lambda_val = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)
            return lambda_val, v, f"Hội tụ sau {m} lần lặp"
    return None, None, f"Không hội tụ sau {max_iter} lần lặp"

def deflation(A, lambda_1, X_1):
    np.set_printoptions(precision=6, suppress=True)
    norm = np.linalg.norm(X_1)
    if norm == 0:
        return None, "X_1 = 0"
    X_1 = X_1 / norm
    A_prime = A - lambda_1 * np.outer(X_1, X_1)
    print(f"\n  Thực hiện triệt tiêu với λ = {round(lambda_1, 6)}")
    print("  Vector X chuẩn hóa:", np.round(X_1, 6))
    print("  Ma trận A' mới sau triệt tiêu:\n", np.round(A_prime, 6))
    return A_prime, "Ma trận mới A' tính thành công"

def compute_singular_values(A, Y, E, max_iter):
    n = A.shape[1]
    ATA = A.T @ A
    print("Ma trận A^T A:")
    print(np.round(ATA, 6))
    print("_" * 100)

    A_curr = ATA.copy()
    Y_curr = Y.copy()
    singular_values = []
    right_singular_vectors = []

    for i in range(n):
        print(f"\n=== Xuống thang trị riêng thứ {i+1} ===")
        lambda_i, X_i, message = power_method(A_curr, Y_curr, E, max_iter)
        if lambda_i is None:
            print("  Không tìm được trị riêng.")
            break
        print(f"\n  Trị riêng λ = {round(lambda_i, 6)}")
        print("  Vector riêng tương ứng:", np.round(X_i, 6))

        singular_value = np.sqrt(lambda_i) if lambda_i > 0 else 0.0
        singular_values.append(singular_value)
        right_singular_vectors.append(X_i)

        A_curr, _ = deflation(A_curr, lambda_i, X_i)
        Y_curr = np.random.rand(n)

    return singular_values, right_singular_vectors
A = np.array([[3, 3, 8, 2],
              [3, 9, 4, 10],
              [7, 10, 6, 9]], dtype=float)
Y = np.array([1, 1, 1, 1], dtype=float)
E = 1e-6
max_iter = 200

singular_values, right_singular_vectors = compute_singular_values(A, Y, E, max_iter)
print("\n=== Giá trị kỳ dị ===")
for i, sv in enumerate(singular_values, 1):
    print(f"σ{i} = {sv:.6f}")

