import numpy as np

def largest_singular_value(A, tol=1e-6, max_iter=200):
    m, n = A.shape
    B = A.T @ A  # Tính A^T A
    print("Ma trận A^T A (B):\n", np.round(B, 6))

    v = np.random.rand(n)  # Vector ngẫu nhiên
    v = v / np.linalg.norm(v)  # Chuẩn hóa ban đầu
    print("\nVector v bắt đầu (chuẩn hóa):", np.round(v, 6))

    for k in range(1, max_iter + 1):
        v_new = B @ v
        print(f"\nBước {k}:")
        print("  B @ v =", np.round(v_new, 6))

        norm = np.linalg.norm(v_new)
        if norm < 1e-10:
            print("  Vector gần bằng 0 → dừng.")
            return None, "Vector bằng 0"

        v_new = v_new / norm
        print("  v mới (chuẩn hóa) =", np.round(v_new, 6))
        print("  Sai số:", round(np.linalg.norm(v_new - v), 6))

        if np.linalg.norm(v_new - v) < tol:
            print("  → Hội tụ sau", k, "bước lặp.")
            v = v_new
            break

        v = v_new

    lambda_1 = v.T @ B @ v / (v.T @ v)  # Trị riêng lớn nhất
    sigma_1 = np.sqrt(lambda_1)  # Giá trị kỳ dị lớn nhất
    print("\nGiá trị riêng lớn nhất (λ₁):", round(lambda_1, 6))
    print("Giá trị kỳ dị lớn nhất (σ₁):", round(sigma_1, 6))
    return sigma_1

# Ví dụ
A = np.array([[2, 3, 2],
              [4, 3, 5],
              [3, 2, 9]], dtype=float)

print("Ma trận A:\n", A)
print("\n--- Tìm giá trị kỳ dị lớn nhất bằng phương pháp lũy thừa ---")
sigma_1 = largest_singular_value(A)
print("\nGiá trị kỳ dị lớn nhất:", sigma_1)
