import numpy as np

def largest_singular_value(A, tol=1e-6, max_iter=200):
    m, n = A.shape
    B = A.T @ A  # Tính A^T A
    v = np.random.rand(n)  # Vector ngẫu nhiên
    v = v / np.linalg.norm(v)  # Chuẩn hóa
    
    for _ in range(max_iter):
        v_new = B @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-10:
            return None, "Vector bằng 0"
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    
    lambda_1 = v.T @ B @ v / (v.T @ v)  # Giá trị riêng lớn nhất
    sigma_1 = np.sqrt(lambda_1)  # Giá trị kỳ dị lớn nhất
    return sigma_1

# Ví dụ
A = np.array([[2, 3, 2],
              [4, 3, 5],
              [3, 2, 9]], dtype=float)
sigma_1 = largest_singular_value(A)
print("Giá trị kỳ dị lớn nhất:", sigma_1)

print(np.linalg.svd(A, compute_uv=False)[0])