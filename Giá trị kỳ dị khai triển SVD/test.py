import numpy as np

def svd_decomposition(A, tol=1e-6):
    m, n = A.shape
    B = A.T @ A  # Tính A^T A
    
    # Tìm giá trị riêng và vector riêng của A^T A
    eigenvalues, V = np.linalg.eigh(B)  # Sử dụng eigh vì A^T A đối xứng
    # Sắp xếp giảm dần
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Tính giá trị kỳ dị
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))  # Tránh căn của số âm nhỏ
    r = np.sum(singular_values > tol)  # Hạng của ma trận
    
    # Tính U
    U = np.zeros((m, m))
    for i in range(r):
        if singular_values[i] > tol:
            U[:, i] = (A @ V[:, i]) / singular_values[i]
    
    # Bổ sung các cột trực chuẩn cho U nếu cần
    if r < m:
        U[:, r:] = np.linalg.qr(np.eye(m) - U[:, :r] @ U[:, :r].T)[0][:, r:]
    
    # Tạo ma trận Sigma
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = singular_values[i]
    
    return U, Sigma, V.T

# Ví dụ
A = np.array([[2, 3, 2],
              [4, 3, 5],
              [3, 2, 9]], dtype=float)
U, Sigma, Vt = svd_decomposition(A)
print("Ma trận U:\n", U)
print("Ma trận Sigma:\n", Sigma)
print("Ma trận V^T:\n", Vt)

# Kiểm tra
print("Kiểm tra SVD: A ≈ U @ Sigma @ V^T\n", U @ Sigma @ Vt)