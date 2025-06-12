import numpy as np

def svd_decomposition(A, tol=1e-6):
    np.set_printoptions(precision=6, suppress=True)
    m, n = A.shape
    print("Ma trận A:")
    print(A)

    B = A.T @ A
    print("\nA^T * A =")
    print(B)
    
    # Tìm trị riêng và vector riêng của A^T A
    eigenvalues, V = np.linalg.eigh(B)
    print("\nGiá trị riêng ban đầu của A^T A:")
    print(eigenvalues)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    print("\nGiá trị riêng (sắp giảm dần):")
    print(eigenvalues)
    print("Vector riêng (V):")
    print(V)

    # Tính giá trị kỳ dị
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    print("\nGiá trị kỳ dị (singular values):")
    print(singular_values)

    r = np.sum(singular_values > tol)
    print(f"\nHạng r = {r} (với ngưỡng tol = {tol})")

    # Tính U
    U = np.zeros((m, m))
    print("\nTính các cột đầu của U từ A @ V / singular value:")
    for i in range(r):
        if singular_values[i] > tol:
            U[:, i] = (A @ V[:, i]) / singular_values[i]
            print(f"U[:, {i}] = (A @ V[:, {i}]) / {singular_values[i]:.6f} = {U[:, i]}")

    # Bổ sung cột trực chuẩn cho U
    if r < m:
        print("\nBổ sung các cột trực chuẩn còn lại cho U bằng Gram-Schmidt:")
        orthonormal_basis = [U[:, i] for i in range(r)]

        for i in range(m):
            candidate = np.zeros(m)
            candidate[i] = 1.0
            for v in orthonormal_basis:
                candidate -= np.dot(candidate, v) * v
            norm = np.linalg.norm(candidate)
            if norm > 1e-10:
                new_vec = candidate / norm
                orthonormal_basis.append(new_vec)
                print(f"Thêm vector trực chuẩn mới vào U: {np.round(new_vec, 6)}")
            if len(orthonormal_basis) == m:
                break

        for i in range(m):
            U[:, i] = orthonormal_basis[i]

    print("\nMa trận U:")
    print(np.round(U, 6))

    # Tạo Sigma
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = singular_values[i]

    print("\nMa trận Sigma:")
    print(np.round(Sigma, 6))

    Vt = V.T
    print("\nMa trận V^T:")
    print(np.round(Vt, 6))

    return U, Sigma, Vt

# Ví dụ
A = np.array([[1, 0],
              [0, 1],
              [1, 1]], dtype=float)

U, Sigma, Vt = svd_decomposition(A)

print("\n================ KIỂM TRA SVD =================")
print("A xấp xỉ U @ Sigma @ V^T:")
print(np.round(U @ Sigma @ Vt, 6))
print("A gốc:")
print(np.round(A, 6))
