import numpy as np

def power_method(A, Y, E, max_iter=200):
    B = [Y.copy()]
    m = 1
    while m < max_iter:
        Z = np.dot(A, B[m-1])
        maxi = np.max(np.abs(Z))
        if maxi == 0:
            return None, None, "Vector Z bằng 0"
        B.append(Z / maxi)
        F = B[m] - B[m-1]
        if np.max(np.abs(F)) <= E:
            break
        m += 1
    if m >= max_iter:
        return None, None, "Không hội tụ sau {} lần lặp".format(max_iter)
    v = B[m]
    lambda_val = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)
    return lambda_val, v, f"Hội tụ sau {m} lần lặp"

def deflation(A, lambda_1, X_1):
    norm = np.sqrt(np.dot(X_1, X_1))
    if norm == 0:
        return None, "X_1 = 0"
    X_1 = X_1 / norm
    X1_X1T = np.outer(X_1, X_1)
    A_prime = A - lambda_1 * X1_X1T
    return A_prime, "Ma trận mới A' tính thành công"

def main(A, Y, E, max_iter, k):
    A_curr = A.copy()
    Y_curr = Y.copy()
    eigenvalues = []
    eigenvectors = []
    
    for i in range(k):
        lambda_i, X_i, message = power_method(A_curr, Y_curr, E, max_iter)
        if lambda_i is None:
            return None, None, "Thất bại: " + message
        eigenvalues.append(lambda_i)
        eigenvectors.append(X_i)
        print(f"\nLần {i+1}:")
        print("Trị riêng:", lambda_i)
        print("Vector riêng:", X_i)
        print("Message:", message)
        A_prime, message_deflation = deflation(A_curr, lambda_i, X_i)
        if A_prime is None:
            return None, None, "Thất bại: " + message_deflation
        print("Ma trận mới A':")
        print(A_prime)
        A_curr = A_prime
        Y_curr = np.random.rand(A.shape[0])  # Vector ngẫu nhiên
    
    return eigenvalues, eigenvectors, "Hoàn thành"

# Dữ liệu đầu vào
A = np.array([[2, 3, 2],
              [4, 3, 5],
              [3, 2, 9]], dtype=float)
Y = np.array([1, 1, 1], dtype=float)
E = 1e-6
max_iter = 200
k = 2

# Chạy thuật toán
eigenvalues, eigenvectors, message = main(A, Y, E, max_iter, k)
print("\nKết quả cuối cùng:")
print("Trị riêng:", eigenvalues)
print("Vector riêng:", eigenvectors)
print("Message:", message)