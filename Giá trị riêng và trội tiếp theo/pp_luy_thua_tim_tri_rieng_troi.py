import numpy as np

def power_method(A, Y, E, max_iter=200):
    """
    Tìm trị riêng trội và vector riêng tương ứng bằng phương pháp lũy thừa.
    
    Parameters:
    A (numpy.ndarray): Ma trận vuông n x n
    Y (numpy.ndarray): Vector ban đầu
    E (float): Sai số mong muốn
    max_iter (int): Số lần lặp tối đa
    
    Returns:
    tuple: (trị riêng trội λ, vector riêng X)
    """
    # Khởi tạo
    n = A.shape[0]
    B = [Y.copy()]  # Lưu vector Y vào B[0]
    m = 1

    # Lặp phương pháp lũy thừa
    while m < max_iter:
        # Tính A * B[m-1]
        Z = np.dot(A, B[m-1])
        
        # Chuẩn hóa Z
        maxi = np.max(np.abs(Z))
        if maxi == 0:  # Tránh chia cho 0
            return None, None, "Vector Z bằng 0"
        B.append(Z / maxi)
        
        # Kiểm tra hội tụ
        F = B[m] - B[m-1]
        if np.max(np.abs(F)) <= E:
            break
        
        m += 1
    
    # Kiểm tra hội tụ
    if m >= max_iter:
        return None, None, "Không hội tụ sau {} lần lặp".format(max_iter)
    
    # Tính trị riêng trội và vector riêng
    X = B[m]
    lambda_val = np.dot(X.T, np.dot(A, X)) / np.dot(X.T, X)  # Rayleigh quotient
    
    return lambda_val, X, "Hội tụ sau {} lần lặp".format(m)

# Ví dụ sử dụng
if __name__ == "__main__":
    A = np.array([[2, 3, 2],
              [4, 3, 5],
              [3, 2, 9]], dtype=float)
    
    # Vector ban đầu Y
    Y = np.array([1, 1, 1], dtype=float)
    
    # Sai số và số lần lặp tối đa
    E = 1e-6
    max_iter = 200
    
    # Chạy phương pháp lũy thừa
    eigenvalue, eigenvector, message = power_method(A, Y, E, max_iter)
    
    # In kết quả
    print("Message:", message)
    if eigenvalue is not None:
        print("Trị riêng trội:", eigenvalue)
        print("Vector riêng:", eigenvector)