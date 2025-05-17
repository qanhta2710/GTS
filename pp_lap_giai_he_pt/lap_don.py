import numpy as np

def read_input(file_path):
    """
    Đọc ma trận B và vector d từ file.
    """
    with open(file_path, 'r') as f:
        n, m = map(int, f.readline().split())
        B = np.zeros((n, m))
        for i in range(n):
            B[i] = list(map(float, f.readline().split()))
        d = np.zeros(n)
        for i in range(n):
            d[i] = float(f.readline().strip())
    return B, d

def norm_1_matrix(B):
    """
    Tính chuẩn 1 của ma trận B.
    """
    return np.max(np.sum(np.abs(B), axis=0))

def norm_inf_matrix(B):
    """
    Tính chuẩn vô cực của ma trận B.
    """
    return np.max(np.sum(np.abs(B), axis=1))

def norm_1_vector(v):
    """
    Tính chuẩn 1 của vector.
    """
    return np.sum(np.abs(v))

def norm_inf_vector(v):
    """
    Tính chuẩn vô cực của vector.
    """
    return np.max(np.abs(v))

def simple_iteration(B, d, epsilon, x0, max_iter=1000):
    """
    Phương pháp lặp đơn, kiểm tra sai số ||x_n - x_(n-1)|| < eps * (1 - ||B||) / ||B||.
    Bỏ qua tính sai số so với x*.
    """
    # Kiểm tra chuẩn
    norm_1 = norm_1_matrix(B)
    norm_inf = norm_inf_matrix(B)
    print(f"Chuẩn 1 của B: {norm_1:.6f}")
    print(f"Chuẩn vô cực của B: {norm_inf:.6f}")
    
    # Chọn chuẩn
    if norm_1 < 1:
        norm_type = "norm_1"
        norm_vector = norm_1_vector
        norm_B = norm_1
        print("Chuẩn 1 < 1, sử dụng chuẩn 1.")
    elif norm_inf < 1:
        norm_type = "norm_inf"
        norm_vector = norm_inf_vector
        norm_B = norm_inf
        print("Chuẩn 1 >= 1 nhưng chuẩn vô cực < 1, sử dụng chuẩn vô cực.")
    else:
        print("Cả hai chuẩn >= 1, phương pháp không hội tụ. Thoát chương trình.")
        return None, 0
    
    # Tính ngưỡng sai số
    error_threshold = epsilon * (1 - norm_B) / norm_B
    print(f"Ngưỡng sai số: {epsilon} * (1 - {norm_B:.6f}) / {norm_B:.6f} = {error_threshold:.8e}")
    
    # Khởi tạo lặp
    x = x0.copy()
    k = 0
    
    print(f"{'Bước':<8}{'Nghiệm xấp xỉ x_n':<30}{'Sai số ||x_n - x_(n-1)||':<25}{'Ngưỡng sai số':<25}")
    print("-" * 88)
    
    while k < max_iter:
        # Tính x_n = Bx_(n-1) + d
        x_new = np.dot(B, x) + d
        # Tính sai số ||x_n - x_(n-1)||
        error = norm_vector(x_new - x)
        # Định dạng nghiệm với 5 chữ số sau dấu phẩy
        x_new_str = "[" + "  ".join(f"{val:.5f}" for val in x_new) + "]"
        print(f"{k+1:<8}{x_new_str:<30}{error:<25.8e}{error_threshold:<25.8e}")
        
        if error < error_threshold:
            print(f"\nHội tụ sau {k+1} bước với nghiệm xấp xỉ: {x_new}")
            return x_new, k + 1
        
        x = x_new
        k += 1
    
    print(f"\nKhông hội tụ sau {max_iter} bước.")
    return None, k

# Chương trình chính
if __name__ == "__main__":
    file_path = "input.txt"
    try:
        B, d = read_input(file_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file {file_path}.")
        exit(1)
    
    epsilon = 1e-6
    m = B.shape[1]
    x0 = np.zeros(m)
    
    print(f"Ma trận B:\n{B}")
    print(f"Vector d: {d}")
    print(f"Sai số epsilon: {epsilon}")
    print(f"Nghiệm ban đầu x0: {x0}\n")
    
    x, iterations = simple_iteration(B, d, epsilon, x0)
    if x is not None:
        x_str = "[" + "  ".join(f"{val:.5f}" for val in x) + "]"
        print(f"\nNghiệm cuối cùng: {x_str}")
        print(f"Số bước lặp: {iterations}")