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
    return np.max(np.sum(np.abs(B), axis=0))

def norm_inf_matrix(B):
    return np.max(np.sum(np.abs(B), axis=1))

def norm_1_vector(v):
    return np.sum(np.abs(v))

def norm_inf_vector(v):
    return np.max(np.abs(v))

def simple_iteration(B, d, x0, num_iter=5):
    """
    Phương pháp lặp đơn, lặp đúng num_iter bước.
    Sau khi kết thúc, in ra sai số tuyệt đối giữa hai bước cuối.
    """
    norm_1 = norm_1_matrix(B)
    norm_inf = norm_inf_matrix(B)
    print(f"Chuẩn 1 của B: {norm_1:.6f}")
    print(f"Chuẩn vô cực của B: {norm_inf:.6f}")
    
    # Chọn chuẩn để tính sai số
    if norm_1 < 1:
        norm_type = "norm_1"
        norm_vector = norm_1_vector
        print("Chuẩn 1 < 1, sử dụng chuẩn 1.")
    elif norm_inf < 1:
        norm_type = "norm_inf"
        norm_vector = norm_inf_vector
        print("Chuẩn 1 >= 1 nhưng chuẩn vô cực < 1, sử dụng chuẩn vô cực.")
    else:
        print("Cả hai chuẩn >= 1, phương pháp không hội tụ. Thoát chương trình.")
        return None, 0

    x = x0.copy()
    print(f"{'Bước':<8}{'Nghiệm xấp xỉ x_n':<30}")
    print("-" * 40)

    for k in range(num_iter):
        x_new = np.dot(B, x) + d
        x_new_str = "[" + "  ".join(f"{val:.5f}" for val in x_new) + "]"
        print(f"{k+1:<8}{x_new_str:<30}")
        x_old = x
        x = x_new

    # Tính sai số tuyệt đối giữa hai bước cuối
    abs_error = norm_vector(x - x_old)
    print(f"\nSai số tuyệt đối giữa hai bước cuối: {abs_error:.8e}")
    return x, num_iter

# Chương trình chính
if __name__ == "__main__":
    file_path = "input.txt"
    try:
        B, d = read_input(file_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file {file_path}.")
        exit(1)
    
    epsilon = 1e-4
    m = B.shape[1]
    x0 = np.zeros(m)
    n = 5  # Số bước lặp cố định
    
    print(f"Ma trận B:\n{B}")
    print(f"Vector d: {d}")
    print(f"Số bước lặp cố định: {n}")
    print(f"Nghiệm ban đầu x0: {x0}\n")
    
    x, iterations = simple_iteration(B, d, x0, num_iter=n)
    if x is not None:
        x_str = "[" + "  ".join(f"{val:.5f}" for val in x) + "]"
        print(f"\nNghiệm cuối cùng sau {iterations} bước: {x_str}")
