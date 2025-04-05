import numpy as np

# Định nghĩa hệ phương trình dưới dạng hàm g(x)
def g(x):
    x1, x2 = x
    # Ví dụ hệ phương trình:
    # x1 = cos(x2)
    # x2 = sin(x1)
    g1 = np.cos(x2)
    g2 = np.sin(x1)
    return np.array([g1, g2])

# Hàm giải bằng phương pháp lặp đơn với epsilon
def fixed_point_iteration(x0, epsilon):
    """
    x0: Nghiệm ban đầu (mảng numpy)
    epsilon: Sai số chấp nhận được
    """
    x = np.array(x0, dtype=float)
    iteration = 0
    
    while True:
        x_new = g(x)
        # Tính sai số
        error = np.max(np.abs(x_new - x))
        
        print(f"Lan lap {iteration}: x = {x_new}, sai so = {error}")
        
        # Kiểm tra điều kiện hội tụ
        if error < epsilon:
            print(f"Da hoi tu sau {iteration + 1} lan lap")
            return x_new
        
        x = x_new
        iteration += 1
        
        # Thêm điều kiện dừng an toàn (tránh vòng lặp vô hạn)
# Chương trình chính
def main():
    # Giá trị ban đầu
    x0 = np.array([0.5, 0.5])  # Có thể thay đổi giá trị khởi đầu
    epsilon = 1e-6  # Sai số mong muốn
    
    # Gọi hàm giải
    solution = fixed_point_iteration(x0, epsilon)
    
    print("\nNghiem can tim:", solution)

if __name__ == "__main__":
    main()