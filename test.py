def find_initial_interval(a, n):
    """
    Xác định khoảng cách ly và giá trị ban đầu x0 dựa trên n và a.
    Trả về x0 (giá trị ban đầu).
    """
    if n <= 0:
        raise ValueError("Bậc n phải là số nguyên dương")
    
    # Trường hợp n chẵn
    if n % 2 == 0:
        if a <= 0:
            raise ValueError("Không thể tính căn bậc chẵn của số âm hoặc 0")
        elif 0 < a < 1:
            left, right = 0, 1
        else:  # a > 1
            left, right = 1, a
        x0 = (left + right) / 2
        print(f"Khoảng cách ly: ({left}, {right})")
        return x0
    
    # Trường hợp n lẻ
    else:
        if a > 0:
            k = 1
            while True:
                if k**n <= a <= (k + 1)**n:
                    left, right = k, k + 1
                    break
                elif a < 1:
                    left, right = 0, 1
                    break
                k += 1
        else:  # a < 0
            k = -1
            while True:
                if (k - 1)**n <= a <= k**n:
                    left, right = k - 1, k
                    break
                k -= 1
        x0 = (left + right) / 2
        print(f"Khoảng cách ly: ({left}, {right})")
        return x0

def newton_nth_root(a, n):
    """
    Tính căn bậc n của a bằng phương pháp tiếp tuyến (Newton-Raphson).
    a: Số cần tính căn
    n: Bậc của căn
    """
    # Tìm giá trị ban đầu từ khoảng cách ly
    x = find_initial_interval(a, n)
    epsilon = 1e-6  # Sai số mặc định
    iteration = 0
    
    print(f"Tính căn bậc {n} của {a}")
    print(f"Giá trị ban đầu: {x}")
    
    while True:
        # Tính x^(n-1) và x^n
        x_n_minus_1 = x ** (n - 1)
        x_n = x_n_minus_1 * x
        
        # Công thức Newton-Raphson
        x_new = ((n - 1) * x_n + a) / (n * x_n_minus_1)
        
        # Tính sai số
        error = abs(x_new - x)
        
        print(f"Lần lặp {iteration}: x = {x_new}, sai số = {error}")
        
        # Kiểm tra hội tụ
        if error < epsilon:
            print(f"Đã hội tụ sau {iteration + 1} lần lặp")
            return x_new
        
        x = x_new
        iteration += 1
        
        # Dừng an toàn
        if iteration > 10000:
            print("Dừng vì vượt quá 10000 lần lặp - có thể không hội tụ")
            return x

# Chương trình chính
def main():
    # Các trường hợp thử nghiệm
    test_cases = [(17, 5)]
    
    for a, n in test_cases:
        try:
            result = newton_nth_root(a, n)
            print(f"\nKết quả: Căn bậc {n} của {a} ≈ {result}")
            print(f"Kiểm tra: {result} ^ {n} = {result ** n}")
        except ValueError as e:
            print(f"Lỗi với a = {a}, n = {n}: {e}")
        print("-" * 50)

if __name__ == "__main__":
    main()