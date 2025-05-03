import numpy as np

# Các hàm f1, f2 tương ứng với các phương trình trong hệ
def f1(x1, x2):
    return (1/3) * np.sin(x1*x2) 

def f2(x1, x2):
    return (1/4) * np.cos(x1**2 + x2**2)

# Hàm tính ma trận Jacobian
def jacobian(x1, x2):
    df1_dx1 = (1/3) * x2 * np.cos(x1 * x2)
    df1_dx2 = (1/3) * x1 * np.cos(x1 * x2)

    df2_dx1 = - (1/4) * 2 * x1 * np.sin(x1 ** 2 + x2 ** 2)
    df2_dx2 = - (1/4) * 2 * x2 * np.sin(x1 ** 2 + x2 ** 2)

    J = np.array([[df1_dx1, df1_dx2],
                  [df2_dx1, df2_dx2]])
    return J

# Hàm kiểm tra điều kiện hội tụ
def check_convergence(x1, x2, tol = 1e-10):
    J = jacobian(x1, x2)
    
    # Kiểm tra ma trận Jacobian suy biến
    det_J = np.linalg.det(J)
    if abs(det_J) < tol:
        print("Ma tran Jacobian suy bien")
        return False, 0
    
    # Tính chuẩn hàng (q)
    q = np.max(np.sum(np.abs(J), axis=1))

    print(f"Chuan hang (q): {q:.6f}")

    # Kiểm tra điều kiện hội tụ
    if q < 1:
        print("Thoa man dieu kien hoi tu.")
        return True, q
    else:
        print("Khong thoa man dieu kien hoi tu.")
        return False, q

# Hàm lặp giải hệ phương trình phi tuyến
def iterate_system(relative_error):
    # Nhập giá trị ban đầu
    x1, x2 = 1, 1
    # Kiểm tra điều kiện hội tụ
    is_convergent, q = check_convergence(x1, x2)
    if not is_convergent:
        return

    print(f"{'i':<6}{'x1':>15}{'x2':>15}{'Sai so':>15}")
    print(f"{0:<6}{x1:>15.8f}{x2:>15.8f}{0:>15.8f}")

    iteration = 1
    while True:
        x1_new = f1(x1, x2)
        x2_new = f2(x1, x2)

        # Tính sai số tương đối
        error_x1 = abs(x1_new - x1) / abs(x1_new) if x1_new != 0 else float('inf')
        error_x2 = abs(x2_new - x2) / abs(x2_new) if x2_new != 0 else float('inf')

        error = max(error_x1, error_x2)

        print(f"{iteration:<6}{x1_new:>15.9f}{x2_new:>15.9f}{error:>20.9e}")

        if error < relative_error:
            print(f"\nSai so {error:.9e} sau {iteration} buoc lap.")
            break

        x1, x2 = x1_new, x2_new
        iteration += 1

# Nhập giá trị sai số tương đối
iterate_system(relative_error=0.01/100)