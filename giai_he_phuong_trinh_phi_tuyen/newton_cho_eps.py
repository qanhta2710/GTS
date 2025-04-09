import numpy as np

# Các hàm f1, f2, f3 tương ứng với các phương trình trong hệ
def f1(x1, x2, x3):
    return 15 * x1 + x2**2 - 4 * x3 - 13

def f2(x1, x2, x3):
    return 10 * x2 + x1**2 - x3 - 11

def f3(x1, x2, x3):
    return 25 * x3 - x2**3 - 22

# Hàm tính ma trận Jacobian
def jacobian(x1, x2, x3):
    df1_dx1 = 15
    df1_dx2 = 2 * x2
    df1_dx3 = -4

    df2_dx1 = 2 * x1
    df2_dx2 = 10
    df2_dx3 = -1

    df3_dx1 = 0
    df3_dx2 = -3 * x2**2
    df3_dx3 = 25

    J = np.array([[df1_dx1, df1_dx2, df1_dx3],
                  [df2_dx1, df2_dx2, df2_dx3],
                  [df3_dx1, df3_dx2, df3_dx3]])
    return J

def newton_method(eps):
    # Khởi tạo giá trị ban đầu
    x = np.array([1.0, 1.0, 0.5]) 

    print(f"{'i':<6}{'x1':>15}{'x2':>15}{'x3':>15}{'Sai so':>15}")
    print(f"{0:<6}{x[0]:>15.9f}{x[1]:>15.9f}{x[2]:>15.9f}{0:>20.9e}")

    iteration = 1
    while True:
        F = np.array([
            f1(x[0], x[1], x[2]),
            f2(x[0], x[1], x[2]),
            f3(x[0], x[1], x[2])
        ])

        J = jacobian(x[0], x[1], x[2])

        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            print("Ma tran Jacobi suy bien")
            break

        delta_x = np.linalg.solve(J, -F)

        x_new = x + delta_x

        # Tính sai số
        error = np.linalg.norm(delta_x, ord=np.inf)

        print(f"{iteration:<6}{x_new[0]:>15.9f}{x_new[1]:>15.9f}{x_new[2]:>15.9f}{error:>20.9e}")

        # Kiểm tra điều kiện dừng
        if error < eps:
            print(f"\nSai so: {error:.9e}")
            break

        x = x_new
        iteration += 1

# Gọi hàm với giá trị epsilon
newton_method(eps=0.5e-6)