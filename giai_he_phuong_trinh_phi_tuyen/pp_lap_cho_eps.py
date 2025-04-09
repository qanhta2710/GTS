import numpy as np

# Nhập các hàm f1, f2, f3 tương ứng với các phương trình trong hệ
def f1(x1, x2, x3):
    return (13 - x2**2 + 4 * x3) / 15

def f2(x1, x2, x3):
    return (11 + x3 - x1**2) / 10

def f3(x1, x2, x3):
    return (22 + x2**3) / 25

# Nhập ma trận Jacobi 
def jacobian(x1, x2, x3):
    df1_dx1 = 0
    df1_dx2 = -2 * x2 / 15
    df1_dx3 = 4 / 15

    df2_dx1 = -2 * x1 / 10
    df2_dx2 = 0
    df2_dx3 = 1 / 10

    df3_dx1 = 0
    df3_dx2 = 3 * x2**2 / 25
    df3_dx3 = 0

    J = np.array([[df1_dx1, df1_dx2, df1_dx3],
                  [df2_dx1, df2_dx2, df2_dx3],
                  [df3_dx1, df3_dx2, df3_dx3]])
    return J

def iterate_system(eps, q):
    # Nhập giá trị ban đầu
    x1, x2, x3 = 1.0, 1.0, 0.5

    # Điều kiện dừng
    stop_condition = eps * (1 - q) / q
    
    print(f"{'i':<6}{'x1':>15}{'x2':>15}{'x3':>15}{'Sai so':>15}")
    print(f"{0:<6}{x1:>15.8f}{x2:>15.8f}{x3:>15.8f}{0:>15.8f}")

    iteration = 1
    while True:
        x1_new = f1(x1, x2, x3)
        x2_new = f2(x1, x2, x3)
        x3_new = f3(x1, x2, x3)

        error = max(abs(x1_new - x1), abs(x2_new - x2), abs(x3_new - x3))

        print(f"{iteration:<6}{x1_new:>15.9f}{x2_new:>15.9f}{x3_new:>15.9f}{error:>20.9e}")

        if error < stop_condition:
            print(f"\nSai so {error:.9e} sau {iteration} buoc lap.")
            break

        x1, x2, x3 = x1_new, x2_new, x3_new
        iteration += 1

# Nhập giá trị eps và q
iterate_system(eps=0.5e-6, q=7.0/15.0)