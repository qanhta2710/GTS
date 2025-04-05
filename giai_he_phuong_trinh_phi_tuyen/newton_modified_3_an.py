import numpy as np

def F(x): # Nhap he phuong trinh can giai
    F1 = x[0] ** 2 + x[1] -37
    F2 = x[0] -x[1] ** 2 - 5
    F3 = x[0] + x[1] + x[2] -3
    return np.array([F1, F2, F3])

def Jacobian(x): 
    J = np.zeros((3,3))
    J[0,0] = 2 * x[0]
    J[0,1] = 1
    J[0,2] = 0

    J[1,0] = 1
    J[1,1] = -2 * x[1]
    J[1,2] = 0

    J[2,0] = 1
    J[2,1] = 1
    J[2,2] = 1
    return J

n = int(input("Nhap so lan lap: "))
def newton_modified(x0, n):
    x = np.array(x0, dtype=float)
    J = Jacobian(x)  
    for i in range(n):
        Fx = F(x)
        dx = np.linalg.solve(J, -Fx)
        x = x + dx
        print(f"n = {i + 1}, x1 = {x[0]:.9f}, x2 = {x[1]:.9f}, x3 = {x[2]:.9f}")
    return x


# Khởi tạo gần nghiệm
x0 = [6, 6, -9]
nghiem = newton_modified(x0, n)

