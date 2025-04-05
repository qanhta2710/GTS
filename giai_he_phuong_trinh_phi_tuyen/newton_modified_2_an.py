import numpy as np

def F(x):
    F1 = x[0] * (1 - x[0]) + 4 * x[1] - 12
    F2 = (x[0] - 2)**2 + (2*x[1] - 3)**2 - 25
    return np.array([F1, F2])

def Jacobian(x0): 
    J = np.zeros((2,2))
    J[0,0] = 1 - 2*x0[0]
    J[0,1] = 4
    J[1,0] = 2*(x0[0] - 2)
    J[1,1] = 4*(2*x0[1] - 3)
    return J

def newton_modified(x0, n):
    x = np.array(x0, dtype=float)
    
    
    Jx = Jacobian(x)
    
    for i in range(n):
        Fx = F(x)
        delta_x = np.linalg.solve(Jx, -Fx) 
        x = x + delta_x 
        
        print(f"n = {i + 1}, x1 = {x[0]:.9f}, x2 = {x[1]:.9f}")

    return x

# Nhập số lần lặp từ người dùng
n = int(input("Nhap so lan lap: "))

# Giá trị khởi tạo
x0 = [0, 0]

# Chạy phương pháp Newton Modified
nghiem = newton_modified(x0, n)
