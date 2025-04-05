import numpy as np

def F(x):
    F1 = 5 * x[0]**2 - x[1]**2
    F2 = x[1] - 0.25*(np.sin(x[0]) + np.cos(x[1]))
    return np.array([F1, F2])

def Jacobian(x): 
    J = np.zeros((2,2))
    J[0,0] = 10*x[0]
    J[0,1] =  -2*x[1]
    J[1,0] =  -0.25*np.cos(x[0])
    J[1,1] =  1 + 0.25*np.sin(x[1])
    return J

def newton_method(F, J, x0, max_iter=100):
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        
        try:
            delta_x = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print("Jacobian matrix is singular. Terminating iteration.")
            break
        x = x + delta_x
        
        print(f"n = {i + 1}, x1 = {x[0]:.9f}, x2 = {x[1]:.9f}")
    return x


# Giá trị khởi tạo
x0 = [1, 1]

# Nhập số lần lặp từ người dùng
max_iter = int(input("Nhap so lan lap: "))

# Giải hệ
solution = newton_method(F, Jacobian, x0, max_iter=max_iter)

