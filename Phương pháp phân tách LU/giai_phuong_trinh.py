import numpy as np

def read_matrices_from_file(filename):
    """
    Đọc ma trận A và B từ file.
    File có dạng: A trên cùng, phân cách bởi '---', rồi B.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Tìm dòng phân cách '---'
    separator_index = lines.index('---\n')

    # Đọc ma trận A
    A = np.array([list(map(float, line.split())) for line in lines[:separator_index]])
    # Đọc ma trận B
    B = np.array([list(map(float, line.split())) for line in lines[separator_index+1:]])
    return A, B

def check_lu_conditions(A, tol=1e-10):
    """
    Kiểm tra điều kiện để phân tách LU:
    - Ma trận vuông
    - Tất cả định thức con chính khác 0
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Ma trận A không vuông: {n}x{m}")

    # Kiểm tra định thức các con chính
    for k in range(1, n + 1):
        det = np.linalg.det(A[:k, :k])
        if abs(det) < tol:
            raise ValueError(f"Định thức con chính thứ {k} quá nhỏ hoặc bằng 0: det = {det}")
    print("Tất cả điều kiện cho phân tách LU Crout được thỏa mãn.")

def lu_decomposition_crout(A, tol=1e-10):
    """
    Phân tách LU Crout: A = L * U
    L: ma trận tam giác dưới
    U: ma trận tam giác trên với u_ii = 1
    Hiển thị từng bước phân tách.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    np.fill_diagonal(U, 1.0)  # Đường chéo của U là 1

    print("Bắt đầu phân tách LU Crout:")
    print("Ma trận A:")
    print(A)
    print("-" * 50)

    # Duyệt qua từng cột t
    for t in range(n):
        print(f"\nBước t = {t}:")
        
        # Tính cột t của L (i >= t)
        print(f"  Tính cột {t} của L:")
        for i in range(t, n):
            sum_lu = sum(L[i, j] * U[j, t] for j in range(t))
            L[i, t] = A[i, t] - sum_lu
            print(f"    l_{i}{t} = a_{i}{t} - sum(l_{i}j * u_j{t}) = {A[i, t]} - {sum_lu} = {L[i, t]}")

        # Kiểm tra l_tt != 0
        if abs(L[t, t]) < tol:
            raise ValueError(f"Phân tách thất bại: l_{t}{t} = {L[t, t]} quá nhỏ")

        # Tính hàng t của U (k >= t+1, vì u_tt = 1)
        print(f"  Tính hàng {t} của U:")
        for k in range(t + 1, n):
            sum_lu = sum(L[t, j] * U[j, k] for j in range(t))
            U[t, k] = (A[t, k] - sum_lu) / L[t, t]
            print(f"    u_{t}{k} = (a_{t}{k} - sum(l_{t}j * u_j{k})) / l_{t}{t} = ({A[t, k]} - {sum_lu}) / {L[t, t]} = {U[t, k]}")

        # In trạng thái L và U
        print(f"  Ma trận L sau bước {t}:")
        print(L)
        print(f"  Ma trận U sau bước {t}:")
        print(U)
        print("-" * 50)

    return L, U

def solve_lu_crout(L, U, B, tol=1e-10):
    """
    Giải AX = B bằng phân tách LU Crout.
    Bước 1: Giải LY = B (thế xuôi)
    Bước 2: Giải UX = Y (thế ngược)
    """
    n = L.shape[0]
    m = B.shape[1]
    Y = np.zeros((n, m))
    X = np.zeros((n, m))

    print("\nGiải hệ phương trình AX = B:")
    print("Ma trận B:")
    print(B)
    print("-" * 50)

    # Thế xuôi: LY = B
    print("Bước 1: Giải LY = B (thế xuôi):")
    for k in range(m):  # Với mỗi cột của B
        print(f"  Cột {k} của Y:")
        for i in range(n):
            sum_ly = sum(L[i, j] * Y[j, k] for j in range(i))
            Y[i, k] = (B[i, k] - sum_ly) / L[i, i]
            print(f"    y_{i}{k} = (b_{i}{k} - sum(l_{i}j * y_j{k})) / l_{i}{i} = ({B[i, k]} - {sum_ly}) / {L[i, i]} = {Y[i, k]}")
    print("Ma trận Y:")
    print(Y)
    print("-" * 50)

    # Thế ngược: UX = Y
    print("Bước 2: Giải UX = Y (thế ngược):")
    for k in range(m):  # Với mỗi cột của Y
        print(f"  Cột {k} của X:")
        for i in range(n-1, -1, -1):
            sum_ux = sum(U[i, j] * X[j, k] for j in range(i+1, n))
            X[i, k] = Y[i, k] - sum_ux
            print(f"    x_{i}{k} = y_{i}{k} - sum(u_{i}j * x_j{k}) = {Y[i, k]} - {sum_ux} = {X[i, k]}")
    print("Ma trận X:")
    print(X)
    print("-" * 50)

    return X

def check_solution(A, X, B, tol=1e-10):
    """
    Kiểm tra: AX ≈ B
    """
    AX = np.dot(A, X)
    if np.allclose(AX, B, atol=tol):
        print("Nghiệm đúng: AX ≈ B")
    else:
        print("Nghiệm sai: AX ≠ B")
        print("AX:")
        print(AX)
        print("B:")
        print(B)

# Đọc ma trận từ file
filename = 'matrix.txt'
try:
    A, B = read_matrices_from_file(filename)
    print("Ma trận A đọc từ file:")
    print(A)
    print("Ma trận B đọc từ file:")
    print(B)
    print("-" * 50)

    # Kiểm tra điều kiện
    check_lu_conditions(A)

    # Thực hiện phân tách LU Crout
    L, U = lu_decomposition_crout(A)

    # Giải AX = B
    X = solve_lu_crout(L, U, B)

    # Kiểm tra kết quả
    print("\nKiểm tra nghiệm:")
    check_solution(A, X, B)

except FileNotFoundError:
    print(f"Không tìm thấy file {filename}")
except ValueError as e:
    print(f"Lỗi: {e}")