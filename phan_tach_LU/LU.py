import numpy as np

def read_matrix_from_file(filename):
    """
    Đọc ma trận A và B từ file.
    File có dạng: A trên cùng, phân cách bởi '---', rồi B.
    Chỉ trả về A cho phân tách LU.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Tìm dòng phân cách '---'
    separator_index = lines.index('---\n')

    # Đọc ma trận A
    A = np.array([list(map(float, line.split())) for line in lines[:separator_index]])
    return A

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

def check_lu_decomposition(A, L, U, tol=1e-10):
    """
    Kiểm tra: A ≈ L * U
    """
    A_reconstructed = np.dot(L, U)
    if np.allclose(A, A_reconstructed, atol=tol):
        print("Phân tách LU Crout đúng: A ≈ L * U")
    else:
        print("Phân tách LU Crout sai: A ≠ L * U")
        print("A:")
        print(A)
        print("L * U:")
        print(A_reconstructed)

# Đọc ma trận từ file
filename = 'matrix.txt'
try:
    A = read_matrix_from_file(filename)
    print("Ma trận A đọc từ file:")
    print(A)
    print("-" * 50)

    # Kiểm tra điều kiện
    check_lu_conditions(A)

    # Thực hiện phân tách LU Crout
    L, U = lu_decomposition_crout(A)
    print("\nKết quả cuối cùng:")
    print("Ma trận L:")
    print(np.round(L, 5))
    print("Ma trận U:")
    print(np.round(U,5))

    # Kiểm tra kết quả
    check_lu_decomposition(A, L, U)

except FileNotFoundError:
    print(f"Không tìm thấy file {filename}")
except ValueError as e:
    print(f"Lỗi: {e}")