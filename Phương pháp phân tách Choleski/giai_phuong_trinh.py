import numpy as np

def read_A_B_from_file_with_separator(file_path):
    """
    Đọc ma trận A và vector/matrix B từ file có chứa dấu phân cách '---'.
    A là ma trận vuông n x n, B có thể là vector (n,) hoặc ma trận (n x m).
    """
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]

        if '---' not in lines:
            raise ValueError("Không tìm thấy dòng phân cách '---'")

        sep_index = lines.index('---')
        A_lines = lines[:sep_index]
        B_lines = lines[sep_index + 1:]

        A = np.array([list(map(float, row.split())) for row in A_lines])
        B = np.array([list(map(float, row.split())) for row in B_lines])

        if B.ndim == 1:
            B = B.reshape(-1, 1)

        return A, B

    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None, None


def is_symmetric(A):
    """Kiểm tra ma trận A có đối xứng không"""
    return np.array_equal(A, A.T)

def is_positive_definite(A):
    """Kiểm tra ma trận A có dương xác định không"""
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

def cholesky_decomposition(A):
    """Phân tách Cholesky: A = U^T U"""
    n = A.shape[0]
    U = np.zeros((n, n))
    
    for i in range(n):
        sum_squares = sum(U[k, i]**2 for k in range(i))
        if A[i, i] - sum_squares < 0:
            raise ValueError("Ma trận không dương xác định")
        U[i, i] = np.sqrt(A[i, i] - sum_squares)
        
        for j in range(i + 1, n):
            sum_products = sum(U[k, i] * U[k, j] for k in range(i))
            U[i, j] = (A[i, j] - sum_products) / U[i, i]
    
    return U

def solve_cholesky_detailed(A, B):
    """Giải hệ AX = B bằng Cholesky, in chi tiết giải xuôi và ngược"""
    # Phân tách Cholesky
    U = cholesky_decomposition(A)
    print("\nMa trận U:")
    print(np.round(U, 5))
    
    n, m = B.shape
    Y = np.zeros((n, m))
    X = np.zeros((n, m))
    
    # Giải U^T Y = B (giải xuôi)
    print("\nGiải U^T Y = B (giải xuôi):")
    for k in range(m):
        print(f"\nCột {k+1} của B: {B[:, k]}")
        for i in range(n):
            sum_terms = sum(U[j, i] * Y[j, k] for j in range(i))
            Y[i, k] = (B[i, k] - sum_terms) / U[i, i]
            print(f"  y[{i+1},{k+1}] = (b[{i+1},{k+1}] - sum(u[j,{i+1}] * y[j,{k+1}])) / u[{i+1},{i+1}]")
            print(f"             = ({B[i, k]:.5f} - {sum_terms:.5f}) / {U[i, i]:.5f} = {Y[i, k]:.5f}")
    print(f"Cột {k+1} của Y: {np.round(Y[:, k], 5)}")
    
    # Giải U X = Y (giải ngược)
    print("\nGiải U X = Y (giải ngược):")
    for k in range(m):
        for i in range(n-1, -1, -1):
            sum_terms = sum(U[i, j] * X[j, k] for j in range(i+1, n))
            X[i, k] = (Y[i, k] - sum_terms) / U[i, i]
            print(f"  x[{i+1},{k+1}] = (y[{i+1},{k+1}] - sum(u[{i+1},j] * x[j,{k+1}])) / u[{i+1},{i+1}]")
            print(f"             = ({Y[i, k]:.5f} - {sum_terms:.5f}) / {U[i, i]:.5f} = {X[i, k]:.5f}")
        print(f"Cột {k+1} của X: {np.round(X[:, k], 5)}")
    
    return X

def main():
    # Đường dẫn file
    file_path = "matrix.txt"
    
    # Đọc ma trận A và B
    A, B = read_A_B_from_file_with_separator('matrix.txt')
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if A is None or B is None:
        return
    
    print("Ma trận A:")
    print(A)
    print("\nMa trận B:")
    print(B)
    
    # Kiểm tra điều kiện
    if not is_symmetric(A):
        print("Lỗi: Ma trận A không đối xứng")
        return
    if not is_positive_definite(A):
        print("Lỗi: Ma trận A không dương xác định")
        return
    
    # Giải hệ phương trình
    try:
        X = solve_cholesky_detailed(A, B)
        print("\nMa trận nghiệm X:")
        print(np.round(X, 5))
        
        # Kiểm tra kết quả
        AX = np.dot(A, X)
        print("\nKiểm tra A * X:")
        print(AX)
        
        if np.allclose(AX, B, rtol=1e-5, atol=1e-8):
            print("\nNghiệm chính xác!")
        else:
            print("\nNghiệm không chính xác!")
            
    except ValueError as e:
        print(f"Lỗi trong quá trình giải: {e}")

if __name__ == "__main__":
    main()