import numpy as np

def read_matrix_from_file(file_path):
    """Đọc ma trận từ file .txt"""
    try:
        matrix = []
        with open(file_path, 'r') as file:
            for line in file:
                row = [float(x) for x in line.strip().split()]
                matrix.append(row)
        matrix = np.array(matrix)
        # Kiểm tra ma trận có vuông không
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Ma trận không vuông")
        return matrix
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

def cholesky_decomposition(A):
    n = A.shape[0]
    U = np.zeros((n, n))
    
    print("Bắt đầu phân tách Cholesky (A = U^T U):")
    for i in range(n):
        print(f"\nBước {i+1}: Tính hàng {i+1} của ma trận U")
        
        # Tính phần tử đường chéo U_ii
        sum_squares = sum(U[k, i]**2 for k in range(i))
        if A[i, i] - sum_squares < 0:
            raise ValueError("Ma trận không dương xác định")
        U[i, i] = np.sqrt(A[i, i] - sum_squares)
        print(f"  - U[{i+1},{i+1}] = sqrt(A[{i+1},{i+1}] - sum(U[k,{i+1}]^2))")
        print(f"             = sqrt({A[i, i]:.5f} - {sum_squares:.5f})")
        print(f"             = {U[i, i]:.5f}")
        
        # Tính các phần tử bên phải đường chéo U_ij
        for j in range(i + 1, n):
            sum_products = sum(U[k, i] * U[k, j] for k in range(i))
            U[i, j] = (A[i, j] - sum_products) / U[i, i]
            print(f"  - U[{i+1},{j+1}] = (A[{i+1},{j+1}] - sum(U[k,{i+1}] * U[k,{j+1}])) / U[{i+1},{i+1}]")
            print(f"             = ({A[i, j]:.5f} - {sum_products:.5f}) / {U[i, i]:.5f}")
            print(f"             = {U[i, j]:.5f}")
        
        # In ma trận U hiện tại
        print("\n  Ma trận U sau khi tính hàng", i+1, ":")
        print(np.round(U, 5))
    
    return U

def main():
    # Đường dẫn file
    file_path = "matrix.txt"
    
    # Đọc ma trận từ file
    A = read_matrix_from_file(file_path)
    if A is None:
        return
    
    print("Ma trận A:")
    print(A)
    
    # Kiểm tra điều kiện
    if not is_symmetric(A):
        print("Lỗi: Ma trận không đối xứng")
        return
    
    # Thực hiện phân tách Cholesky
    try:
        U = cholesky_decomposition(A)
        print("\nMa trận U cuối cùng:")
        print(np.round(U, 5))
        
        # Kiểm tra kết quả: A = U^T * U
        A_reconstructed = np.dot(U.T, U)
        print("\nKiểm tra U^T * U:")
        print(np.round(A_reconstructed, 5))
        
        # So sánh với ma trận gốc
        if np.allclose(A, A_reconstructed, rtol=1e-5, atol=1e-8):
            print("\nPhân tách chính xác!")
        else:
            print("\nPhân tách không chính xác!")
            
    except ValueError as e:
        print(f"Lỗi trong quá trình phân tách: {e}")

if __name__ == "__main__":
    main()