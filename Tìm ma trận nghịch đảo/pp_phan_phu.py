import numpy as np

def read_matrix(file_path):
    """Đọc ma trận từ file input."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())
        matrix = []
        for line in lines[1:]:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
        return np.array(matrix), n

def minor_matrix(matrix, i, j):
    """Tạo ma trận con bằng cách xóa hàng i và cột j."""
    return np.delete(np.delete(matrix, i, axis=0), j, axis=1)

def determinant(matrix):
    """Tính định thức của ma trận."""
    if matrix.shape[0] == 1:
        return matrix[0, 0]
    if matrix.shape[0] == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    det = 0
    for j in range(matrix.shape[1]):
        det += ((-1) ** j) * matrix[0, j] * determinant(minor_matrix(matrix, 0, j))
    return det

def cofactor_matrix(matrix, n):
    """Tính ma trận phụ hợp."""
    cofactors = np.zeros((n, n))
    print("\nBước 2: Tính ma trận phụ hợp (Cofactor Matrix)")
    for i in range(n):
        for j in range(n):
            minor = minor_matrix(matrix, i, j)
            cofactors[i, j] = ((-1) ** (i + j)) * determinant(minor)
            print(f"  C_{i+1}{j+1} = (-1)^{i+1}+{j+1} * det(M_{i+1}{j+1}) = {cofactors[i, j]}")
    return cofactors

def inverse_matrix(matrix, n):
    """Tính ma trận nghịch đảo."""
    # Bước 1: Kiểm tra định thức
    det = determinant(matrix)
    print(f"\nBước 1: Tính định thức det(A) = {det}")
    if det == 0:
        print("Định thức bằng 0, ma trận không khả nghịch. Thoát chương trình.")
        return None
    
    # Bước 2: Tính ma trận phụ hợp
    cofactors = cofactor_matrix(matrix, n)
    
    # Bước 3: Chuyển vị ma trận phụ hợp
    adjugate = cofactors.T
    print("\nBước 3: Ma trận adjoint (chuyển vị của ma trận phụ hợp):")
    print(adjugate)
    
    # Bước 4: Tính ma trận nghịch đảo
    inverse = adjugate / det
    print("\nBước 4: Ma trận nghịch đảo A^-1 = (1/det(A)) * adj(A):")
    print(inverse)
    return inverse

def main():
    file_path = 'input.txt'
    print("Bước 0: Đọc ma trận từ file input.txt")
    matrix, n = read_matrix(file_path)
    print("Ma trận A:")
    print(matrix)
    
    result = inverse_matrix(matrix, n)

if __name__ == "__main__":
    main()