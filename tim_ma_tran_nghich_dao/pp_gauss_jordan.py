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

def gauss_jordan_inverse(matrix, n):
    """Tìm ma trận nghịch đảo bằng phương pháp Gauss-Jordan."""
    # Tạo ma trận bổ sung [A | E]
    augmented = np.hstack((matrix, np.identity(n)))
    print("\nBước 1: Tạo ma trận bổ sung [A | E]:")
    print(augmented)
    
    # Biến đổi Gauss-Jordan
    for i in range(n):
        # Kiểm tra phần tử chéo chính
        if augmented[i, i] == 0:
            # Tìm hàng có phần tử khác 0 để hoán đổi
            for k in range(i + 1, n):
                if augmented[k, i] != 0:
                    augmented[[i, k]] = augmented[[k, i]]
                    print(f"\nBước {i+2}: Hoán đổi hàng {i+1} và hàng {k+1}:")
                    print(augmented)
                    break
            else:
                print("\nMa trận không khả nghịch (phần tử chéo chính bằng 0). Thoát chương trình.")
                return None
        
        # Chuẩn hóa hàng i (làm phần tử chéo chính thành 1)
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot
        print(f"\nBước {i+2}: Chuẩn hóa hàng {i+1} (chia cho {pivot}):")
        print(augmented)
        
        # Loại bỏ các phần tử khác trong cột i
        for j in range(n):
            if j != i:
                factor = augmented[j, i]
                augmented[j] = augmented[j] - factor * augmented[i]
                if factor != 0:
                    print(f"\nBước {i+2}.{j+1}: Khử cột {i+1} ở hàng {j+1} (R{j+1} = R{j+1} - {factor}*R{i+1}):")
                    print(augmented)
    
    # Kiểm tra xem phần trái có phải là ma trận đơn vị không
    left_side = augmented[:, :n]
    if not np.allclose(left_side, np.identity(n)):
        print("\nMa trận không khả nghịch. Thoát chương trình.")
        return None
    
    # Phần bên phải là ma trận nghịch đảo
    inverse = augmented[:, n:]
    print("\nBước cuối: Ma trận nghịch đảo A^-1:")
    print(inverse)
    return inverse

def main():
    file_path = 'input.txt'
    print("Bước 0: Đọc ma trận từ file input.txt")
    matrix, n = read_matrix(file_path)
    print("Ma trận A:")
    print(matrix)
    
    result = gauss_jordan_inverse(matrix, n)

if __name__ == "__main__":
    main()