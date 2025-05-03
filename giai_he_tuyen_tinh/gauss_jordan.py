import numpy as np

def read_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Tìm vị trí dòng phân cách "---"
    separator_index = lines.index('---\n')

    # Đọc ma trận A
    A = np.array([list(map(float, line.split())) for line in lines[:separator_index]])

    # Đọc vector b
    b = np.array([float(line.strip()) for line in lines[separator_index + 1:]])

    return A, b

def gauss_jordan_elimination(A, b, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    m, n = A.shape
    Ab = np.hstack([A, b])  # Ma trận mở rộng

    print("Ma trận mở rộng ban đầu:")
    print(Ab)
    print("-" * 50)

    for i in range(min(m, n)):
        # Tìm pivot ưu tiên trị tuyệt đối bằng 1
        pivot_row = i
        for t in range(i, m):
            if abs(Ab[t, i] - 1) < tol:  # Ưu tiên trị tuyệt đối bằng 1
                pivot_row = t
                break
        else:
            # Nếu không có pivot trị tuyệt đối bằng 1, chọn phần tử lớn nhất
            pivot_row = np.argmax(np.abs(Ab[i:, i])) + i

        # Hoán đổi hàng nếu cần
        if pivot_row != i:
            Ab[[i, pivot_row]] = Ab[[pivot_row, i]]
            print(f"Hoán đổi hàng {i} và hàng {pivot_row}:")
            print(Ab)
            print("-" * 50)

        # Kiểm tra nếu pivot quá nhỏ (gần 0)
        if abs(Ab[i, i]) < tol:
            continue

        # Chuẩn hóa hàng hiện tại (pivot = 1)
        pivot = Ab[i, i]
        Ab[i, i:] /= pivot
        print(f"Chuẩn hóa hàng {i} (pivot = {pivot:.6f}):")
        print(Ab)
        print("-" * 50)

        # Khử các phần tử khác trong cột hiện tại
        for j in range(m):
            if j != i:
                factor = Ab[j, i]
                Ab[j, i:] -= factor * Ab[i, i:]
                print(f"Khử hàng {j} bằng hàng {i} với hệ số {factor:.6f}:")
                print(Ab)
                print("-" * 50)

    # Kiểm tra vô nghiệm
    for i in range(m):
        if np.all(np.abs(Ab[i, :-1]) < tol) and abs(Ab[i, -1]) > tol:
            print("Hệ phương trình vô nghiệm.")
            return {
                "type": "inconsistent",
                "Ab": Ab
            }

    # Kiểm tra vô số nghiệm
    rank = np.linalg.matrix_rank(Ab[:, :-1])
    if rank < n:
        print("Hệ có vô số nghiệm.")
        return {
            "type": "infinite",
            "Ab": Ab
        }

    # Hệ có nghiệm duy nhất
    x = Ab[:, -1]
    print("Nghiệm duy nhất:")
    print(x)

    return {
        "type": "unique",
        "solution": x,
        "Ab": Ab
    }

# Đọc ma trận từ file
filename = 'matrix.txt'
A, b = read_matrix_from_file(filename)

# Giải hệ phương trình bằng Gauss-Jordan
gauss_jordan_elimination(A, b)