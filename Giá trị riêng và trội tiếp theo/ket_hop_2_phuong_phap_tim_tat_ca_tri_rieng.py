import numpy as np
import math

def power_method(A, Y, E, max_iter=200):
    np.set_printoptions(precision=6, suppress=True)
    B = [Y.copy()]
    for m in range(1, max_iter + 1):
        Z = np.dot(A, B[-1])
        print(f"\nBước lặp {m}:")
        print("  A * B[m-1] =", np.round(Z, 6))
        maxi = np.max(np.abs(Z))
        if maxi == 0:
            return None, None, "Vector Z bằng 0"
        B.append(Z / maxi)
        print("  B[m] (chuẩn hóa) =", np.round(B[-1], 6))
        print("  Sai số max |F| =", round(np.max(np.abs(B[-1] - B[-2])), 6))
        if np.max(np.abs(B[-1] - B[-2])) <= E or np.max(np.abs(B[-1] + B[-2])) <= E:
            v = B[-1]
            lambda_val = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)
            return lambda_val, v, f"Hội tụ sau {m} lần lặp"
    return None, None, f"Không hội tụ sau {max_iter} lần lặp"

def complex_eigen(A, Y, max_iter=200):
    """
    Xử lý trường hợp trị riêng phức bằng phương trình đặc trưng bậc hai.
    """
    np.set_printoptions(precision=6, suppress=True)
    print("\nGiải phương trình đặc trưng bậc hai từ 3 vector cuối:")
    print("  Am1Y =", np.round(Am1Y, 6))
    print("  AmY  =", np.round(AmY, 6))
    print("  M    =", np.round(M, 6))

    B = [Y.copy()]
    for _ in range(max_iter):
        Z = np.dot(A, B[-1])
        maxi = np.max(np.abs(Z))
        if maxi < 1e-10:
            return [], [], "Vector Z gần bằng 0"
        B.append(Z / maxi)
    
    # Lấy các vector cuối để xây dựng phương trình đặc trưng
    M = B[-1]
    AmY = np.dot(A, M)
    Am1Y = np.dot(A, AmY)

    # Tính hệ số phương trình đặc trưng bậc hai
    a_1, a_2 = Am1Y[0], Am1Y[1]
    b_1, b_2 = AmY[0], AmY[1]
    c_1, c_2 = M[0], M[1]

    denom = c_1 * b_2 - b_1 * c_2
    if abs(denom) < 1e-10:
        return [], [], "Mẫu số bằng 0 khi tính phương trình đặc trưng"

    a = 1
    b = (a_1 * c_2 - c_1 * a_2) / denom
    c = (b_1 * a_2 - a_1 * b_2) / denom
    delta = b**2 - 4 * a * c

    eigenvalues, eigenvectors = [], []

    if delta >= 0:
        lambda_1 = (-b + math.sqrt(delta)) / (2 * a)
        lambda_2 = (-b - math.sqrt(delta)) / (2 * a)
    else:
        sqrt_delta = math.sqrt(abs(delta)) / (2 * a)
        real_part = -b / (2 * a)
        lambda_1 = complex(real_part, sqrt_delta)
        lambda_2 = complex(real_part, -sqrt_delta)
    print(f"\n  Phương trình đặc trưng: λ² + ({round(b, 6)})λ + ({round(c, 6)}) = 0")
    if delta >= 0:
        print("  Δ =", round(delta, 6), "→ nghiệm thực")
    else:
        print("  Δ =", round(delta, 6), "→ nghiệm phức")

    print("  Trị riêng 1:", lambda_1)
    print("  Trị riêng 2:", lambda_2)

    for lam in [lambda_1, lambda_2]:
        v = Am1Y - lam * AmY
        v = v / np.max(np.abs(v))
        eigenvalues.append(lam)
        eigenvectors.append(v)

    msg = "Tìm được hai trị riêng thực" if delta >= 0 else "Tìm được hai trị riêng phức"
    return eigenvalues, eigenvectors, msg

def deflation(A, lambda_1, X_1):
    np.set_printoptions(precision=6, suppress=True)
    norm = np.linalg.norm(X_1)
    if norm == 0:
        return None, "X_1 = 0"
    X_1 = X_1 / norm
    A_prime = A - lambda_1 * np.outer(X_1, X_1)
    print(f"  Thực hiện triệt tiêu với λ = {round(lambda_1, 6)}")
    print("  Vector X chuẩn hóa:", np.round(X_1, 6))
    print("  Ma trận A' mới sau triệt tiêu:\n", np.round(A_prime, 6))
    return A_prime, "Ma trận mới A' tính thành công"

def main(A, Y, E, max_iter):
    n = A.shape[0]
    A_curr = A.copy()
    Y_curr = Y.copy()
    eigenvalues, eigenvectors = [], []
    
    i = 0
    while i < n:
        lambda_i, X_i, message = power_method(A_curr, Y_curr, E, max_iter)
        if lambda_i is None:
            complex_eigs, complex_vecs, complex_message = complex_eigen(A_curr, Y_curr, max_iter)
            if complex_eigs:
                eigenvalues.extend(complex_eigs)
                eigenvectors.extend(complex_vecs)
                print(f"\nTrị riêng phức tại lần {i+1} và {i+2}:")
                for j, (lam, vec) in enumerate(zip(complex_eigs, complex_vecs), start=i+1):
                    print(f"  Trị riêng {j}:", lam)
                    print(f"  Vector riêng {j}:", vec)
                print("Thông điệp:", complex_message)
                i += len(complex_eigs)
                break;
            else:
                return eigenvalues, eigenvectors, "Thất bại: " + message
        
        eigenvalues.append(lambda_i)
        eigenvectors.append(X_i)
        print(f"\nLần {i+1}:")
        print("  Trị riêng:", lambda_i)
        print("  Vector riêng:", X_i)
        print("  Thông điệp:", message)
        
        A_curr, msg_defl = deflation(A_curr, lambda_i, X_i)
        if A_curr is None:
            return eigenvalues, eigenvectors, "Thất bại: " + msg_defl
        print("  Ma trận mới A':\n", A_curr)
        
        Y_curr = np.random.rand(n)
        print(f"  Vector Y mới cho lần tiếp theo: {np.round(Y_curr, 6)}")
        i += 1

    return eigenvalues, eigenvectors, "Hoàn thành"

if __name__ == "__main__":
    try:
        with open("input.txt", "r") as f:
            n = int(f.readline())
            Y = np.array([float(i) for i in f.readline().split()])
            A = np.array([float(i) for i in f.read().split()]).reshape(n, n)

        print('Kích thước ma trận:', A.shape)
        print('Ma trận A:')
        print(A)
        print('_' * 100)

        E = 1e-6
        max_iter = 200
        eigenvalues, eigenvectors, message = main(A, Y, E, max_iter)

        print("\nKết quả cuối cùng:")
        print("Trị riêng:", eigenvalues)
        print("Vector riêng:")
        for vec in eigenvectors:
            print(vec)
        print("Thông điệp:", message)

    except FileNotFoundError:
        print("Không tìm thấy file input.txt")
    except ValueError:
        print("Dữ liệu trong file input.txt không hợp lệ")