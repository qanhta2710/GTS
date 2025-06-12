import numpy as np

def power_method(A, Y, E, max_iter=200):
    """
    Tìm trị riêng trội và vector riêng tương ứng bằng phương pháp lũy thừa.
    """
    np.set_printoptions(precision=6, suppress=True)  # In 6 chữ số thập phân, không dùng e

    n = A.shape[0]
    B = [Y.copy()]
    m = 1

    print("Bước {:>3} | B[0] =", np.round(B[0], 6))

    while m < max_iter:
        Z = np.dot(A, B[m - 1])
        maxi = np.max(np.abs(Z))
        if maxi == 0:
            return None, None, "Vector Z bằng 0"

        B.append(Z / maxi)

        print("Bước {:>3} | A * B[{}] = {}".format(m, m - 1, np.round(Z, 6)))
        print("         | B[{}] (chuẩn hóa) = {}".format(m, np.round(B[m], 6)))
        
        F = B[m] - B[m - 1]
        sai_so = np.max(np.abs(F))
        print("         | Sai số max |F| = {:.6f}".format(sai_so))

        if sai_so <= E:
            break
        m += 1

    if m >= max_iter:
        return None, None, "Không hội tụ sau {} lần lặp".format(max_iter)

    X = B[m]
    lambda_val = np.dot(X.T, np.dot(A, X)) / np.dot(X.T, X)

    return lambda_val, X, "Hội tụ sau {} lần lặp".format(m)

# Ví dụ sử dụng
if __name__ == "__main__":
    A = np.array([[4.0327, 2.6090, 2.3283, 4.8132, 2.8724],
                  [2.6090, 3.6586, 4.6534, 3.5740, 3.9131],
                  [2.3283, 4.6534, 6.7322, 3.4631, 5.0275],
                  [4.8132, 3.5740, 3.4631, 6.8665, 3.1182],
                  [2.8724, 3.9131, 5.0275, 3.1182, 4.8099]], dtype=float)

    Y = np.array([1, 1, 1, 1, 1], dtype=float)
    E = 1e-6
    max_iter = 200

    eigenvalue, eigenvector, message = power_method(A, Y, E, max_iter)

    print("\nMessage:", message)
    if eigenvalue is not None:
        print("Trị riêng trội:", round(eigenvalue, 6))
        print("Vector riêng tương ứng:", np.round(eigenvector, 6))