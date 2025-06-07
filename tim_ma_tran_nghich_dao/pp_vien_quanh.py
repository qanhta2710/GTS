import numpy as np

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        matrix = []
        for _ in range(n):
            row = list(map(float, f.readline().strip().split()))
            matrix.append(row)
    return np.array(matrix)

def schur_complement_inverse(A):
    n = A.shape[0]
    if n == 1:
        if A[0, 0] == 0:
            raise ValueError("Matrix is not invertible (singular at 1x1 level).")
        return np.array([[1 / A[0, 0]]])

    # Initialize A_1^{-1} as the inverse of the 1x1 top-left element
    A_k_inv = np.array([[1 / A[0, 0]]])
    print("A_1^(-1) =")
    print(A_k_inv)
    print()

    for k in range(2, n + 1):
        # Extract blocks
        A_k_minus_1 = A[:k-1, :k-1]
        u_k = A[:k-1, k-1:k]
        v_k_T = A[k-1:k, :k-1]
        a_kk = A[k-1, k-1]

        # Compute x_k = A_{k-1}^(-1) * u_k
        x_k = A_k_inv @ u_k

        # Compute y_k = v_k^T * A_{k-1}^(-1)
        y_k = v_k_T @ A_k_inv

        # Compute theta_k = a_kk - y_k * u_k
        theta_k = a_kk - y_k @ u_k
        if abs(theta_k) < 1e-10:
            raise ValueError(f"Matrix is not invertible (singular at step k={k}).")

        # Compute theta_k^(-1)
        theta_k_inv = 1 / theta_k

        # Compute blocks for A_k^(-1)
        top_left = A_k_inv + x_k @ (theta_k_inv * y_k)
        top_right = -x_k * theta_k_inv
        bottom_left = -theta_k_inv * y_k
        bottom_right = np.array([[theta_k_inv]])

        # Construct A_k^(-1)
        A_k_inv = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])

        print(f"A_{k}^(-1) =")
        print(A_k_inv)
        print()

    return A_k_inv

def main():
    try:
        # Read matrix from input.txt
        A = read_matrix('input.txt')
        print("Input matrix A:")
        print(A)
        print()

        # Check if matrix is square
        if A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix must be square.")

        # Compute inverse using Schur complement
        A_inv = schur_complement_inverse(A)
        print("Final inverse matrix A^(-1):")
        print(A_inv)

        # Verify the result
        identity = A @ A_inv
        print("\nVerification (A * A^(-1) should be identity matrix):")
        print(np.round(identity, decimals=10))

    except FileNotFoundError:
        print("Error: input.txt not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()