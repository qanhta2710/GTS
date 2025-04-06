def find_initial_interval(a, n):
    if n <= 0:
        raise ValueError("Invalid")
    # n chan
    if n % 2 == 0:
        if a <= 0:
            raise ValueError("Invalid")
        elif 0 < a < 1:
            h = 1.0
            while True:
                if h**n < a < 1:
                    left, right = h, 1
                    break
                h /= 2
            print(f"Interval: ({left}, {right}) - Case 0 < a < 1")
        else:  # a > 1
            k = 1
            while True:
                if k**n < a < (k + 1)**n:
                    left, right = k, k + 1
                    break
                k += 1
            print(f"Interval: ({left}, {right}) - Case a > 1")
        x0 = (left + right) / 2
        return x0
    
    # n le 
    else:
        if a > 0:
            if a < 1:
                h = 1.0
                while True:
                    if h**n < a < 1:
                        left, right = h, 1
                        break
                    h /= 2
                print(f"Interval: ({left}, {right})")
            else:  # a > 1
                k = 1
                while True:
                    if k**n < a < (k + 1)**n:
                        left, right = k, k + 1
                        break
                    k += 1
                print(f"Interval: ({left}, {right})")
            x0 = (left + right) / 2
            return x0
        else:  # a < 0
            b = -a 
            if 0 < b < 1:
                h = 1.0
                while True:
                    if h**n < b < 1:
                        left, right = h, 1
                        break
                    h /= 2
                print(f"Interval for nth root of {b}: ({left}, {right})")
            else:  # b > 1
                k = 1
                while True:
                    if k**n < b < (k + 1)**n:
                        left, right = k, k + 1
                        break
                    k += 1
                print(f"Interval for nth root of {b}: ({left}, {right})")
            x0 = (left + right) / 2
            return x0

def newton_nth_root(a, n):
    x = find_initial_interval(a, n)
    epsilon = 1e-6  # Nhap sai so epsilon
    i = 0
    # Neu a < 0 va n le tinh b = -a
    if a < 0 and n % 2 == 1:
        b = -a
    else:
        b = a
    while True:
        # Giai phuong trinh x^n - b = 0 bang phuong phap Newton
        # f(x) = x^n - b
        f_x = x**n - b
        f_prime_x = n * x**(n - 1)
        x_new = x - f_x / f_prime_x
        error = abs(x_new - x)
        print(f"Iteration {i}: x = {x_new}, error = {error}")
        if error < epsilon:
            # neu a < 0 va n le thi tra ve -x_new
            if a < 0 and n % 2 == 1:
                return -x_new
            else:
                return x_new
        x = x_new
        i += 1

def main():
    a = -17 
    n = 5  
    
    try:
        result = newton_nth_root(a, n)
        print(f"\nResult: nth root {n} of {a} = {result}")
    except ValueError as e:
        print(f"Error with a = {a}, n = {n}: {e}")

if __name__ == "__main__":
    main()