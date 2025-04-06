#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Ham tim khoang cach ly va gia tri ban dau x0
double find_initial_interval(double a, int n) {
    if (n <= 0) {
        printf("Error: n must be a positive integer\n");
        exit(1);
    }

    double left, right, x0;

    // Truong hop n chan
    if (n % 2 == 0) {
        if (a <= 0) {
            printf("Invalid\n");
            exit(1);
        } else if (0 < a && a < 1) {
            double h = 1.0;
            while (1) {
                if (pow(h, n) < a && a < 1) {
                    left = h;
                    right = 1;
                    break;
                }
                h /= 2;
            }
            printf("Interval: (%lf, %lf)\n", left, right);
        } else { // a > 1
            int k = 1;
            while (1) {
                if (pow(k, n) < a && a < pow(k + 1, n)) {
                    left = k;
                    right = k + 1;
                    break;
                }
                k++;
            }
            printf("Interval: (%lf, %lf) - Case a > 1\n", left, right);
        }
        x0 = (left + right) / 2;
        return x0;
    }
    // Truong hop n le
    else {
        if (a > 0) {
            if (a < 1) {
                double h = 1.0;
                while (1) {
                    if (pow(h, n) < a && a < 1) {
                        left = h;
                        right = 1;
                        break;
                    }
                    h /= 2;
                }
                printf("Interval: (%lf, %lf)\n", left, right);
            } else { // a > 1
                int k = 1;
                while (1) {
                    if (pow(k, n) < a && a < pow(k + 1, n)) {
                        left = k;
                        right = k + 1;
                        break;
                    }
                    k++;
                }
                printf("Interval: (%lf, %lf)\n", left, right);
            }
            x0 = (left + right) / 2;
            return x0;
        } else { // a < 0
            double b = -a;
            if (0 < b && b < 1) {
                double h = 1.0;
                while (1) {
                    if (pow(h, n) < b && b < 1) {
                        left = h;
                        right = 1;
                        break;
                    }
                    h /= 2;
                }
                printf("Interval for nth root of %f: (%lf, %lf)\n", b, left, right);
            } else { // b > 1
                int k = 1;
                while (1) {
                    if (pow(k, n) < b && b < pow(k + 1, n)) {
                        left = k;
                        right = k + 1;
                        break;
                    }
                    k++;
                }
                printf("Interval for nth root of %lf: (%lf, %lf)\n", b, left, right);
            }
            x0 = (left + right) / 2;
            return x0;
        }
    }
}

// Ham tinh can bac n bang phuong phap Newton-Raphson
double newton_nth_root(double a, int n) {
    double x = find_initial_interval(a, n);
    double epsilon = 1e-6; // Sai so epsilon
    int i = 0;
    double b;

    // Neu a < 0 va n le, tinh b = -a
    if (a < 0 && n % 2 == 1) {
        b = -a;
    } else {
        b = a;
    }

    while (1) {
        // Giai phuong trinh x^n - b = 0 bang phuong phap Newton
        double f_x = pow(x, n) - b;         // f(x) = x^n - b
        double f_prime_x = n * pow(x, n - 1); // f'(x) = n * x^(n-1)
        double x_new = x - f_x / f_prime_x; // x_new = x - f(x)/f'(x)
        double error = fabs(x_new - x);     // Tinh sai so
        
        printf("Iteration %d: x = %.9lf, error = %.9e\n", i, x_new, error);
        
        if (error < epsilon) {
            // Neu a < 0 va n le thi tra ve -x_new
            if (a < 0 && n % 2 == 1) {
                return -x_new;
            } else {
                return x_new;
            }
        }
        x = x_new;
        i++;
    }
}

int main() {
    double a = -17;
    int n = 5;
    double result = newton_nth_root(a, n);
    printf("\nResult: nth root %d of %lf = %.9lf\n", n, a, result);
    return 0;
}