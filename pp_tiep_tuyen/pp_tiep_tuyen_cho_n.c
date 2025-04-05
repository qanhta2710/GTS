#include <stdio.h>
#include <math.h>

double f(double x) {
    return pow(x, 4) - 27;  // Nhap phuong trinh f(x)
}

double f_derivative(double x) {
    return 4 * pow(x, 3); // Nhap f'(x)
}


void printSolution(double (*f)(double), double (*f_derivative)(double), double xk, double m1, double M2, int n) {
    int i = 0;
    double tmp = 0;
    printf("%3s %15s %15s\n", "k", "xk", "|f(xk)|");
    printf("%3d %15.9lf %15.9e\n", i, xk, fabs(f(xk)));
    for (i = 1; i <= n; i++) {
        tmp = xk;
        xk = xk - ((f(xk)/f_derivative(xk)));
        printf("%3d %15.9lf %15.9e\n", i, xk, fabs(f(xk)));
    }
    printf("Sai so tuyet doi: %.15e\n", (M2 / (2*m1)) * pow(fabs(xk - tmp), 2));
}
int main() {
    double x0 = 3; // Nhap diem khoi tao x0
    int n = 5; // Nhap so lan lap
    double m1 = 32; // m1 = min(|f'(x)|)
    double M2 = 108; // M2 = max(|f''(x)|)
    printSolution(f, f_derivative, x0, m1, M2, n);
    return 0;
}
