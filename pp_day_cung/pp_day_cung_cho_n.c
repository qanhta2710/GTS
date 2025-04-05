#include <stdio.h>
#include <math.h>

double f(double x) {
    return log(x) - 1; // Nhap phuong trinh f(x)
}

// CAM DONG VAO DAY
void printSolution(double (*f)(double), double xk, double d, double m1, double M1, int n) {
    int i = 0;
    double tmp = 0;
    printf("%3s %15s %15s\n", "k", "xk", "|f(xk)|");
    printf("%3d %15.9lf %15.9e\n", i, xk, fabs(f(xk)));
    for (i = 1; i <= n; i++) {
        tmp = xk;
        xk = xk - ((f(xk) * (xk - d)) / (f(xk) - f(d)));
        printf("%3d %15.9lf %15.15e\n", i, xk, fabs(f(xk)));
    }
    printf("Sai so tuyet doi: %.15e\n", ((M1 - m1) / m1) * fabs(xk - tmp));
}
int main() {
    double d = 2; // Nhap diem khoi tao d
    double x0 = 3; // Nhap diem khoi tao x0
    int n = 5; // Nhap so lan lap
    double m1 = 1.0/3.0; // m1 = min(f'(x))
    double M1 = 1.0/2.0; // M1 = max(f'(x))
    printSolution(f, x0, d, m1, M1, n);
    return 0;
}
