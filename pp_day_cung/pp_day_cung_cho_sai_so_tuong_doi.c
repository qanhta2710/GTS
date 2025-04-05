#include <stdio.h>
#include <math.h>

double f(double x) {
    return pow(x, 5) - 3 * pow(x, 3) + 2 * pow(x, 2) - x + 5; // Nhap phuong trinh f(x)
}

// CAM DONG VAO DAY
void printSolution(double (*f)(double), double xk, double d, double m1, double M1, double relative_error) {
    int k = 1;
    double tmp;
    printf("%3s %15s %15s %15s\n", "k", "xk", "|f(xk)|", "error");
    printf("%3d %15.9lf %15.9e\n", 0, xk, fabs(f(xk)));
    tmp = xk;
    xk = xk - ((f(xk) * (xk - d)) / (f(xk) - f(d)));
    printf("%3d %15.9lf %15.9e\n", k, xk, fabs(f(xk)));
    double currentError = (((M1 - m1) / m1) * fabs(xk - tmp)) / fabs(xk);
    while (currentError > relative_error) {
        k++;
        tmp = xk;
        xk = xk - ((f(xk) * (xk - d)) / (f(xk) - f(d)));
        currentError = (((M1 - m1) / m1) * fabs(xk - tmp)) / fabs(xk);
        printf("%3d %15.9lf %15.15e %15.15e\n", k, xk, fabs(f(xk)), currentError);
    }
}

int main() {
    double d = -2.5; // Nhap diem khoi tao d
    double x0 = -2; // Nhap diem khoi tao x0
    double relative_error = 0.05/100; // Nhap sai so tuong doi
    double m1 = 35; // m1 = min(|f'(x)|)
    double M1 = 128.0625; // M1 = max(|f'(x)|)
    printSolution(f, x0, d, m1, M1, relative_error);
    return 0;
}
