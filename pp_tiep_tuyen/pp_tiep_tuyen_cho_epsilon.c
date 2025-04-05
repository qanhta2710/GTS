#include <stdio.h>
#include <math.h>

double f(double x) { 
    return pow(x, 5) - 17; // Nhap phuong trinh f(x)
}

double f_derivative(double x) {
    return 5 * pow(x, 4); // Nhap dao ham cua f(x)
}

// CAM DONG VAO DAY
void printSolution(double (*f)(double), double xk, double m1, double M2, double epsilon) {
    int k = 1;
    double tmp;
    double tolerance = epsilon; // Tolerance for stopping condition based on consecutive approximations
    printf("%3s %15s %15s %15s\n", "k", "xk", "|f(xk)|", "|xk - xk-1|");
    printf("%3d %15.9lf %15.9lf %15s\n", 0, xk, fabs(f(xk)), "-");

    tmp = xk;
    xk = xk - (f(xk) / f_derivative(xk));
    printf("%3d %15.9lf %15.9e %15.9lf\n", k, xk, fabs(f(xk)), fabs(xk - tmp));

    while (pow(fabs(xk - tmp), 2) > 2 * epsilon * m1 / M2 && fabs(xk - tmp) > tolerance) {
        k++;
        tmp = xk;
        xk = xk - (f(xk) / f_derivative(xk));
        printf("%3d %15.9lf %15.9e %15.9lf\n", k, xk, fabs(f(xk)), fabs(xk - tmp));
    }
}

int main() {
    double x0 = 2; // Nhap diem khoi tao x0
    double epsilon = 0.5 * pow(10, -5); // Nhap sai so
    double m1 = 5; // m1 = min(|f'(x)|)
    double M2 = 160; // M2 = max(|f''(x)|)
    printSolution(f, x0, m1, M2, epsilon);
    return 0;
}