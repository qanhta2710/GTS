#include <stdio.h>
#include <math.h>

double f(double x) {
    return pow(x, 5) - 3 * pow(x, 3) + 2 * pow(x, 2) - x + 5;  // Nhap phuong trinh f(x)
}

double f_derivative(double x) {
    return 5 * pow(x, 4) - 9 * pow(x, 2) + 4 * x - 1; // Nhap dao ham cua f(x)
}

// CAM DONG VAO DAY
void printSolution(double (*f)(double), double xk, double m1, double M2, double relative_error) {
    int k = 1;
    double tmp;
    printf("%3s %15s %15s %15s\n", "k", "xk", "|f(xk)|", "Error");
    printf("%3d %15.9lf %15.9lf\n", 0, xk, fabs(f(xk)));

    tmp = xk;
    xk = xk - ((f(xk)/f_derivative(xk)));
    printf("%3d %15.9lf %15.9e\n", k, xk, fabs(f(xk)));

    double currentError = ((M2 / 2 * m1) * pow(fabs(xk - tmp), 2)) / fabs(xk);
    while (currentError > relative_error) {
        k++;
        tmp = xk;
        xk = xk - ((f(xk)/f_derivative(xk)));
        currentError = ((M2 / 2 * m1) * pow(fabs(xk - tmp), 2)) / fabs(xk);
        printf("%3d %15.9lf %15.9e %15.15e\n", k, xk, fabs(f(xk)), currentError);
    }
}

int main() {
    double x0 = -2.5; // Nhap diem khoi tao x0
    double relative_error = 0.05/100; // Nhap sai so tuong doi
    double m1 = 35; // m1 = min(|f'(x)|)
    double M2 = 263.5; // M2 = max(|f''(x)|)
    printSolution(f, x0, m1, M2, relative_error);
    return 0;
}
